// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <opencv2/dnn/dnn.hpp>
#include "utils.hpp"
#include "unicode.hpp"
#include "core_bpe.hpp"
#include "core_gemma.hpp"
#include "core_unigram.hpp"
#include "core_wordpiece.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <functional>
#include <sstream>
#include <unordered_set>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using ImplRegistry = std::function<Ptr<Tokenizer::Impl>(const FileStorage& cfg, const std::string& dir)>;

static std::unordered_map<std::string, ImplRegistry>& tokenizerRegistry()
{
    static std::unordered_map<std::string, ImplRegistry> reg;
    return reg;
}

static CoreBPE buildTokenizerFromJson(cv::FileStorage& fs,
                          std::unordered_set<std::string>* outSpecial = nullptr);

static Ptr<Tokenizer::Impl> buildWordPieceTokenizerImpl(cv::FileStorage& fs, const std::string& dir);

// Opens a tokenizer.json file for FileStorage/JSON parsing while stripping
// out any oversized field values (currently just "precompiled_charsmap")
// that would otherwise overflow FileStorage's JSON parser. Pass a non-null
// outCharsmap to receive the stripped "precompiled_charsmap" value, if any.
static cv::FileStorage openTokenizerJson(const std::string& jsonPath,
    std::string* outCharsmap = nullptr)
{
    std::ifstream in(jsonPath, std::ios::binary);
    if (!in.is_open())
        CV_Error(cv::Error::StsError, "Failed to open tokenizer.json: " + jsonPath);

    std::ostringstream ss;
    ss << in.rdbuf();
    std::string text = ss.str();

    std::string charsmap = extractAndStripLongStringField(text, "precompiled_charsmap");
    if (outCharsmap)
        *outCharsmap = std::move(charsmap);

    cv::FileStorage fs(text, cv::FileStorage::MEMORY | cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!fs.isOpened())
        CV_Error(cv::Error::StsError, "Failed to parse tokenizer.json: " + jsonPath);
    return fs;
}

struct Tokenizer::Impl {
    virtual ~Impl() {}
    virtual std::vector<int> encode(const std::string& text) = 0;
    // Two-sequence encode, for models that support pairing a second sequence
    // (e.g. WordPiece: [CLS] A [SEP] B [SEP]). Default: no pairing support;
    // ignores an empty textPair (single-sequence call) and errors otherwise.
    virtual std::vector<int> encode(const std::string& text, const std::string& textPair) {
        if (!textPair.empty())
            CV_Error(cv::Error::StsNotImplemented, "This tokenizer does not support paired-sequence encoding");
        return encode(text);
    }
    virtual std::string decode(const std::vector<int>& tokens) = 0;
};

class BpeTokenizerImpl : public Tokenizer::Impl {
public:
    BpeTokenizerImpl(CoreBPE core, std::unordered_set<std::string> special = {});

    std::vector<int> encode(const std::string& text) override;
    std::string decode(const std::vector<int>& tokens) override;

private:
    Ptr<CoreBPE> coreBPE_;
    std::unordered_set<std::string> allowedSpecial_;
};

BpeTokenizerImpl::BpeTokenizerImpl(CoreBPE core, std::unordered_set<std::string> special)
    : coreBPE_(makePtr<CoreBPE>(std::move(core)))
    , allowedSpecial_(std::move(special)) {}

std::vector<int> BpeTokenizerImpl::encode(const std::string& text) {
    CV_Assert(coreBPE_);
    std::vector<uint32_t> tok = coreBPE_->encode(text, allowedSpecial_).first;
    return std::vector<int>(tok.begin(), tok.end());
}

std::string BpeTokenizerImpl::decode(const std::vector<int>& tokens) {
    CV_Assert(coreBPE_);
    std::vector<uint32_t> t32(tokens.begin(), tokens.end());
    const std::vector<std::uint8_t> optBytes = coreBPE_->decodeBytes(t32);
    if (optBytes.empty())
        CV_Error(cv::Error::StsError, "Invalid decode.");
    return std::string(reinterpret_cast<const char*>(optBytes.data()), optBytes.size());
}

class SentencePieceTokenizerImpl : public Tokenizer::Impl {
public:
    SentencePieceTokenizerImpl(CoreGemmaBPE model,
                                std::unordered_set<std::string> special = {},
                                int bos = -1);

    std::vector<int> encode(const std::string& text) override;
    std::string decode(const std::vector<int>& tokens) override;

private:
    CoreGemmaBPE model_;
    std::unordered_set<std::string> allowedSpecial_;
    int bosTokenId_;
};

SentencePieceTokenizerImpl::SentencePieceTokenizerImpl(CoreGemmaBPE model,
                                                        std::unordered_set<std::string> special,
                                                        int bos)
    : model_(std::move(model)), allowedSpecial_(std::move(special)), bosTokenId_(bos) {}

std::vector<int> SentencePieceTokenizerImpl::encode(const std::string& text) {
    std::vector<int> ids = model_.encode(text, allowedSpecial_);
    if (bosTokenId_ >= 0) {
        ids.insert(ids.begin(), bosTokenId_);
    }
    return ids;
}

std::string SentencePieceTokenizerImpl::decode(const std::vector<int>& tokens) {
    if (bosTokenId_ >= 0 && !tokens.empty() && tokens.front() == bosTokenId_) {
        std::vector<int> stripped(tokens.begin() + 1, tokens.end());
        return model_.decode(stripped);
    }
    return model_.decode(tokens);
}

// SentencePiece Unigram (T5-style): CoreUnigram already appends the trailing
// eos (via post_processor's TemplateProcessing) inside encode() and strips
// special tokens inside decode(), so this wrapper is a thin pass-through --
// unlike SentencePieceTokenizerImpl above, it must NOT re-add or re-strip
// anything itself.
class UnigramTokenizerImpl : public Tokenizer::Impl {
public:
    UnigramTokenizerImpl(CoreUnigram model, std::unordered_set<std::string> special = {});

    std::vector<int> encode(const std::string& text) override;
    std::string decode(const std::vector<int>& tokens) override;

private:
    CoreUnigram model_;
    std::unordered_set<std::string> allowedSpecial_;
};

UnigramTokenizerImpl::UnigramTokenizerImpl(CoreUnigram model, std::unordered_set<std::string> special)
    : model_(std::move(model)), allowedSpecial_(std::move(special)) {}

std::vector<int> UnigramTokenizerImpl::encode(const std::string& text) {
    return model_.encode(text, allowedSpecial_);
}

std::string UnigramTokenizerImpl::decode(const std::vector<int>& tokens) {
    return model_.decode(tokens);
}

// ASCII punctuation per the reference BasicTokenizer._run_split_on_punc /
// is_punctuation: treat all non-alnum ASCII printable chars as punctuation,
// same ranges as Python's chr(i) categories checked there.
static bool isBertAsciiPunct(uint32_t cpt)
{
    return (cpt >= 33 && cpt <= 47) || (cpt >= 58 && cpt <= 64) ||
           (cpt >= 91 && cpt <= 96) || (cpt >= 123 && cpt <= 126);
}

// CJK ranges per the reference BasicTokenizer._is_chinese_char.
static bool isBertChineseChar(uint32_t cpt)
{
    return (cpt >= 0x4E00 && cpt <= 0x9FFF) ||
           (cpt >= 0x3400 && cpt <= 0x4DBF) ||
           (cpt >= 0x20000 && cpt <= 0x2A6DF) ||
           (cpt >= 0x2A700 && cpt <= 0x2B73F) ||
           (cpt >= 0x2B740 && cpt <= 0x2B81F) ||
           (cpt >= 0x2B820 && cpt <= 0x2CEAF) ||
           (cpt >= 0xF900 && cpt <= 0xFAFF) ||
           (cpt >= 0x2F800 && cpt <= 0x2FA1F);
}

class WordPieceTokenizerImpl : public Tokenizer::Impl {
public:
    WordPieceTokenizerImpl(CoreWordPiece model,
                            bool cleanText,
                            bool handleChineseChars,
                            bool stripAccents,
                            bool lowercase,
                            int clsId,
                            int sepId,
                            std::unordered_map<std::string, int> specialToId);

    std::vector<int> encode(const std::string& text) override;
    std::vector<int> encode(const std::string& textA, const std::string& textB) override;
    std::string decode(const std::vector<int>& tokens) override;

private:
    std::string normalize(const std::string& text) const;

    // BertPreTokenizer: split on whitespace, and split off punctuation as its
    // own token (whitespace itself is already collapsed to ' ' by normalize()).
    static std::vector<std::string> preTokenize(const std::string& text);

    void encodeNormalized(const std::string& text, std::vector<int>& ids) const;

    // Splits on literal added/special-token matches first so they map to their
    // single vocab id instead of being punctuation-split and subword-tokenized.
    // Mirrors the special-token splitting in the BPE/Gemma/Unigram impls below.
    void encodeSegment(const std::string& text, std::vector<int>& ids) const;

    bool isSpecialId(int id) const;

    CoreWordPiece model_;
    bool cleanText_;
    bool handleChineseChars_;
    bool stripAccents_;
    bool lowercase_;
    int clsId_;
    int sepId_;
    // Added/special tokens from tokenizer.json's "added_tokens" (e.g. "[MASK]")
    // that must be recognized as literal whole tokens in the input text,
    // bypassing normalize()/preTokenize()/model_.encode() (see encodeSegment()).
    std::unordered_map<std::string, int> specialToId_;
};

WordPieceTokenizerImpl::WordPieceTokenizerImpl(CoreWordPiece model,
                                                bool cleanText,
                                                bool handleChineseChars,
                                                bool stripAccents,
                                                bool lowercase,
                                                int clsId,
                                                int sepId,
                                                std::unordered_map<std::string, int> specialToId)
    : model_(std::move(model)),
      cleanText_(cleanText),
      handleChineseChars_(handleChineseChars),
      stripAccents_(stripAccents),
      lowercase_(lowercase),
      clsId_(clsId),
      sepId_(sepId),
      specialToId_(std::move(specialToId))
{}

std::string WordPieceTokenizerImpl::normalize(const std::string& text) const {
    std::string out;
    out.reserve(text.size());
    for (uint32_t cpt : unicode_cpts_from_utf8(text)) {
        unicode_cpt_flags flags = unicode_cpt_flags_from_cpt(cpt);
        if (cleanText_ && (cpt == 0 || cpt == 0xFFFD || flags.is_control)) {
            continue;
        }
        if (cleanText_ && flags.is_whitespace) {
            out += ' ';
            continue;
        }
        if (handleChineseChars_ && isBertChineseChar(cpt)) {
            out += ' ';
            out += unicode_cpt_to_utf8(cpt);
            out += ' ';
            continue;
        }
        if (stripAccents_ && flags.is_accent_mark) {
            // Standalone combining mark (already-decomposed input): drop it.
            continue;
        }
        uint32_t effective = stripAccents_ ? unicode_strip_accent_base(cpt) : cpt;
        if (lowercase_) effective = unicode_tolower(effective);
        out += unicode_cpt_to_utf8(effective);
    }
    return out;
}

std::vector<std::string> WordPieceTokenizerImpl::preTokenize(const std::string& text) {
    std::vector<std::string> out;
    std::string cur;
    for (uint32_t cpt : unicode_cpts_from_utf8(text)) {
        unicode_cpt_flags flags = unicode_cpt_flags_from_cpt(cpt);
        bool isWs = flags.is_whitespace || cpt == ' ';
        bool isPunct = flags.is_punctuation || isBertAsciiPunct(cpt);
        if (isWs) {
            if (!cur.empty()) { out.push_back(cur); cur.clear(); }
            continue;
        }
        if (isPunct) {
            if (!cur.empty()) { out.push_back(cur); cur.clear(); }
            out.push_back(unicode_cpt_to_utf8(cpt));
            continue;
        }
        cur += unicode_cpt_to_utf8(cpt);
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

void WordPieceTokenizerImpl::encodeNormalized(const std::string& text, std::vector<int>& ids) const {
    for (const std::string& word : preTokenize(normalize(text))) {
        std::vector<int> wordIds = model_.encode(word);
        ids.insert(ids.end(), wordIds.begin(), wordIds.end());
    }
}

void WordPieceTokenizerImpl::encodeSegment(const std::string& text, std::vector<int>& ids) const {
    if (specialToId_.empty()) {
        encodeNormalized(text, ids);
        return;
    }
    size_t chunkStart = 0;
    size_t pos = 0;
    while (pos < text.size()) {
        std::string matched;
        int matchedId = -1;
        for (const auto& kv : specialToId_) {
            const std::string& sp = kv.first;
            if (sp.empty()) continue;
            if (pos + sp.size() > text.size()) continue;
            if (text.compare(pos, sp.size(), sp) != 0) continue;
            if (sp.size() > matched.size()) { matched = sp; matchedId = kv.second; }
        }
        if (matchedId >= 0) {
            if (pos > chunkStart)
                encodeNormalized(text.substr(chunkStart, pos - chunkStart), ids);
            ids.push_back(matchedId);
            pos += matched.size();
            chunkStart = pos;
        } else {
            ++pos;
        }
    }
    if (chunkStart < text.size())
        encodeNormalized(text.substr(chunkStart), ids);
}

std::vector<int> WordPieceTokenizerImpl::encode(const std::string& text) {
    return encode(text, std::string());
}

std::vector<int> WordPieceTokenizerImpl::encode(const std::string& textA, const std::string& textB) {
    std::vector<int> ids;
    if (clsId_ >= 0) ids.push_back(clsId_);
    encodeSegment(textA, ids);
    if (sepId_ >= 0) ids.push_back(sepId_);
    if (!textB.empty()) {
        encodeSegment(textB, ids);
        if (sepId_ >= 0) ids.push_back(sepId_);
    }
    return ids;
}

std::string WordPieceTokenizerImpl::decode(const std::vector<int>& tokens) {
    std::vector<int> filtered;
    filtered.reserve(tokens.size());
    for (int id : tokens) {
        if (id == clsId_ || id == sepId_) continue;
        if (isSpecialId(id)) continue;
        filtered.push_back(id);
    }
    return model_.decode(filtered);
}

bool WordPieceTokenizerImpl::isSpecialId(int id) const {
    return std::any_of(specialToId_.begin(), specialToId_.end(),
                        [id](const std::pair<const std::string, int>& kv) { return kv.second == id; });
}

static std::string expandCaseInsensitiveGroups(const std::string& in)
{
    std::string out;
    out.reserve(in.size());
    size_t i = 0;
    while (i < in.size()) {
        if (in.compare(i, 4, "(?i:") == 0) {
            size_t j = i + 4;
            int depth = 1;
            while (j < in.size() && depth > 0) {
                if (in[j] == '(') depth++;
                else if (in[j] == ')') depth--;
                if (depth > 0) j++;
            }
            std::string inner = in.substr(i + 4, j - (i + 4));
            out += "(?:";
            for (char c : inner) {
                if (std::isalpha(static_cast<unsigned char>(c))) {
                    out += '[';
                    out += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                    out += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                    out += ']';
                } else {
                    out += c;
                }
            }
            out += ")";
            i = (j < in.size()) ? j + 1 : j;
        } else {
            out += in[i++];
        }
    }
    return out;
}

static std::string stripPossessiveQuantifiers(const std::string& in)
{
    std::string out;
    out.reserve(in.size());
    for (char c : in) {
        if (c == '+' && !out.empty()) {
            char prev = out.back();
            if (prev == '+' || prev == '*' || prev == '?') {
                continue;
            }
            if (prev == '}') {
                // '}' can either close a {m,n} repetition quantifier (in which
                // case a following '+' is a possessive quantifier that should
                // be stripped) or close a \p{...}/\P{...} Unicode property
                // escape (in which case the following '+' is a normal,
                // legitimate quantifier applied to the whole \p{...} atom and
                // must NOT be stripped). Walk back to the matching '{' to
                // tell these two cases apart.
                int depth = 1;
                size_t j = out.size() - 1;
                while (j > 0 && depth > 0) {
                    --j;
                    if (out[j] == '}') ++depth;
                    else if (out[j] == '{') --depth;
                }
                bool isPropertyEscape = (depth == 0 && out[j] == '{' &&
                    j >= 2 && out[j - 1] == 'p' && out[j - 2] == '\\');
                if (!isPropertyEscape) {
                    isPropertyEscape = (depth == 0 && out[j] == '{' &&
                        j >= 2 && out[j - 1] == 'P' && out[j - 2] == '\\');
                }
                if (depth == 0 && out[j] == '{' && !isPropertyEscape) {
                    continue;
                }
            }
        }
        out.push_back(c);
    }
    return out;
}

static std::string adaptHfPreTokenizerRegex(const std::string& raw)
{
    return stripPossessiveQuantifiers(expandCaseInsensitiveGroups(raw));
}

static bool findEmbeddedSplitRegex(const cv::FileNode& preTok, std::string& outRegex)
{
    if (preTok.empty()) return false;
    std::string type;
    preTok["type"] >> type;
    if (type == "Sequence") {
        cv::FileNode list = preTok["pretokenizers"];
        for (auto it = list.begin(); it != list.end(); ++it) {
            cv::FileNode child = *it;
            cv::FileNode regexNode = child["pattern"]["Regex"];
            if (!regexNode.empty() && regexNode.isString()) {
                regexNode >> outRegex;
                return true;
            }
        }
        return false;
    }
    cv::FileNode regexNode = preTok["pattern"]["Regex"];
    if (!regexNode.empty() && regexNode.isString()) {
        regexNode >> outRegex;
        return true;
    }
    return false;
}

static std::string detectSplitPattern(const cv::FileStorage& fs)
{
    std::string raw;
    if (findEmbeddedSplitRegex(fs["pre_tokenizer"], raw))
        return adaptHfPreTokenizerRegex(raw);
    return R50K_UTF8;
}

static Ptr<Tokenizer::Impl> buildSentencePieceTokenizerImpl(
        cv::FileStorage& fs,
        std::unordered_set<std::string>* outSpecial = nullptr)
{

    cv::FileNode modelNode = fs["model"];
    CV_CheckFalse(modelNode.empty(), "tokenizer.json missing 'model'");

    std::string modelType;
    modelNode["type"] >> modelType;
    if (!modelType.empty() && modelType != "BPE")
        CV_Error(cv::Error::StsError,
            "Expected a byte-fallback BPE model in tokenizer.json, got: " + modelType);

    CoreGemmaBPE gemma;

    cv::FileNode vocabNode = modelNode["vocab"];
    CV_CheckFalse(vocabNode.empty(), "tokenizer.json model missing 'vocab'");

    int maxId = -1;
    for (auto it = vocabNode.begin(); it != vocabNode.end(); ++it) {
        cv::FileNode entry = *it;
        std::string piece = entry.name();
        int id = (int)entry;
        if (id > maxId) maxId = id;
        gemma.pieceToId[piece] = id;
    }

    gemma.idToPiece.resize(maxId + 1);
    for (const auto& kv : gemma.pieceToId)
        gemma.idToPiece[kv.second] = kv.first;

    bool mergesAreStringFormat = false;
    cv::FileNode mergesNode = modelNode["merges"];
    if (!mergesNode.empty()) {
        cv::FileNode firstEntry = *mergesNode.begin();
        mergesAreStringFormat = firstEntry.isString();

        uint32_t rank = 0;
        for (auto it = mergesNode.begin(); it != mergesNode.end(); ++it) {
            cv::FileNode entry = *it;
            std::string a, b;
            if (entry.isString()) {
                std::string mergeStr;
                entry >> mergeStr;
                size_t sp = mergeStr.find(' ');
                if (sp == std::string::npos) {
                    ++rank;
                    continue;
                }
                a = mergeStr.substr(0, sp);
                b = mergeStr.substr(sp + 1);
            } else if (entry.size() == 2) {
                entry[0] >> a;
                entry[1] >> b;
            } else {
                ++rank;
                continue;
            }
            gemma.addMerge(a, b, rank);
            ++rank;
        }
    }

    std::unordered_set<std::string> special;
    int bosId = -1;
    cv::FileNode added = fs["added_tokens"];
    if (!added.empty()) {
        for (auto it = added.begin(); it != added.end(); ++it) {
            cv::FileNode t = *it;
            int id = -1;         t["id"]      >> id;
            std::string content; t["content"] >> content;
            if (id >= 0 && !content.empty()) {
                gemma.specialToId[content] = id;
                gemma.idToSpecial[id]      = content;
                special.insert(content);
                if (outSpecial) outSpecial->insert(content);
                if (content == "<bos>") bosId = id;
            }
        }
    }

    int bosTokenId = (mergesAreStringFormat && bosId >= 0) ? bosId : -1;
    return makePtr<SentencePieceTokenizerImpl>(std::move(gemma), std::move(special), bosTokenId);
}

static Ptr<Tokenizer::Impl> buildUnigramTokenizerImpl(
        cv::FileStorage& fs, const std::string& charsmapB64,
        std::unordered_set<std::string>* outSpecial = nullptr)
{

    cv::FileNode modelNode = fs["model"];
    CV_CheckFalse(modelNode.empty(), "tokenizer.json missing 'model'");

    std::string modelType;
    modelNode["type"] >> modelType;
    if (!modelType.empty() && modelType != "Unigram")
        CV_Error(cv::Error::StsError,
            "Expected a Unigram model in tokenizer.json, got: " + modelType);

    cv::FileNode vocabNode = modelNode["vocab"];
    CV_CheckFalse(vocabNode.empty(), "tokenizer.json model missing 'vocab'");

    std::vector<std::pair<std::string, float>> vocab;
    vocab.reserve(vocabNode.size());
    for (auto it = vocabNode.begin(); it != vocabNode.end(); ++it) {
        cv::FileNode entry = *it;
        std::string piece;
        double score = 0.0;
        if (entry.size() >= 2) {
            entry[0] >> piece;
            entry[1] >> score;
        }
        vocab.emplace_back(piece, (float)score);
    }

    int unkId = -1;
    modelNode["unk_id"] >> unkId;

    std::unordered_set<std::string> special;
    std::unordered_map<std::string, int> specialToId;
    int eosId = -1;
    cv::FileNode added = fs["added_tokens"];
    if (!added.empty()) {
        for (auto it = added.begin(); it != added.end(); ++it) {
            cv::FileNode t = *it;
            bool isSpecial = false; t["special"] >> isSpecial;
            int tid = -1;           t["id"]      >> tid;
            std::string content;    t["content"] >> content;
            if (isSpecial && tid >= 0 && !content.empty()) {
                specialToId[content] = tid;
                special.insert(content);
                if (outSpecial) outSpecial->insert(content);
                if (content == "</s>") eosId = tid;
            }
        }
    }

    CoreUnigram unigram(vocab, unkId, buildUnigramPrecompiledNormalizer(charsmapB64), specialToId, eosId);

    return makePtr<UnigramTokenizerImpl>(std::move(unigram), std::move(special));
}

static Ptr<Tokenizer::Impl> buildWordPieceTokenizerImpl(cv::FileStorage& fs, const std::string& /*dir*/)
{
    cv::FileNode modelNode = fs["model"];
    CV_CheckFalse(modelNode.empty(), "tokenizer.json missing 'model'");

    std::string modelType;
    modelNode["type"] >> modelType;
    if (!modelType.empty() && modelType != "WordPiece")
        CV_Error(cv::Error::StsError,
            "Expected a WordPiece model in tokenizer.json, got: " + modelType);

    cv::FileNode vocabNode = modelNode["vocab"];
    CV_CheckFalse(vocabNode.empty(), "tokenizer.json model missing 'vocab'");

    std::string unkToken = "[UNK]";
    modelNode["unk_token"] >> unkToken;
    std::string continuingSubwordPrefix = "##";
    modelNode["continuing_subword_prefix"] >> continuingSubwordPrefix;
    int maxInputCharsPerWord = 100;
    modelNode["max_input_chars_per_word"] >> maxInputCharsPerWord;

    std::unordered_map<std::string, int> vocab;
    for (auto it = vocabNode.begin(); it != vocabNode.end(); ++it) {
        cv::FileNode entry = *it;
        vocab[entry.name()] = (int)entry;
    }

    CoreWordPiece core(vocab, unkToken, continuingSubwordPrefix,
                        (size_t)std::max(1, maxInputCharsPerWord));

    // See WordPieceTokenizerImpl::specialToId_ / encodeSegment() for how these
    // are used to bypass the normal split.
    std::unordered_map<std::string, int> specialToId;
    cv::FileNode added = fs["added_tokens"];
    if (!added.empty()) {
        for (auto it = added.begin(); it != added.end(); ++it) {
            cv::FileNode t = *it;
            bool isSpecial = false; t["special"] >> isSpecial;
            int tid = -1;           t["id"]      >> tid;
            std::string content;    t["content"] >> content;
            if (isSpecial && tid >= 0 && !content.empty())
                specialToId[content] = tid;
        }
    }

    // Only a flat (non-Sequence) BertNormalizer node is read. A present-but-null
    // "strip_accents" is HF's encoding for "follow lowercase" -- it must resolve
    // to the current "lowercase" value, not false, to match the reference tokenizer.
    bool cleanText = true, handleChineseChars = true, lowercase = true, stripAccents;
    cv::FileNode normNode = fs["normalizer"];
    if (!normNode.empty()) {
        normNode["clean_text"] >> cleanText;
        normNode["handle_chinese_chars"] >> handleChineseChars;
        normNode["lowercase"] >> lowercase;
        cv::FileNode stripAccentsNode = normNode["strip_accents"];
        if (!stripAccentsNode.empty())
            stripAccentsNode >> stripAccents;
        else
            stripAccents = lowercase;
    } else {
        stripAccents = lowercase;
    }

    int clsId = -1, sepId = -1;
    core.tryGetId("[CLS]", clsId);
    core.tryGetId("[SEP]", sepId);

    return makePtr<WordPieceTokenizerImpl>(std::move(core), cleanText, handleChineseChars,
                                            stripAccents, lowercase, clsId, sepId,
                                            std::move(specialToId));
}

static Ptr<Tokenizer::Impl> buildBPETokenizerImpl(cv::FileStorage& fs)
{
    std::unordered_set<std::string> special;
    CoreBPE core = buildTokenizerFromJson(fs, &special);
    return makePtr<BpeTokenizerImpl>(std::move(core), std::move(special));
}

static Ptr<Tokenizer::Impl> buildFromTokenizerDir(const std::string& dir)
{
    std::string tokJson = dir + "tokenizer.json";
    std::string charsmapB64;
    cv::FileStorage fs = openTokenizerJson(tokJson, &charsmapB64);

    cv::FileNode model = fs["model"];
    if (model.empty())
        CV_Error(cv::Error::StsError,
            "tokenizer.json has no 'model' field; raw rank-table tokenizers are not "
            "supported by this loader: " + tokJson);

    std::string modelType;
    model["type"] >> modelType;

    // Some older HF tokenizer.json snapshots omit the model "type" field
    // entirely. Detect that case via each model kind's distinctive schema
    // instead of falling through to the BPE path below and erroring out on
    // the missing 'merges' table.
    bool looksLikeUnigram = modelType == "Unigram" ||
        (modelType.empty() &&
         !model["unk_id"].empty() &&
         model["unk_token"].empty() &&
         model["merges"].empty());
    if (looksLikeUnigram)
        return buildUnigramTokenizerImpl(fs, charsmapB64);

    bool looksLikeWordPiece = modelType == "WordPiece" ||
        (modelType.empty() &&
         !model["unk_token"].empty() &&
         !model["continuing_subword_prefix"].empty() &&
         model["merges"].empty());
    if (looksLikeWordPiece)
        return buildWordPieceTokenizerImpl(fs, dir);

    bool byteFallback = false;
    model["byte_fallback"] >> byteFallback;
    bool looksLikeSentencePiece = byteFallback &&
        (modelType == "BPE" || modelType.empty());
    if (looksLikeSentencePiece)
        return buildSentencePieceTokenizerImpl(fs);

    if (!modelType.empty() && modelType != "BPE")
        CV_Error(cv::Error::StsError,
            "Unsupported tokenizer model type '" + modelType + "' in " + tokJson +
            " (only BPE-family, Unigram and WordPiece models are currently supported)");

    if (model["merges"].empty())
        CV_Error(cv::Error::StsError,
            "tokenizer.json model has no 'merges' table in " + tokJson +
            " (only merge-based BPE models are currently supported)");

    return buildBPETokenizerImpl(fs);
}

static Ptr<Tokenizer::Impl> buildDefaultTokenizerImpl(const FileStorage& /*cfg*/, const std::string& dir)
{
    return buildFromTokenizerDir(dir);
}

static void registerDefaultTokenizers()
{
    auto& reg = tokenizerRegistry();
    static const char* const kFamilies[] = { "BPE", "Gemma", "SentencePiece", "Unigram", "WordPiece" };
    for (const char* family : kFamilies) {
        if (reg.find(family) == reg.end())
            reg[family] = buildDefaultTokenizerImpl;
    }
}

Tokenizer::Tokenizer() : impl_(nullptr) {}

std::vector<int> Tokenizer::encode(const std::string& text, const std::string& textPair)
{
    if (!impl_) CV_Error(cv::Error::StsError, "Tokenizer impl null");
    return impl_->encode(text, textPair);
}

std::string Tokenizer::decode(const std::vector<int>& tokens)
{
    if (!impl_) CV_Error(cv::Error::StsError, "Tokenizer impl null");
    return impl_->decode(tokens);
}

static std::vector<uint8_t> tokenToBytes(const std::string& tokenUtf8)
{
    std::vector<std::uint8_t> out;
    auto cps = unicode_cpts_from_utf8(tokenUtf8);
    out.reserve(cps.size());
    for (uint32_t cp : cps) {
        const std::string one = unicode_cpt_to_utf8(cp);
        out.push_back(unicode_utf8_to_byte(one));
    }
    return out;
}

static CoreBPE buildTokenizerFromJson(cv::FileStorage& fs,
                          std::unordered_set<std::string>* outSpecial)
{
    cv::FileNode model = fs["model"];
    CV_CheckFalse(model.empty(), "tokenizer.json missing 'model'");
    cv::FileNode vocab = model["vocab"];
    CV_CheckFalse(vocab.empty(), "tokenizer.json missing model.vocab");

    std::string modelType;
    model["type"] >> modelType;
    if (!modelType.empty() && modelType != "BPE")
        CV_Error(cv::Error::StsError,
            "Expected a BPE model in tokenizer.json, got: " + modelType);

    std::string pattern = detectSplitPattern(fs);

    std::unordered_set<std::string> skipTokens;
    FileNode addedPeek = fs["added_tokens"];
    if (!addedPeek.empty()) {
        for (auto it = addedPeek.begin(); it != addedPeek.end(); ++it) {
            cv::FileNode t = *it;
            bool isSpecial = false; t["special"]  >> isSpecial;
            std::string content;     t["content"]  >> content;
            if (isSpecial && !content.empty())
                skipTokens.insert(content);
        }
    }

    ByteVecRankMap mergeableRanks;
    mergeableRanks.reserve((size_t)vocab.size());
    int maxId = -1;

    for (cv::FileNodeIterator it = vocab.begin(); it != vocab.end(); ++it) {
        FileNode val = *it;
        std::string token = val.name();
        if (skipTokens.find(token) != skipTokens.end()) continue;
        int id = (int)val;
        mergeableRanks.emplace(tokenToBytes(token), (uint32_t)id);
        if (id > maxId) maxId = id;
    }

    std::unordered_map<std::string, uint32_t> specialTokens;
    FileNode added = fs["added_tokens"];
    if (!added.empty()) {
        for (auto it = added.begin(); it != added.end(); ++it) {
            cv::FileNode t = *it;
            bool special = false; t["special"] >> special;
            int id = -1;          t["id"]      >> id;
            std::string content;  t["content"] >> content;
            if (special && id >= 0 && !content.empty()) {
                specialTokens.emplace(content, (uint32_t)id);
                if (id > maxId) maxId = id;
                if (outSpecial) outSpecial->insert(content);
            }
        }
    }

    return CoreBPE(std::move(mergeableRanks), std::move(specialTokens), pattern);
}

Tokenizer Tokenizer::load(const std::string& modelConfig)
{
    cv::FileStorage cfg(modelConfig, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!cfg.isOpened())
        CV_Error(cv::Error::StsError, "Could not open config.json: " + modelConfig);

    std::string dir = modelConfig;
    size_t pos = dir.find_last_of("/\\");
    dir = (pos == std::string::npos) ? std::string() : dir.substr(0, pos + 1);

    std::string methodType = "BPE";
    if (!cfg["method"].empty())
        cfg["method"] >> methodType;

    registerDefaultTokenizers();
    auto& reg = tokenizerRegistry();
    auto it = reg.find(methodType);
    if (it == reg.end())
        CV_Error(cv::Error::StsError,
            "Unsupported tokenizer method: '" + methodType + "'. Supported: BPE, Gemma, SentencePiece, Unigram, WordPiece");

    Tokenizer tok;
    tok.impl_ = it->second(cfg, dir);
    return tok;
}

CV__DNN_INLINE_NS_END
}}
