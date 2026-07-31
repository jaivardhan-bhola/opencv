// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_DNN_TOKENIZER_CORE_WORDPIECE_HPP__
#define __OPENCV_DNN_TOKENIZER_CORE_WORDPIECE_HPP__

#include "unicode.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

// Vocab lookup + greedy longest-match-first split, word -> ids only.
// No [CLS]/[SEP]/segment-id assembly here -- that's WordPieceTokenizerImpl's
// job (tokenizer.cpp).
struct CoreWordPiece {
    std::unordered_map<std::string, int> pieceToId;
    std::vector<std::string> idToPiece;

    std::string unkToken = "[UNK]";
    int unkId = 0;
    std::string continuingSubwordPrefix = "##";
    size_t maxInputCharsPerWord = 100;
    size_t maxPieceCps = 0;

    void addPiece(const std::string& piece, int id) {
        if (id < 0)
            CV_Error(cv::Error::StsBadArg, "WordPiece vocab entry '" + piece + "' has a negative id: " + std::to_string(id));
        pieceToId[piece] = id;
        if ((size_t)id >= idToPiece.size())
            idToPiece.resize(id + 1);
        idToPiece[id] = piece;
        maxPieceCps = std::max(maxPieceCps, unicode_cpts_from_utf8(piece).size());
    }

    // Greedy longest-match-first split of one pre-tokenized word into ids.
    // All-or-nothing: on any no-match position, discards partial ids and
    // emits a single unkId for the whole word -- matches reference
    // BertTokenizer/WordpieceTokenizer, not a bug.
    void encodeWord(const std::string& word, std::vector<int>& out) const {
        std::vector<uint32_t> cps = unicode_cpts_from_utf8(word);
        if (cps.size() > maxInputCharsPerWord) {
            out.push_back(unkId);
            return;
        }

        // Byte offsets let candidates be substr()'d out of `word` directly;
        // rebuilding each candidate cp-by-cp via unicode_cpt_to_utf8() instead
        // makes this function O(L^3) in the word length.
        std::vector<size_t> byteOffset(cps.size() + 1);
        size_t off = 0;
        for (size_t i = 0; i < cps.size(); ++i) {
            byteOffset[i] = off;
            off += unicode_cpt_to_utf8(cps[i]).size();
        }
        byteOffset[cps.size()] = off;

        std::vector<int> sub;
        size_t start = 0;
        while (start < cps.size()) {
            size_t end = maxPieceCps > 0 ? std::min(cps.size(), start + maxPieceCps) : cps.size();
            int matchedId = -1;
            while (end > start) {
                std::string substr = word.substr(byteOffset[start], byteOffset[end] - byteOffset[start]);
                std::string candidate = (start > 0) ? (continuingSubwordPrefix + substr) : substr;
                auto it = pieceToId.find(candidate);
                if (it != pieceToId.end()) {
                    matchedId = it->second;
                    break;
                }
                --end;
            }
            if (matchedId < 0) {
                out.push_back(unkId);
                return;
            }
            sub.push_back(matchedId);
            start = end;
        }
        out.insert(out.end(), sub.begin(), sub.end());
    }

    std::vector<int> encode(const std::string& word) const {
        std::vector<int> out;
        encodeWord(word, out);
        return out;
    }

    std::string decode(const std::vector<int>& tokens) const {
        std::string result;
        for (int id : tokens) {
            if (id < 0 || (size_t)id >= idToPiece.size())
                continue;
            const std::string& piece = idToPiece[id];
            // Empty prefix would make compare() match vacuously on every piece,
            // joining all pieces with no spaces.
            bool isCont = !continuingSubwordPrefix.empty() &&
                          piece.compare(0, continuingSubwordPrefix.size(), continuingSubwordPrefix) == 0;
            if (isCont) {
                result += piece.substr(continuingSubwordPrefix.size());
            } else {
                if (!result.empty())
                    result += ' ';
                result += piece;
            }
        }
        return result;
    }
};

CV__DNN_INLINE_NS_END
}}
#endif
