// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "granite_docling_engine.hpp"
#include "../local_vlm_model_base.hpp"
#include "../vlm_generation.hpp"
#include "../config_json.hpp"

#include <cstring>
#include <sstream>

namespace cv { namespace vlm {

using namespace cv::dnn;

namespace {

const String DEFAULT_PROMPT =
    "Convert this page to docling. Preserve OCR text, table structure, "
    "form fields, and layout/section structure.";

void resizeAA(const Mat& src, Mat& dst, Size size)
{
    bool shrinking = (int64_t)size.width * size.height < (int64_t)src.cols * src.rows;
    resize(src, dst, size, 0, 0, shrinking ? INTER_AREA : INTER_LANCZOS4);
}

Mat tileImage(const Mat& imageBgr, int longestEdge, int tileSize,
              const Vec3f& mean, const Vec3f& std_, int& rowsOut, int& colsOut)
{
    int h0 = imageBgr.rows, w0 = imageBgr.cols;
    int newW, newH;
    if (w0 >= h0)
    {
        newW = longestEdge;
        newH = std::max(1, (int)std::round((double)longestEdge * h0 / w0));
    }
    else
    {
        newH = longestEdge;
        newW = std::max(1, (int)std::round((double)longestEdge * w0 / h0));
    }

    Mat resized;
    resizeAA(imageBgr, resized, Size(newW, newH));

    int rows = (newH + tileSize - 1) / tileSize;
    int cols = (newW + tileSize - 1) / tileSize;
    rowsOut = rows;
    colsOut = cols;

    Mat grid;
    resizeAA(resized, grid, Size(cols * tileSize, rows * tileSize));

    int numTiles = rows * cols + 1;
    int sizes[] = {1, numTiles, 3, tileSize, tileSize};
    Mat pixelValues(5, sizes, CV_32F);

    auto normalizeTile = [&](const Mat& tileBgr, int tileIdx)
    {
        Mat tileRgb;
        cvtColor(tileBgr, tileRgb, COLOR_BGR2RGB);
        tileRgb.convertTo(tileRgb, CV_32F, 1.0 / 255.0);
        std::vector<Mat> channels(3);
        split(tileRgb, channels);
        for (int c = 0; c < 3; c++)
        {
            Mat dst(tileSize, tileSize, CV_32F, pixelValues.ptr<float>(0, tileIdx, c));
            channels[c].convertTo(dst, -1, 1.0 / std_[c], -mean[c] / std_[c]);
        }
    };

    int idx = 0;
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
        {
            Rect roi(c * tileSize, r * tileSize, tileSize, tileSize);
            normalizeTile(grid(roi), idx++);
        }
    Mat thumbnail;
    resizeAA(resized, thumbnail, Size(tileSize, tileSize));
    normalizeTile(thumbnail, idx);

    return pixelValues;
}

String buildPrompt(int rows, int cols, int imageSeqLen, const String& userText)
{
    std::ostringstream imagePart;
    for (int h = 0; h < rows; h++)
    {
        for (int w = 0; w < cols; w++)
        {
            imagePart << "<fake_token_around_image><row_" << (h + 1) << "_col_" << (w + 1) << ">";
            for (int i = 0; i < imageSeqLen; i++)
                imagePart << "<image>";
        }
        imagePart << "\n";
    }
    imagePart << "\n<fake_token_around_image><global-img>";
    for (int i = 0; i < imageSeqLen; i++)
        imagePart << "<image>";
    imagePart << "<fake_token_around_image>";

    std::ostringstream full;
    full << "<|start_of_role|>user<|end_of_role|>" << imagePart.str() << userText
         << "<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>";
    return full.str();
}

class GraniteDoclingModel CV_FINAL : public LocalVLMModelBase
{
public:
    GraniteDoclingModel(const String& model_dir, int engine, const String& device)
    {
        tokenizer_ = Tokenizer::loadVLM(model_dir + "/", "granite-docling");

        FileStorage config = openJsonConfigOrThrow(model_dir + "/config.json");
        FileStorage preprocessor = openJsonConfigOrThrow(model_dir + "/preprocessor_config.json");
        FileStorage processor = openJsonConfigOrThrow(model_dir + "/processor_config.json");

        imageTokenId_ = getIntWithTextConfigFallback(config, "image_token_id", 0);
        eosTokenId_ = getIntWithTextConfigFallback(config, "eos_token_id", 2);
        imageSeqLen_ = getInt(processor.root(), "image_seq_len", 64);
        longestEdge_ = getInt(preprocessor["size"], "longest_edge", 1536);
        maxTileEdge_ = getInt(preprocessor["max_image_size"], "longest_edge", 512);
        getVec3f(preprocessor.root(), "image_mean", mean_, Vec3f(0.5f, 0.5f, 0.5f));
        getVec3f(preprocessor.root(), "image_std", std_, Vec3f(0.5f, 0.5f, 0.5f));

        visionNet_ = readNetFromONNX(model_dir + "/onnx/vision_encoder.onnx", engine);
        embedNet_ = readNetFromONNX(model_dir + "/onnx/embed_tokens.onnx", engine);
        decoderNet_ = readNetFromONNX(model_dir + "/onnx/decoder_model_merged.onnx", engine);
        registerNets(visionNet_, embedNet_, decoderNet_);
        setPreferableDevice(device);
    }

    String infer(InputArray image, const String& prompt, int max_new_tokens) CV_OVERRIDE
    {
        Mat imageBgr = image.getMat();
        CV_CheckFalse(imageBgr.empty(), "vlm: input image is empty");
        String actualPrompt = prompt.empty() ? DEFAULT_PROMPT : prompt;

        int rows, cols;
        Mat pixelValues = tileImage(imageBgr, longestEdge_, maxTileEdge_, mean_, std_, rows, cols);
        String fullPrompt = buildPrompt(rows, cols, imageSeqLen_, actualPrompt);

        std::vector<int> tokens = tokenizer_.encode(fullPrompt);
        int promptLen = (int)tokens.size();
        std::vector<int64_t> inputIdsData(tokens.begin(), tokens.end());
        int idsShape[] = {1, promptLen};
        Mat inputIds(2, idsShape, CV_64S, inputIdsData.data());

        int maskShape[] = {1, pixelValues.size[1], pixelValues.size[3], pixelValues.size[4]};
        Mat pixelAttentionMask(4, maskShape, CV_Bool, Scalar(1));

        visionNet_.setInput(pixelValues, "pixel_values");
        visionNet_.setInput(pixelAttentionMask, "pixel_attention_mask");
        Mat imageFeatures = visionNet_.forward();

        embedNet_.setInput(inputIds, "input_ids");
        Mat inputsEmbeds = embedNet_.forward();

        int hiddenDim = inputsEmbeds.size[2];
        float* embedsData = inputsEmbeds.ptr<float>();
        const float* featData = imageFeatures.ptr<float>();
        int featIdx = 0;
        for (int i = 0; i < promptLen; i++)
            if (tokens[i] == imageTokenId_)
                memcpy(embedsData + (size_t)i * hiddenDim, featData + (size_t)(featIdx++) * hiddenDim,
                       hiddenDim * sizeof(float));

        std::vector<int> generated = generateWithKVCache(embedNet_, decoderNet_, inputsEmbeds,
                                                          promptLen, max_new_tokens, eosTokenId_);
        return tokenizer_.decode(generated);
    }

private:
    Net visionNet_, embedNet_, decoderNet_;
    Tokenizer tokenizer_;
    int imageTokenId_ = 0, eosTokenId_ = 2, imageSeqLen_ = 64;
    int longestEdge_ = 1536, maxTileEdge_ = 512;
    Vec3f mean_ = Vec3f(0.5f, 0.5f, 0.5f), std_ = Vec3f(0.5f, 0.5f, 0.5f);
};

} // namespace

Ptr<VLMModel> createGraniteDoclingModel(const String& model_dir, int engine, const String& device)
{
    return makePtr<GraniteDoclingModel>(model_dir, engine, device);
}

}} // namespace cv::vlm
