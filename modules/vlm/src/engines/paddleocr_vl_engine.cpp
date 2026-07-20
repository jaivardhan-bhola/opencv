// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "paddleocr_vl_engine.hpp"
#include "../vlm_model_base.hpp"
#include "../vlm_generation.hpp"
#include "../config_json.hpp"

#include <cmath>
#include <cstring>
#include <sstream>

namespace cv { namespace vlm {

using namespace cv::dnn;

namespace {

const String DEFAULT_PROMPT = "OCR";

void smartResize(int height, int width, int factor, int minPixels, int maxPixels,
                  int& outHeight, int& outWidth)
{
    if (height < factor)
    {
        width = (int)std::round((double)(width * factor) / height);
        height = factor;
    }
    if (width < factor)
    {
        height = (int)std::round((double)(height * factor) / width);
        width = factor;
    }
    CV_CheckLE((double)std::max(height, width) / std::min(height, width), 200.0,
               "vlm: absolute aspect ratio is too large");

    int hBar = (int)std::round((double)height / factor) * factor;
    int wBar = (int)std::round((double)width / factor) * factor;
    if ((int64_t)hBar * wBar > maxPixels)
    {
        double beta = std::sqrt((double)(height * width) / maxPixels);
        hBar = (int)(std::floor(height / beta / factor)) * factor;
        wBar = (int)(std::floor(width / beta / factor)) * factor;
    }
    else if ((int64_t)hBar * wBar < minPixels)
    {
        double beta = std::sqrt((double)minPixels / (height * width));
        hBar = (int)(std::ceil(height * beta / factor)) * factor;
        wBar = (int)(std::ceil(width * beta / factor)) * factor;
    }
    outHeight = hBar;
    outWidth = wBar;
}

Mat preprocessImage(const Mat& imageBgr, int patchSize, int mergeSize, int minPixels, int maxPixels,
                    const Vec3f& mean, const Vec3f& std_, float rescaleFactor, int& gridH, int& gridW)
{
    int factor = patchSize * mergeSize;
    int resizedHeight, resizedWidth;
    smartResize(imageBgr.rows, imageBgr.cols, factor, minPixels, maxPixels, resizedHeight, resizedWidth);

    Mat resized;
    resize(imageBgr, resized, Size(resizedWidth, resizedHeight), 0, 0, INTER_CUBIC);
    Mat rgb;
    cvtColor(resized, rgb, COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, rescaleFactor);

    std::vector<Mat> channels(3);
    split(rgb, channels);
    for (int c = 0; c < 3; c++)
        channels[c].convertTo(channels[c], -1, 1.0 / std_[c], -mean[c] / std_[c]);

    gridH = resizedHeight / patchSize;
    gridW = resizedWidth / patchSize;
    int numPatches = gridH * gridW;

    int sizes[] = {1, numPatches, 3, patchSize, patchSize};
    Mat pixelValues(5, sizes, CV_32F);

    int idx = 0;
    for (int h = 0; h < gridH; h++)
        for (int w = 0; w < gridW; w++)
        {
            Rect roi(w * patchSize, h * patchSize, patchSize, patchSize);
            for (int c = 0; c < 3; c++)
            {
                Mat dst(patchSize, patchSize, CV_32F, pixelValues.ptr<float>(0, idx, c));
                channels[c](roi).copyTo(dst);
            }
            idx++;
        }
    return pixelValues;
}

String buildPrompt(const String& prompt, int imageTokenRepeats)
{
    std::ostringstream oss;
    oss << "<|begin_of_sentence|>User: <|IMAGE_START|>";
    for (int i = 0; i < imageTokenRepeats; i++)
        oss << "<|IMAGE_PLACEHOLDER|>";
    oss << "<|IMAGE_END|>" << prompt << "\nAssistant:\n";
    return oss.str();
}

class PaddleOCRVLModel CV_FINAL : public VLMModelBase
{
public:
    PaddleOCRVLModel(const String& model_dir, int engine, const String& device)
    {
        tokenizer_ = Tokenizer::loadVLM(model_dir + "/", "paddleocr-vl");

        FileStorage config = openJsonConfigOrThrow(model_dir + "/config.json");
        FileStorage processorFs = openJsonConfigOrThrow(model_dir + "/processor_config.json");
        FileNode preprocessor = processorFs["image_processor"];

        imageTokenId_ = getIntWithTextConfigFallback(config, "image_token_id", 0);
        eosTokenId_ = getIntWithTextConfigFallback(config, "eos_token_id", 2);

        patchSize_ = getInt(preprocessor, "patch_size", 14);
        mergeSize_ = getInt(preprocessor, "merge_size", 2);
        minPixels_ = getInt(preprocessor, "min_pixels", 28 * 28 * 130);
        maxPixels_ = getInt(preprocessor, "max_pixels", 28 * 28 * 1280);
        rescaleFactor_ = getFloat(preprocessor, "rescale_factor", 1.0f / 255.0f);
        getVec3f(preprocessor, "image_mean", mean_, Vec3f(0.5f, 0.5f, 0.5f));
        getVec3f(preprocessor, "image_std", std_, Vec3f(0.5f, 0.5f, 0.5f));

        visionNet_ = readNetFromONNX(model_dir + "/onnx/vision_encoder.onnx", engine);
        embedNet_ = readNetFromONNX(model_dir + "/onnx/embedding.onnx", engine);
        decoderNet_ = readNetFromONNX(model_dir + "/onnx/decoder.onnx", engine);
        registerNets(visionNet_, embedNet_, decoderNet_);
        setPreferableDevice(device);
    }

    String infer(InputArray image, const String& prompt, int max_new_tokens) CV_OVERRIDE
    {
        Mat imageBgr = image.getMat();
        CV_CheckFalse(imageBgr.empty(), "vlm: input image is empty");
        String actualPrompt = prompt.empty() ? DEFAULT_PROMPT : prompt;

        int gridH, gridW;
        Mat pixelValues = preprocessImage(imageBgr, patchSize_, mergeSize_, minPixels_, maxPixels_,
                                          mean_, std_, rescaleFactor_, gridH, gridW);
        int gridShape[] = {1, 3};
        Mat imageGridThw(2, gridShape, CV_64S);
        imageGridThw.at<int64_t>(0, 0) = 1;
        imageGridThw.at<int64_t>(0, 1) = gridH;
        imageGridThw.at<int64_t>(0, 2) = gridW;

        int imageTokenRepeats = (int)((1LL * gridH * gridW) / mergeSize_ / mergeSize_);
        std::vector<int> tokens = tokenizer_.encode(buildPrompt(actualPrompt, imageTokenRepeats));
        int promptLen = (int)tokens.size();
        std::vector<int64_t> inputIdsData(tokens.begin(), tokens.end());
        int idsShape[] = {1, promptLen};
        Mat inputIds(2, idsShape, CV_64S, inputIdsData.data());

        visionNet_.setInput(pixelValues, "pixel_values");
        visionNet_.setInput(imageGridThw, "image_grid_thw");
        Mat imageEmbeds = visionNet_.forward();

        embedNet_.setInput(inputIds, "input_ids");
        Mat inputsEmbeds = embedNet_.forward();

        int hiddenDim = inputsEmbeds.size[2];
        float* embedsData = inputsEmbeds.ptr<float>();
        const float* featData = imageEmbeds.ptr<float>();
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
    int imageTokenId_ = 0, eosTokenId_ = 2;
    int patchSize_ = 14, mergeSize_ = 2, minPixels_ = 28 * 28 * 130, maxPixels_ = 28 * 28 * 1280;
    float rescaleFactor_ = 1.0f / 255.0f;
    Vec3f mean_ = Vec3f(0.5f, 0.5f, 0.5f), std_ = Vec3f(0.5f, 0.5f, 0.5f);
};

} // namespace

Ptr<VLMModel> createPaddleOCRVLModel(const String& model_dir, int engine, const String& device)
{
    return makePtr<PaddleOCRVLModel>(model_dir, engine, device);
}

}} // namespace cv::vlm
