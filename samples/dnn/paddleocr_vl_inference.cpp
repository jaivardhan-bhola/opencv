/*
 * This is a sample program to run PaddleOCR-VL-1.5 vision-language inference in OpenCV
 * using ONNX models and OpenCV's new DNN engine (ENGINE_NEW). Given an image and a text
 * prompt, it generates a text response (e.g. recognized document text).
 *
 * The model is split into three ONNX files:
 *     - Vision encoder : image patches -> image embeddings
 *     - Embedding      : prompt token ids -> text embeddings
 *     - Decoder        : [image embeddings | text embeddings] -> logits (with KV-cache)
 *
 * Model: https://huggingface.co/PaddlePaddle/PaddleOCR-VL
 * ONNX:  https://huggingface.co/onnx-community/PaddleOCR-VL-1.5-ONNX
 *
 * Run the sample:
 * 1. Download the plain (non-quantized) fp32 ONNX export into <model_dir>, keeping the
 *    upstream layout (config.json, tokenizer.json, processor_config.json at the root;
 *    onnx/vision_encoder.onnx, onnx/embedding.onnx, onnx/decoder.onnx under onnx/).
 * 2. Run:
 *
 *      ./paddleocr_vl_inference --model_dir=<model_dir> \
 *                               --input=<path-to-image>
 */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace cv::dnn;
using namespace std;

static int configValue(const FileStorage& config, const string& name, int fallback)
{
    FileNode node = config[name];
    if (!node.empty())
        return (int)node;
    FileNode textConfig = config["text_config"];
    if (!textConfig.empty() && !textConfig[name].empty())
        return (int)textConfig[name];
    return fallback;
}

static void smartResize(int height, int width, int factor, int minPixels, int maxPixels,
                        int& outHeight, int& outWidth)
{
    if (height < factor)
    {
        width = (int)round((double)(width * factor) / height);
        height = factor;
    }
    if (width < factor)
    {
        height = (int)round((double)(height * factor) / width);
        width = factor;
    }
    CV_CheckLE((double)max(height, width) / min(height, width), 200.0, "absolute aspect ratio is too large");

    int hBar = (int)round((double)height / factor) * factor;
    int wBar = (int)round((double)width / factor) * factor;
    if ((int64_t)hBar * wBar > maxPixels)
    {
        double beta = sqrt((double)(height * width) / maxPixels);
        hBar = (int)(floor(height / beta / factor)) * factor;
        wBar = (int)(floor(width / beta / factor)) * factor;
    }
    else if ((int64_t)hBar * wBar < minPixels)
    {
        double beta = sqrt((double)minPixels / (height * width));
        hBar = (int)(ceil(height * beta / factor)) * factor;
        wBar = (int)(ceil(width * beta / factor)) * factor;
    }
    outHeight = hBar;
    outWidth = wBar;
}

static Mat preprocessImage(const Mat& imageBgr, int patchSize, int mergeSize, int minPixels, int maxPixels,
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

    vector<Mat> channels(3);
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

static string buildPrompt(const string& prompt, int imageTokenRepeats)
{
    ostringstream oss;
    oss << "<|begin_of_sentence|>User: <|IMAGE_START|>";
    for (int i = 0; i < imageTokenRepeats; i++)
        oss << "<|IMAGE_PLACEHOLDER|>";
    oss << "<|IMAGE_END|>" << prompt << "\nAssistant:\n";
    return oss.str();
}

static int argmaxLastToken(const Mat& logits)
{
    int seqLen = logits.size[1];
    int vocabSize = logits.size[2];
    const float* row = logits.ptr<float>(0, seqLen - 1);
    return (int)(max_element(row, row + vocabSize) - row);
}

static vector<int> paddleocrVlInference(Net& visionNet, Net& embedNet, Net& decoderNet,
                                        const Mat& pixelValues, const Mat& imageGridThw,
                                        const string& prompt, int maxNewTokens, Tokenizer& tokenizer,
                                        int imageTokenId, int eosTokenId, int mergeSize)
{
    cout << "Inferencing PaddleOCR-VL-1.5 model..." << endl;

    int64_t gridT = imageGridThw.at<int64_t>(0, 0);
    int64_t gridH = imageGridThw.at<int64_t>(0, 1);
    int64_t gridW = imageGridThw.at<int64_t>(0, 2);
    int imageTokenRepeats = (int)((gridT * gridH * gridW) / mergeSize / mergeSize);

    vector<int> tokens = tokenizer.encode(buildPrompt(prompt, imageTokenRepeats));
    int promptLen = (int)tokens.size();
    vector<int64_t> inputIdsData(tokens.begin(), tokens.end());
    int idsShape[] = {1, promptLen};
    Mat inputIds(2, idsShape, CV_64S, inputIdsData.data());

    visionNet.setInput(pixelValues, "pixel_values");
    visionNet.setInput(imageGridThw, "image_grid_thw");
    Mat imageEmbeds = visionNet.forward();

    embedNet.setInput(inputIds, "input_ids");
    Mat inputsEmbeds = embedNet.forward();

    int hiddenDim = inputsEmbeds.size[2];
    float* embedsData = inputsEmbeds.ptr<float>();
    const float* featData = imageEmbeds.ptr<float>();
    int featIdx = 0;
    for (int i = 0; i < promptLen; i++)
        if (tokens[i] == imageTokenId)
            memcpy(embedsData + (size_t)i * hiddenDim, featData + (size_t)(featIdx++) * hiddenDim, hiddenDim * sizeof(float));

    decoderNet.enableKVCache();
    vector<int64_t> maskData(promptLen, 1);
    int attnShape[] = {1, promptLen};
    Mat attentionMask(2, attnShape, CV_64S, maskData.data());

    decoderNet.setInput(inputsEmbeds, "inputs_embeds");
    decoderNet.setInput(attentionMask, "attention_mask");
    Mat logits = decoderNet.forward();
    int newId = argmaxLastToken(logits);
    vector<int> generated = {newId};

    for (int step = 0; step < maxNewTokens - 1; step++)
    {
        if (newId == eosTokenId)
            break;

        int64_t idData[1] = {newId};
        int idShape[] = {1, 1};
        Mat newIdMat(2, idShape, CV_64S, idData);
        embedNet.setInput(newIdMat, "input_ids");
        Mat newEmbed = embedNet.forward();

        maskData.push_back(1);
        int newAttnShape[] = {1, (int)maskData.size()};
        Mat newAttentionMask(2, newAttnShape, CV_64S, maskData.data());

        decoderNet.setInput(newEmbed, "inputs_embeds");
        decoderNet.setInput(newAttentionMask, "attention_mask");
        logits = decoderNet.forward();
        newId = argmaxLastToken(logits);
        generated.push_back(newId);
    }

    if (!generated.empty() && generated.back() == eosTokenId)
        generated.pop_back();

    return generated;
}

int main(int argc, char** argv)
{
    const string keys =
        "{ help h         |      | Print help message }"
        "{ model_dir      |      | Path to the local onnx-community/PaddleOCR-VL-1.5-ONNX export "
        "(config.json, processor_config.json, "
        "onnx/vision_encoder.onnx, onnx/embedding.onnx, onnx/decoder.onnx) }"
        "{ input i        |      | Path to the input image }"
        "{ prompt         | OCR | Task prompt }"
        "{ max_new_tokens | 512  | Maximum number of new tokens to generate }"
        "{ seed           | 0    | Random seed }";

    CommandLineParser parser(argc, argv, keys);
    parser.about("Use this sample to run PaddleOCR-VL-1.5 vision-language inference in OpenCV");
    if (parser.has("help") || !parser.has("model_dir") || !parser.has("input"))
    {
        parser.printMessage();
        return 0;
    }

    string modelDir = parser.get<String>("model_dir");
    string inputPath = parser.get<String>("input");
    string prompt = parser.get<String>("prompt");
    int maxNewTokens = parser.get<int>("max_new_tokens");
    setRNGSeed(parser.get<int>("seed"));

    cout << "Preparing PaddleOCR-VL-1.5 model..." << endl;
    Tokenizer tokenizer = Tokenizer::loadVLM(modelDir + "/", "paddleocr-vl");

    FileStorage config(modelDir + "/config.json", FileStorage::READ | FileStorage::FORMAT_JSON);
    if (!config.isOpened())
    {
        cerr << "Could not open " << modelDir + "/config.json" << endl;
        return 1;
    }

    FileStorage processorFs(modelDir + "/processor_config.json", FileStorage::READ | FileStorage::FORMAT_JSON);
    if (!processorFs.isOpened())
    {
        cerr << "Could not open " << modelDir + "/processor_config.json" << endl;
        return 1;
    }
    FileNode preprocessor = processorFs["image_processor"];

    int imageTokenId = configValue(config, "image_token_id", 0);
    int eosTokenId = configValue(config, "eos_token_id", 2);

    int patchSize = preprocessor["patch_size"].empty() ? 14 : (int)preprocessor["patch_size"];
    int mergeSize = preprocessor["merge_size"].empty() ? 2 : (int)preprocessor["merge_size"];
    int minPixels = preprocessor["min_pixels"].empty() ? 28 * 28 * 130 : (int)preprocessor["min_pixels"];
    int maxPixels = preprocessor["max_pixels"].empty() ? 28 * 28 * 1280 : (int)preprocessor["max_pixels"];
    float rescaleFactor = preprocessor["rescale_factor"].empty() ? 1.0f / 255.0f : (float)preprocessor["rescale_factor"];

    Vec3f mean(0.5f, 0.5f, 0.5f), stdDev(0.5f, 0.5f, 0.5f);
    if (!preprocessor["image_mean"].empty())
    {
        vector<float> meanVec, stdVec;
        preprocessor["image_mean"] >> meanVec;
        preprocessor["image_std"] >> stdVec;
        mean = Vec3f(meanVec[0], meanVec[1], meanVec[2]);
        stdDev = Vec3f(stdVec[0], stdVec[1], stdVec[2]);
    }

    Net visionNet  = readNetFromONNX(modelDir + "/onnx/vision_encoder.onnx", ENGINE_NEW);
    Net embedNet   = readNetFromONNX(modelDir + "/onnx/embedding.onnx", ENGINE_NEW);
    Net decoderNet = readNetFromONNX(modelDir + "/onnx/decoder.onnx", ENGINE_NEW);

    Mat image = imread(inputPath);
    if (image.empty())
    {
        cerr << "Could not read image: " << inputPath << endl;
        return 1;
    }

    int gridH, gridW;
    Mat pixelValues = preprocessImage(image, patchSize, mergeSize, minPixels, maxPixels, mean, stdDev, rescaleFactor, gridH, gridW);
    int gridShape[] = {1, 3};
    Mat imageGridThw(2, gridShape, CV_64S);
    imageGridThw.at<int64_t>(0, 0) = 1;
    imageGridThw.at<int64_t>(0, 1) = gridH;
    imageGridThw.at<int64_t>(0, 2) = gridW;

    cout << "Prompt:\n" << prompt << endl;
    vector<int> generated = paddleocrVlInference(visionNet, embedNet, decoderNet, pixelValues, imageGridThw,
                                                 prompt, maxNewTokens, tokenizer, imageTokenId, eosTokenId, mergeSize);
    string response = tokenizer.decode(generated);
    cout << "Response:\n" << response << endl;
    return 0;
}
