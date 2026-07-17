// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/*
 * This is a sample program to run Granite-Docling-258M vision-language inference in OpenCV
 * using ONNX models and OpenCV's new DNN engine (ENGINE_NEW). Given a page image and a
 * text prompt, it generates a "doctag" style text response describing the page (OCR text,
 * tables, form fields, and layout/section structure).
 *
 * The model is split into three ONNX files:
 *     - Vision encoder : image tiles -> image features
 *     - Embed tokens    : prompt token ids -> text embeddings
 *     - Decoder         : [image features | text embeddings] -> logits (with KV-cache)
 *
 * Model: https://huggingface.co/ibm-granite/granite-docling-258M
 * ONNX:  https://huggingface.co/onnx-community/granite-docling-258M-ONNX
 *
 * Run the sample:
 * 1. Download the plain (non-quantized) fp32 ONNX export into <model_dir>, keeping the
 *    upstream layout (config.json, preprocessor_config.json, processor_config.json,
 *    tokenizer.json at the root; onnx/vision_encoder.onnx, onnx/embed_tokens.onnx,
 *    onnx/decoder_model_merged.onnx under onnx/).
 *
 * 2. Run:
 *
 *      ./granite_docling_inference --model_dir=<model_dir> \
 *                                  --input=<path-to-page-image>
 *
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

static const string DEFAULT_PROMPT =
    "Convert this page to docling. Preserve OCR text, table structure, "
    "form fields, and layout/section structure.";

static void resizeAA(const Mat& src, Mat& dst, Size size)
{
    bool shrinking = (int64_t)size.width * size.height < (int64_t)src.cols * src.rows;
    resize(src, dst, size, 0, 0, shrinking ? INTER_AREA : INTER_LANCZOS4);
}

static Mat tileImage(const Mat& imageBgr, int longestEdge, int tileSize,
                     const Vec3f& mean, const Vec3f& std_, int& rowsOut, int& colsOut)
{
    int h0 = imageBgr.rows, w0 = imageBgr.cols;
    int newW, newH;
    if (w0 >= h0)
    {
        newW = longestEdge;
        newH = max(1, (int)round((double)longestEdge * h0 / w0));
    }
    else
    {
        newH = longestEdge;
        newW = max(1, (int)round((double)longestEdge * w0 / h0));
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
        vector<Mat> channels(3);
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

static string buildPrompt(int rows, int cols, int imageSeqLen, const string& userText)
{

    ostringstream imagePart;
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

    ostringstream full;
    full << "<|start_of_role|>user<|end_of_role|>" << imagePart.str() << userText
         << "<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>";
    return full.str();
}

static int argmaxLastToken(const Mat& logits)
{
    int seqLen = logits.size[1];
    int vocabSize = logits.size[2];
    const float* row = logits.ptr<float>(0, seqLen - 1);
    return (int)(max_element(row, row + vocabSize) - row);
}

static vector<int> graniteDoclingInference(Net& visionNet, Net& embedNet, Net& decoderNet,
                                           const Mat& pixelValues, const string& prompt,
                                           int maxNewTokens, Tokenizer& tokenizer,
                                           int imageTokenId, int eosTokenId)
{
    cout << "Inferencing Granite-Docling-258M model..." << endl;

    vector<int> tokens = tokenizer.encode(prompt);
    int promptLen = (int)tokens.size();
    vector<int64_t> inputIdsData(tokens.begin(), tokens.end());
    int idsShape[] = {1, promptLen};
    Mat inputIds(2, idsShape, CV_64S, inputIdsData.data());

    int maskShape[] = {1, pixelValues.size[1], pixelValues.size[3], pixelValues.size[4]};
    Mat pixelAttentionMask(4, maskShape, CV_Bool, Scalar(1));

    visionNet.setInput(pixelValues, "pixel_values");
    visionNet.setInput(pixelAttentionMask, "pixel_attention_mask");
    Mat imageFeatures = visionNet.forward();

    embedNet.setInput(inputIds, "input_ids");
    Mat inputsEmbeds = embedNet.forward();

    int hiddenDim = inputsEmbeds.size[2];
    float* embedsData = inputsEmbeds.ptr<float>();
    const float* featData = imageFeatures.ptr<float>();
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
        "{ model_dir      |      | Path to the local onnx-community/granite-docling-258M-ONNX export "
        "(config.json, preprocessor_config.json, processor_config.json, "
        "onnx/vision_encoder.onnx, onnx/embed_tokens.onnx, onnx/decoder_model_merged.onnx) }"
        "{ input i        |      | Path to the input page image }"
        "{ prompt         |      | Task prompt (defaults to the built-in docling conversion prompt) }"
        "{ max_new_tokens | 512  | Maximum number of new tokens to generate }"
        "{ seed           | 0    | Random seed }";

    CommandLineParser parser(argc, argv, keys);
    parser.about("Use this sample to run Granite-Docling-258M vision-language inference in OpenCV");
    if (parser.has("help") || !parser.has("model_dir") || !parser.has("input"))
    {
        parser.printMessage();
        return 0;
    }

    string modelDir = parser.get<String>("model_dir");
    string inputPath = parser.get<String>("input");
    string prompt = parser.get<String>("prompt");
    if (prompt.empty())
        prompt = DEFAULT_PROMPT;
    int maxNewTokens = parser.get<int>("max_new_tokens");
    setRNGSeed(parser.get<int>("seed"));

    cout << "Preparing Granite-Docling-258M model..." << endl;
    Tokenizer tokenizer = Tokenizer::loadVLM(modelDir + "/", "granite-docling");

    FileStorage config(modelDir + "/config.json", FileStorage::READ | FileStorage::FORMAT_JSON);
    if (!config.isOpened())
    {
        cerr << "Could not open " << modelDir + "/config.json" << endl;
        return 1;
    }
    FileStorage preprocessor(modelDir + "/preprocessor_config.json", FileStorage::READ | FileStorage::FORMAT_JSON);
    if (!preprocessor.isOpened())
    {
        cerr << "Could not open " << modelDir + "/preprocessor_config.json" << endl;
        return 1;
    }
    FileStorage processor(modelDir + "/processor_config.json", FileStorage::READ | FileStorage::FORMAT_JSON);
    if (!processor.isOpened())
    {
        cerr << "Could not open " << modelDir + "/processor_config.json" << endl;
        return 1;
    }

    int imageTokenId = (int)config["image_token_id"];
    FileNode textConfig = config["text_config"];
    int eosTokenId = (!textConfig.empty() && !textConfig["eos_token_id"].empty())
                        ? (int)textConfig["eos_token_id"]
                        : (int)config["eos_token_id"];
    int imageSeqLen = (int)processor["image_seq_len"];
    int longestEdge = (int)preprocessor["size"]["longest_edge"];
    int maxTileEdge = (int)preprocessor["max_image_size"]["longest_edge"];
    vector<float> meanVec, stdVec;
    preprocessor["image_mean"] >> meanVec;
    preprocessor["image_std"] >> stdVec;
    Vec3f mean(meanVec[0], meanVec[1], meanVec[2]);
    Vec3f stdDev(stdVec[0], stdVec[1], stdVec[2]);

    Net visionNet  = readNetFromONNX(modelDir + "/onnx/vision_encoder.onnx", ENGINE_NEW);
    Net embedNet   = readNetFromONNX(modelDir + "/onnx/embed_tokens.onnx", ENGINE_NEW);
    Net decoderNet = readNetFromONNX(modelDir + "/onnx/decoder_model_merged.onnx", ENGINE_NEW);

    Mat image = imread(inputPath);
    if (image.empty())
    {
        cerr << "Could not read image: " << inputPath << endl;
        return 1;
    }

    int rows, cols;
    Mat pixelValues = tileImage(image, longestEdge, maxTileEdge, mean, stdDev, rows, cols);
    string fullPrompt = buildPrompt(rows, cols, imageSeqLen, prompt);
    cout << "Prompt:\n" << prompt << endl;

    vector<int> generated = graniteDoclingInference(visionNet, embedNet, decoderNet, pixelValues, fullPrompt,
                                                     maxNewTokens, tokenizer, imageTokenId, eosTokenId);
    string response = tokenizer.decode(generated);
    cout << "Response:\n" << response << endl;
    return 0;
}
