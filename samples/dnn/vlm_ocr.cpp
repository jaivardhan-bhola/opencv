// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Third party copyrights are property of their respective owners.

/*
 * This is a sample program demonstrating cv::vlm::VLMModel: a single API for running
 * vision-language OCR / document-understanding inference with either PaddleOCR-VL-1.5
 * or Granite-Docling-258M, given a model type, a local ONNX export directory, and an
 * input image.
 *
 * Run the sample:
 *
 *      ./vlm_ocr --model_type=paddleocr-vl --model_dir=<dir> --input=<path-to-image>
 */

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/vlm.hpp>

using namespace cv;
using namespace cv::vlm;
using namespace std;

int main(int argc, char** argv)
{
    const string keys =
        "{ help h         |      | Print help message }"
        "{ model_type     |      | Which VLM to run: paddleocr-vl or granite-docling }"
        "{ model_dir      |      | Path to the local ONNX export directory for the chosen model_type }"
        "{ input i        |      | Path to the input image (.png/.jpg/.jpeg) }"
        "{ prompt         |      | Task prompt (defaults to the model's built-in prompt) }"
        "{ max_new_tokens | 512  | Maximum number of new tokens to generate }"
        "{ engine         | new  | dnn engine used to load each ONNX sub-model: new or ort }"
        "{ device         | cpu  | Compute device: cpu or cuda }";

    CommandLineParser parser(argc, argv, keys);
    parser.about("Use this sample to run vision-language OCR / document-understanding inference in OpenCV");
    if (parser.has("help") || !parser.has("model_type") || !parser.has("model_dir") || !parser.has("input"))
    {
        parser.printMessage();
        return 0;
    }

    string modelTypeArg = parser.get<String>("model_type");
    VLMModelType modelType;
    if (modelTypeArg == "paddleocr-vl")
        modelType = VLM_MODEL_PADDLEOCR_VL;
    else if (modelTypeArg == "granite-docling")
        modelType = VLM_MODEL_GRANITE_DOCLING;
    else
    {
        cerr << "Unknown model_type: " << modelTypeArg << " (expected paddleocr-vl or granite-docling)" << endl;
        return 1;
    }

    string engine = parser.get<String>("engine");
    if (engine != "new" && engine != "ort")
    {
        cerr << "Unknown engine: " << engine << " (expected new or ort)" << endl;
        return 1;
    }

    string device = parser.get<String>("device");
    if (device != "cpu" && device != "cuda")
    {
        cerr << "Unknown device: " << device << " (expected cpu or cuda)" << endl;
        return 1;
    }

    string modelDir = parser.get<String>("model_dir");
    string inputPath = parser.get<String>("input");
    string prompt = parser.get<String>("prompt");
    int maxNewTokens = parser.get<int>("max_new_tokens");

    cout << "Preparing " << modelTypeArg << " model..." << endl;
    Ptr<VLMModel> model = create(modelType, modelDir, engine, device);

    cout << "Running inference on " << inputPath << "..." << endl;
    vector<String> results = model->inferDocument(inputPath, prompt, maxNewTokens);
    for (size_t i = 0; i < results.size(); i++)
        cout << "Page " << (i + 1) << ":\n" << results[i] << endl;

    return 0;
}
