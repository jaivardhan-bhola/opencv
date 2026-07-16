// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/dnn/all_layers.hpp>

#include <cmath>

namespace opencv_test { namespace {

static Mat makeMat(std::initializer_list<int> shape, std::initializer_list<float> data)
{
    std::vector<int> shp(shape);
    std::vector<float> buf(data);
    size_t total = 1;
    for (int s : shp) total *= (size_t)s;
    CV_Assert(total == buf.size());
    return Mat((int)shp.size(), shp.data(), CV_32F, buf.data()).clone();
}

static Ptr<Layer> createSkipNorm(float epsilon = 1e-5f)
{
    LayerParams lp;
    lp.type = "SkipSimplifiedLayerNormalization";
    lp.name = "test_skip_norm";
    lp.set("epsilon", epsilon);
    Ptr<Layer> layer = LayerFactory::createLayerInstance("SkipSimplifiedLayerNormalization", lp);
    CV_Assert(layer);
    return layer;
}

// Unlike the generic runLayer() helper (which always requests 0 outputs), this
// op's getMemoryShapes()/getTypes() size their output list directly from
// requiredOutputs, so the test controls how many of the (up to 4) declared
// outputs -- output, mean, inv_std_var, input_skip_bias_sum -- get materialized.
static void runSkipNorm(Ptr<Layer>& layer, std::vector<Mat>& inputs,
                         std::vector<Mat>& outputs, int requiredOutputs)
{
    std::vector<MatShape> inShapes, outShapes, intShapes;
    std::vector<cv::dnn::MatType> inTypes, outTypes, intTypes;
    for (auto& m : inputs)
    {
        inShapes.push_back(shape(m));
        inTypes.push_back(cv::dnn::MatType(m.type()));
    }

    layer->getMemoryShapes(inShapes, requiredOutputs, outShapes, intShapes);
    layer->getTypes(inTypes, (int)outShapes.size(), (int)intShapes.size(), outTypes, intTypes);

    outputs.clear();
    for (size_t i = 0; i < outShapes.size(); ++i)
        outputs.push_back(Mat(outShapes[i], outTypes[i]));
    std::vector<Mat> internals;
    for (size_t i = 0; i < intShapes.size(); ++i)
        internals.push_back(Mat(intShapes[i], intTypes[i]));

    layer->finalize(inputs, outputs);
    layer->forward(inputs, outputs, internals);
}

// output = (input + skip) / sqrt(mean((input+skip)^2) + eps) * gamma
TEST(SkipSimplifiedLayerNormalizationLayer, BasicNoBias)
{
    const float epsilon = 1e-5f;
    Ptr<Layer> layer = createSkipNorm(epsilon);

    Mat input = makeMat({1, 1, 4}, {1.f, 2.f, 3.f, 4.f});
    Mat skip  = makeMat({1, 1, 4}, {0.1f, 0.2f, 0.3f, 0.4f});
    Mat gamma = makeMat({4}, {1.f, 1.f, 1.f, 1.f});

    std::vector<Mat> inputs = {input, skip, gamma};
    std::vector<Mat> outputs;
    runSkipNorm(layer, inputs, outputs, 2);
    ASSERT_EQ(outputs.size(), (size_t)2);

    float sum[4], meanSq = 0.f;
    for (int i = 0; i < 4; ++i)
    {
        sum[i] = input.ptr<float>()[i] + skip.ptr<float>()[i];
        meanSq += sum[i] * sum[i];
    }
    meanSq /= 4.f;
    const float rms = std::sqrt(meanSq + epsilon);
    float expectedOut[4];
    for (int i = 0; i < 4; ++i)
        expectedOut[i] = sum[i] / rms;

    Mat expectedOutMat = makeMat({1, 1, 4}, {expectedOut[0], expectedOut[1], expectedOut[2], expectedOut[3]});
    Mat expectedSumMat  = makeMat({1, 1, 4}, {sum[0], sum[1], sum[2], sum[3]});

    normAssert(expectedOutMat, outputs[0], "output", 1e-4, 1e-3);
    normAssert(expectedSumMat, outputs[1], "input_skip_bias_sum");
}

TEST(SkipSimplifiedLayerNormalizationLayer, WithBiasAndNonUniformGamma)
{
    const float epsilon = 1e-5f;
    Ptr<Layer> layer = createSkipNorm(epsilon);

    Mat input = makeMat({1, 1, 4}, {1.f, 2.f, 3.f, 4.f});
    Mat skip  = makeMat({1, 1, 4}, {0.1f, 0.2f, 0.3f, 0.4f});
    Mat gamma = makeMat({4}, {2.f, 0.5f, 1.f, 3.f});
    Mat bias  = makeMat({4}, {0.01f, 0.02f, 0.03f, 0.04f});

    std::vector<Mat> inputs = {input, skip, gamma, bias};
    std::vector<Mat> outputs;
    runSkipNorm(layer, inputs, outputs, 2);
    ASSERT_EQ(outputs.size(), (size_t)2);

    const float gammaData[4] = {2.f, 0.5f, 1.f, 3.f};
    float sum[4], meanSq = 0.f;
    for (int i = 0; i < 4; ++i)
    {
        sum[i] = input.ptr<float>()[i] + skip.ptr<float>()[i] + bias.ptr<float>()[i];
        meanSq += sum[i] * sum[i];
    }
    meanSq /= 4.f;
    const float rms = std::sqrt(meanSq + epsilon);
    float expectedOut[4];
    for (int i = 0; i < 4; ++i)
        expectedOut[i] = sum[i] / rms * gammaData[i];

    Mat expectedOutMat = makeMat({1, 1, 4}, {expectedOut[0], expectedOut[1], expectedOut[2], expectedOut[3]});
    Mat expectedSumMat  = makeMat({1, 1, 4}, {sum[0], sum[1], sum[2], sum[3]});

    normAssert(expectedOutMat, outputs[0], "output", 1e-4, 1e-3);
    normAssert(expectedSumMat, outputs[1], "input_skip_bias_sum");
}

// Each row along the leading (non-normalized) axis must be normalized independently.
TEST(SkipSimplifiedLayerNormalizationLayer, RowsNormalizedIndependently)
{
    const float epsilon = 1e-5f;
    Ptr<Layer> layer = createSkipNorm(epsilon);

    Mat input = makeMat({1, 2, 3}, {1.f, 1.f, 1.f,   2.f, 0.f, 0.f});
    Mat skip  = makeMat({1, 2, 3}, {0.f, 0.f, 0.f,   0.f, 1.f, -1.f});
    Mat gamma = makeMat({3}, {1.f, 2.f, 3.f});

    std::vector<Mat> inputs = {input, skip, gamma};
    std::vector<Mat> outputs;
    runSkipNorm(layer, inputs, outputs, 2);

    const float gammaData[3] = {1.f, 2.f, 3.f};
    float expectedOut[6], expectedSum[6];
    for (int row = 0; row < 2; ++row)
    {
        float sum[3], meanSq = 0.f;
        for (int i = 0; i < 3; ++i)
        {
            sum[i] = input.ptr<float>()[row * 3 + i] + skip.ptr<float>()[row * 3 + i];
            meanSq += sum[i] * sum[i];
        }
        meanSq /= 3.f;
        const float rms = std::sqrt(meanSq + epsilon);
        for (int i = 0; i < 3; ++i)
        {
            expectedSum[row * 3 + i] = sum[i];
            expectedOut[row * 3 + i] = sum[i] / rms * gammaData[i];
        }
    }

    Mat expectedOutMat = makeMat({1, 2, 3},
        {expectedOut[0], expectedOut[1], expectedOut[2], expectedOut[3], expectedOut[4], expectedOut[5]});
    Mat expectedSumMat = makeMat({1, 2, 3},
        {expectedSum[0], expectedSum[1], expectedSum[2], expectedSum[3], expectedSum[4], expectedSum[5]});

    normAssert(expectedOutMat, outputs[0], "output", 1e-4, 1e-3);
    normAssert(expectedSumMat, outputs[1], "input_skip_bias_sum");
}

// With all 4 ONNX-declared outputs requested, the residual sum must land in the
// *last* output regardless of the (unused) mean/inv_std_var placeholders in between.
TEST(SkipSimplifiedLayerNormalizationLayer, FourOutputsResidualSumIsLast)
{
    const float epsilon = 1e-5f;
    Ptr<Layer> layer = createSkipNorm(epsilon);

    Mat input = makeMat({1, 1, 4}, {1.f, 2.f, 3.f, 4.f});
    Mat skip  = makeMat({1, 1, 4}, {0.1f, 0.2f, 0.3f, 0.4f});
    Mat gamma = makeMat({4}, {1.f, 1.f, 1.f, 1.f});

    std::vector<Mat> inputs = {input, skip, gamma};
    std::vector<Mat> outputs;
    runSkipNorm(layer, inputs, outputs, 4);
    ASSERT_EQ(outputs.size(), (size_t)4);

    float sum[4], meanSq = 0.f;
    for (int i = 0; i < 4; ++i)
    {
        sum[i] = input.ptr<float>()[i] + skip.ptr<float>()[i];
        meanSq += sum[i] * sum[i];
    }
    meanSq /= 4.f;
    const float rms = std::sqrt(meanSq + epsilon);
    float expectedOut[4];
    for (int i = 0; i < 4; ++i)
        expectedOut[i] = sum[i] / rms;

    Mat expectedOutMat = makeMat({1, 1, 4}, {expectedOut[0], expectedOut[1], expectedOut[2], expectedOut[3]});
    Mat expectedSumMat  = makeMat({1, 1, 4}, {sum[0], sum[1], sum[2], sum[3]});

    normAssert(expectedOutMat, outputs[0], "output (outputs[0])", 1e-4, 1e-3);
    normAssert(expectedSumMat, outputs[3], "input_skip_bias_sum (outputs.back())");
}

TEST(SkipSimplifiedLayerNormalizationLayer, RequiresAtLeastThreeInputs)
{
    Ptr<Layer> layer = createSkipNorm();
    std::vector<MatShape> inputs = { MatShape({1, 1, 4}), MatShape({1, 1, 4}) }; // missing gamma
    std::vector<MatShape> outputs, internals;
    EXPECT_ANY_THROW(layer->getMemoryShapes(inputs, 2, outputs, internals));
}

}} // namespace opencv_test::(anonymous)
