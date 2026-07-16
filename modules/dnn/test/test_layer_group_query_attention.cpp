// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

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

static Mat makeInt32Mat(std::initializer_list<int> shape, std::initializer_list<int32_t> data)
{
    std::vector<int> shp(shape);
    std::vector<int32_t> buf(data);
    size_t total = 1;
    for (int s : shp) total *= (size_t)s;
    CV_Assert(total == buf.size());
    return Mat((int)shp.size(), shp.data(), CV_32S, buf.data()).clone();
}

static Mat dummyTotalSeqLen()
{
    return Mat(std::vector<int>{1}, CV_32S, Scalar(0));
}

static Mat dummyCache()
{
    return Mat(std::vector<int>{1, 1}, CV_32F, Scalar(0));
}

static Ptr<Layer> createGQA(int numHeads, int kvNumHeads, float scale = 0.f,
                             int localWindow = -1, float softcap = 0.f,
                             bool doRotary = false, bool interleaved = false)
{
    LayerParams lp;
    lp.type = "GroupQueryAttention";
    lp.name = "test_gqa";
    lp.set("num_heads", numHeads);
    lp.set("kv_num_heads", kvNumHeads);
    lp.set("scale", scale);
    lp.set("local_window_size", localWindow);
    lp.set("softcap", softcap);
    lp.set("do_rotary", doRotary ? 1 : 0);
    lp.set("rotary_interleaved", interleaved ? 1 : 0);
    Ptr<Layer> layer = LayerFactory::createLayerInstance("GroupQueryAttention", lp);
    CV_Assert(layer);
    return layer;
}

TEST(GroupQueryAttentionLayer, CausalSelfAttentionNoCache)
{
    Ptr<Layer> layer = createGQA(1, 1);

    Mat query = makeMat({1, 2, 2}, {1, 0, 0, 1});
    Mat key   = makeMat({1, 2, 2}, {1, 0, 0, 1});
    Mat value = makeMat({1, 2, 2}, {1, 2, 3, 4});
    Mat pastKey, pastValue;
    Mat seqlensK = makeInt32Mat({1}, {1});
    Mat totalSeqLen = dummyTotalSeqLen();
    Mat cosCache = dummyCache();
    Mat sinCache = dummyCache();

    std::vector<Mat> inputs = {query, key, value, pastKey, pastValue, seqlensK,
                               totalSeqLen, cosCache, sinCache};
    std::vector<Mat> outputs;
    runLayer(layer, inputs, outputs);
    ASSERT_EQ(outputs.size(), (size_t)3);

    const float scale = 1.f / std::sqrt(2.f);
    const float s0 = 0.f * scale;
    const float s1 = 1.f * scale;
    const float mx = std::max(s0, s1);
    const float e0 = std::exp(s0 - mx), e1 = std::exp(s1 - mx);
    const float a0 = e0 / (e0 + e1), a1 = e1 / (e0 + e1);

    Mat expectedOut = makeMat({1, 2, 2}, {1.f, 2.f, a0 * 1.f + a1 * 3.f, a0 * 2.f + a1 * 4.f});
    Mat expectedPresentKey = makeMat({1, 1, 2, 2}, {1, 0, 0, 1});
    Mat expectedPresentValue = makeMat({1, 1, 2, 2}, {1, 2, 3, 4});

    normAssert(expectedOut, outputs[0], "output", 1e-4, 1e-3);
    normAssert(expectedPresentKey, outputs[1], "present_key");
    normAssert(expectedPresentValue, outputs[2], "present_value");
}

TEST(GroupQueryAttentionLayer, GroupedHeadsMapToCorrectKVHead)
{
    Ptr<Layer> layer = createGQA(4, 2);

    Mat query = makeMat({1, 1, 4}, {1, 1, 1, 1});
    Mat key   = makeMat({1, 1, 2}, {1, 1});
    Mat value = makeMat({1, 1, 2}, {10, 20});
    Mat pastKey, pastValue;
    Mat seqlensK = makeInt32Mat({1}, {0});
    Mat totalSeqLen = dummyTotalSeqLen();
    Mat cosCache = dummyCache();
    Mat sinCache = dummyCache();

    std::vector<Mat> inputs = {query, key, value, pastKey, pastValue, seqlensK,
                               totalSeqLen, cosCache, sinCache};
    std::vector<Mat> outputs;
    runLayer(layer, inputs, outputs);

    Mat expectedOut = makeMat({1, 1, 4}, {10, 10, 20, 20});
    Mat expectedPresentKey = makeMat({1, 2, 1, 1}, {1, 1});
    Mat expectedPresentValue = makeMat({1, 2, 1, 1}, {10, 20});

    normAssert(expectedOut, outputs[0], "output");
    normAssert(expectedPresentKey, outputs[1], "present_key");
    normAssert(expectedPresentValue, outputs[2], "present_value");
}

TEST(GroupQueryAttentionLayer, PresentKVConcatenatesPastAndNew)
{
    Ptr<Layer> layer = createGQA(1, 1);

    Mat pastKey = makeMat({2, 1, 2, 2}, {1, 2, 3, 4,   5, 6, 7, 8});
    Mat pastValue = makeMat({2, 1, 2, 2}, {9, 10, 11, 12,   13, 14, 15, 16});
    Mat key   = makeMat({2, 1, 2}, {100, 101,   102, 103});
    Mat value = makeMat({2, 1, 2}, {200, 201,   202, 203});
    Mat query = makeMat({2, 1, 2}, {0, 0, 0, 0});
    Mat seqlensK = makeInt32Mat({2}, {2, 2});
    Mat totalSeqLen = dummyTotalSeqLen();
    Mat cosCache = dummyCache();
    Mat sinCache = dummyCache();

    std::vector<Mat> inputs = {query, key, value, pastKey, pastValue, seqlensK,
                               totalSeqLen, cosCache, sinCache};
    std::vector<Mat> outputs;
    runLayer(layer, inputs, outputs);

    Mat expectedPresentKey = makeMat({2, 1, 3, 2},
        {1, 2, 3, 4, 100, 101,   5, 6, 7, 8, 102, 103});
    Mat expectedPresentValue = makeMat({2, 1, 3, 2},
        {9, 10, 11, 12, 200, 201,   13, 14, 15, 16, 202, 203});

    normAssert(expectedPresentKey, outputs[1], "present_key");
    normAssert(expectedPresentValue, outputs[2], "present_value");
}

TEST(GroupQueryAttentionLayer, LeftPaddingExcludesGarbageFromAttention)
{
    Ptr<Layer> layer = createGQA(1, 1);

    Mat pastKey = makeMat({1, 1, 2, 1}, {999.f, 2.f});
    Mat pastValue = makeMat({1, 1, 2, 1}, {500.f, 100.f});
    Mat key = makeMat({1, 1, 1}, {3.f});
    Mat value = makeMat({1, 1, 1}, {9.f});
    Mat query = makeMat({1, 1, 1}, {1.f});
    Mat seqlensK = makeInt32Mat({1}, {1});
    Mat totalSeqLen = dummyTotalSeqLen();
    Mat cosCache = dummyCache();
    Mat sinCache = dummyCache();

    std::vector<Mat> inputs = {query, key, value, pastKey, pastValue, seqlensK,
                               totalSeqLen, cosCache, sinCache};
    std::vector<Mat> outputs;
    runLayer(layer, inputs, outputs);

    const float s1 = 1.f * 2.f;
    const float s2 = 1.f * 3.f;
    const float mx = std::max(s1, s2);
    const float e1 = std::exp(s1 - mx), e2 = std::exp(s2 - mx);
    const float expected = (e1 * 100.f + e2 * 9.f) / (e1 + e2);

    Mat expectedOut = makeMat({1, 1, 1}, {expected});
    Mat expectedPresentKey = makeMat({1, 1, 3, 1}, {999.f, 2.f, 3.f});
    Mat expectedPresentValue = makeMat({1, 1, 3, 1}, {500.f, 100.f, 9.f});

    normAssert(expectedOut, outputs[0], "output", 1e-4, 1e-3);
    normAssert(expectedPresentKey, outputs[1], "present_key");
    normAssert(expectedPresentValue, outputs[2], "present_value");
}

TEST(GroupQueryAttentionLayer, LocalWindowRestrictsAttentionRange)
{
    Ptr<Layer> layer = createGQA(1, 1, /*scale*/0.f, /*localWindow*/1);

    Mat pastKey = makeMat({1, 1, 3, 1}, {1000.f, 2000.f, 5.f});
    Mat pastValue = makeMat({1, 1, 3, 1}, {9000.f, 9001.f, 100.f});
    Mat key = makeMat({1, 1, 1}, {6.f});
    Mat value = makeMat({1, 1, 1}, {110.f});
    Mat query = makeMat({1, 1, 1}, {1.f});
    Mat seqlensK = makeInt32Mat({1}, {3});
    Mat totalSeqLen = dummyTotalSeqLen();
    Mat cosCache = dummyCache();
    Mat sinCache = dummyCache();

    std::vector<Mat> inputs = {query, key, value, pastKey, pastValue, seqlensK,
                               totalSeqLen, cosCache, sinCache};
    std::vector<Mat> outputs;
    runLayer(layer, inputs, outputs);

    const float s2 = 1.f * 5.f;
    const float s3 = 1.f * 6.f;
    const float mx = std::max(s2, s3);
    const float e2 = std::exp(s2 - mx), e3 = std::exp(s3 - mx);
    const float expected = (e2 * 100.f + e3 * 110.f) / (e2 + e3);

    Mat expectedOut = makeMat({1, 1, 1}, {expected});
    normAssert(expectedOut, outputs[0], "output", 1e-4, 1e-3);
}

TEST(GroupQueryAttentionLayer, SoftcapClampsScores)
{
    Ptr<Layer> layer = createGQA(1, 1, /*scale*/0.f, /*localWindow*/-1, /*softcap*/5.f);

    Mat pastKey = makeMat({1, 1, 1, 1}, {10.f});
    Mat pastValue = makeMat({1, 1, 1, 1}, {100.f});
    Mat key = makeMat({1, 1, 1}, {1.f});
    Mat value = makeMat({1, 1, 1}, {9.f});
    Mat query = makeMat({1, 1, 1}, {2.f});
    Mat seqlensK = makeInt32Mat({1}, {1});
    Mat totalSeqLen = dummyTotalSeqLen();
    Mat cosCache = dummyCache();
    Mat sinCache = dummyCache();

    std::vector<Mat> inputs = {query, key, value, pastKey, pastValue, seqlensK,
                               totalSeqLen, cosCache, sinCache};
    std::vector<Mat> outputs;
    runLayer(layer, inputs, outputs);

    const float softcap = 5.f;
    const float raw0 = 2.f * 10.f, raw1 = 2.f * 1.f;
    const float c0 = softcap * std::tanh(raw0 / softcap);
    const float c1 = softcap * std::tanh(raw1 / softcap);
    const float mx = std::max(c0, c1);
    const float e0 = std::exp(c0 - mx), e1 = std::exp(c1 - mx);
    const float expected = (e0 * 100.f + e1 * 9.f) / (e0 + e1);

    Mat expectedOut = makeMat({1, 1, 1}, {expected});
    normAssert(expectedOut, outputs[0], "output", 1e-4, 1e-3);
}

TEST(GroupQueryAttentionLayer, RotaryAppliesOnlyToNewTokens)
{
    Ptr<Layer> layer = createGQA(1, 1, /*scale*/0.f, /*localWindow*/-1, /*softcap*/0.f,
                                 /*doRotary*/true, /*interleaved*/false);

    Mat pastKey = makeMat({1, 1, 1, 2}, {5.f, 7.f});
    Mat pastValue = makeMat({1, 1, 1, 2}, {100.f, 200.f});
    Mat key = makeMat({1, 1, 2}, {1.f, 0.f});
    Mat value = makeMat({1, 1, 2}, {9.f, 11.f});
    Mat query = makeMat({1, 1, 2}, {1.f, 0.f});
    Mat seqlensK = makeInt32Mat({1}, {1});
    Mat totalSeqLen = dummyTotalSeqLen();
    Mat cosCache = makeMat({2, 1}, {1.f, 0.f});
    Mat sinCache = makeMat({2, 1}, {0.f, 1.f});

    std::vector<Mat> inputs = {query, key, value, pastKey, pastValue, seqlensK,
                               totalSeqLen, cosCache, sinCache};
    std::vector<Mat> outputs;
    runLayer(layer, inputs, outputs);

    const float cosP = 0.f, sinP = 1.f;
    const float qRotReal = 1.f * cosP - 0.f * sinP, qRotImag = 0.f * cosP + 1.f * sinP;
    const float kRotReal = 1.f * cosP - 0.f * sinP, kRotImag = 0.f * cosP + 1.f * sinP;

    Mat expectedPresentKey = makeMat({1, 1, 2, 2}, {5.f, 7.f, kRotReal, kRotImag});
    Mat expectedPresentValue = makeMat({1, 1, 2, 2}, {100.f, 200.f, 9.f, 11.f});

    const float scale = 1.f / std::sqrt(2.f);
    const float s0 = (qRotReal * 5.f + qRotImag * 7.f) * scale;
    const float s1 = (qRotReal * kRotReal + qRotImag * kRotImag) * scale;
    const float mx = std::max(s0, s1);
    const float e0 = std::exp(s0 - mx), e1 = std::exp(s1 - mx);
    const float a0 = e0 / (e0 + e1), a1 = e1 / (e0 + e1);
    Mat expectedOut = makeMat({1, 1, 2},
        {a0 * 100.f + a1 * 9.f, a0 * 200.f + a1 * 11.f});

    normAssert(expectedPresentKey, outputs[1], "present_key");
    normAssert(expectedPresentValue, outputs[2], "present_value");
    normAssert(expectedOut, outputs[0], "output", 1e-4, 1e-3);
}

TEST(GroupQueryAttentionLayer, InvalidParamsThrow)
{
    auto build = [](int nh, int kvh) {
        LayerParams lp;
        lp.type = "GroupQueryAttention";
        lp.name = "bad_gqa";
        lp.set("num_heads", nh);
        lp.set("kv_num_heads", kvh);
        return LayerFactory::createLayerInstance("GroupQueryAttention", lp);
    };
    EXPECT_ANY_THROW(build(0, 1));
    EXPECT_ANY_THROW(build(2, 0));
    EXPECT_ANY_THROW(build(3, 2));
}

TEST(GroupQueryAttentionLayer, MemoryShapesAccountForPastCache)
{
    Ptr<Layer> layer = createGQA(2, 1);

    std::vector<MatShape> inputs(9);
    inputs[0] = MatShape({2, 3, 4});
    inputs[1] = MatShape({2, 3, 2});
    inputs[2] = MatShape({2, 3, 2});
    inputs[3] = MatShape({2, 1, 5, 2});
    inputs[4] = MatShape({2, 1, 5, 2});
    inputs[5] = MatShape({2});
    inputs[6] = MatShape({1});
    inputs[7] = MatShape({10, 1});
    inputs[8] = MatShape({10, 1});

    std::vector<MatShape> outputs, internals;
    layer->getMemoryShapes(inputs, 0, outputs, internals);
    ASSERT_EQ(outputs.size(), (size_t)3);
    EXPECT_EQ(outputs[0], MatShape({2, 3, 4}));
    EXPECT_EQ(outputs[1], MatShape({2, 1, 8, 2}));
    EXPECT_EQ(outputs[2], MatShape({2, 1, 8, 2}));
}

}} // namespace opencv_test::(anonymous)
