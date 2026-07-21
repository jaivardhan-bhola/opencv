// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"
#include "../src/base64.hpp"
#include "../src/config_json.hpp"
#include "../src/vlm_generation.hpp"
#include "../src/local_vlm_model_base.hpp"

#include <fstream>
#include <cstdio>

namespace opencv_test { namespace {

using namespace cv::vlm;

static std::string b64(const std::string& s)
{
    return base64Encode(reinterpret_cast<const unsigned char*>(s.data()), s.size());
}

TEST(Vlm_Base64, RFC4648TestVectors)
{
    EXPECT_EQ(b64(""), "");
    EXPECT_EQ(b64("f"), "Zg==");
    EXPECT_EQ(b64("fo"), "Zm8=");
    EXPECT_EQ(b64("foo"), "Zm9v");
    EXPECT_EQ(b64("foob"), "Zm9vYg==");
    EXPECT_EQ(b64("fooba"), "Zm9vYmE=");
    EXPECT_EQ(b64("foobar"), "Zm9vYmFy");
}

TEST(Vlm_Base64, FullByteRange)
{
    std::vector<unsigned char> data(256);
    for (int i = 0; i < 256; i++)
        data[i] = (unsigned char)i;

    std::string encoded = base64Encode(data.data(), data.size());
    EXPECT_EQ(encoded.size() % 4, 0u);
    EXPECT_EQ(encoded.size(), ((data.size() + 2) / 3) * 4);
}

TEST(Vlm_ConfigJson, GetIntPresentAndFallback)
{
    FileStorage fs("{\"a\": 5}", FileStorage::READ | FileStorage::MEMORY);
    EXPECT_EQ(getInt(fs.root(), "a", -1), 5);
    EXPECT_EQ(getInt(fs.root(), "missing", -1), -1);
}

TEST(Vlm_ConfigJson, GetFloatPresentAndFallback)
{
    FileStorage fs("{\"a\": 1.5}", FileStorage::READ | FileStorage::MEMORY);
    EXPECT_FLOAT_EQ(getFloat(fs.root(), "a", -1.f), 1.5f);
    EXPECT_FLOAT_EQ(getFloat(fs.root(), "missing", -1.f), -1.f);
}

TEST(Vlm_ConfigJson, GetVec3fPresentAndFallback)
{
    FileStorage fs("{\"a\": [1.0, 2.0, 3.0]}", FileStorage::READ | FileStorage::MEMORY);

    Vec3f value;
    getVec3f(fs.root(), "a", value, Vec3f(0.f, 0.f, 0.f));
    EXPECT_EQ(value, Vec3f(1.f, 2.f, 3.f));

    Vec3f fallbackValue;
    getVec3f(fs.root(), "missing", fallbackValue, Vec3f(9.f, 8.f, 7.f));
    EXPECT_EQ(fallbackValue, Vec3f(9.f, 8.f, 7.f));
}

TEST(Vlm_ConfigJson, GetVec3fWrongSizeThrows)
{
    FileStorage fs("{\"a\": [1.0, 2.0]}", FileStorage::READ | FileStorage::MEMORY);
    Vec3f value;
    EXPECT_THROW(getVec3f(fs.root(), "a", value, Vec3f(0.f, 0.f, 0.f)), cv::Exception);
}

TEST(Vlm_ConfigJson, GetIntWithTextConfigFallback)
{
    FileStorage topLevel("{\"eos_token_id\": 2}", FileStorage::READ | FileStorage::MEMORY);
    EXPECT_EQ(getIntWithTextConfigFallback(topLevel, "eos_token_id", -1), 2);

    FileStorage nested("{\"text_config\": {\"eos_token_id\": 7}}", FileStorage::READ | FileStorage::MEMORY);
    EXPECT_EQ(getIntWithTextConfigFallback(nested, "eos_token_id", -1), 7);

    FileStorage neither("{}", FileStorage::READ | FileStorage::MEMORY);
    EXPECT_EQ(getIntWithTextConfigFallback(neither, "eos_token_id", -1), -1);
}

TEST(Vlm_ConfigJson, GetIntWithTextConfigFallbackPrefersTopLevel)
{
    FileStorage fs("{\"eos_token_id\": 2, \"text_config\": {\"eos_token_id\": 7}}",
                   FileStorage::READ | FileStorage::MEMORY);
    EXPECT_EQ(getIntWithTextConfigFallback(fs, "eos_token_id", -1), 2);
}

TEST(Vlm_ConfigJson, OpenJsonConfigOrThrow_NonexistentPath)
{
    EXPECT_THROW(openJsonConfigOrThrow("/nonexistent/path/config.json"), cv::Exception);
}

TEST(Vlm_ConfigJson, OpenJsonConfigOrThrow_ValidFile)
{
    std::string path = cv::tempfile("vlm_test_config.json");
    {
        std::ofstream ofs(path.c_str());
        ofs << "{\"hello\": \"world\"}";
    }

    FileStorage fs = openJsonConfigOrThrow(path);
    std::string value;
    fs["hello"] >> value;
    EXPECT_EQ(value, "world");
    fs.release();

    remove(path.c_str());
}

TEST(Vlm_Generation, ArgmaxLastToken)
{
    int sizes[] = {1, 2, 4};
    Mat logits(3, sizes, CV_32F, Scalar(0));

    float* row0 = logits.ptr<float>(0, 0);
    row0[0] = 100.f; row0[1] = 100.f; row0[2] = 100.f; row0[3] = 100.f;

    float* row1 = logits.ptr<float>(0, 1);
    row1[0] = 0.1f; row1[1] = 0.2f; row1[2] = 5.0f; row1[3] = -1.0f;

    EXPECT_EQ(argmaxLastToken(logits), 2);
}

TEST(Vlm_Generation, ArgmaxLastTokenTieBreaksToFirst)
{
    int sizes[] = {1, 1, 3};
    Mat logits(3, sizes, CV_32F, Scalar(0));
    float* row = logits.ptr<float>(0, 0);
    row[0] = 5.f; row[1] = 5.f; row[2] = 1.f;

    EXPECT_EQ(argmaxLastToken(logits), 0);
}

class TestLocalVLMModel : public LocalVLMModelBase
{
public:
    using LocalVLMModelBase::registerNets;

    String infer(InputArray, const String&, int) CV_OVERRIDE { return String(); }
};

TEST(Vlm_LocalModelBase, ResetWithoutNetsThrows)
{
    TestLocalVLMModel model;
    EXPECT_THROW(model.reset(), cv::Exception);
}

TEST(Vlm_LocalModelBase, SetDeviceWithoutNetsThrows)
{
    TestLocalVLMModel model;
    EXPECT_THROW(model.setPreferableDevice("cpu"), cv::Exception);
}

TEST(Vlm_LocalModelBase, SetDeviceUnknownDeviceThrows)
{
    TestLocalVLMModel model;
    dnn::Net vision, embed, decoder;
    model.registerNets(vision, embed, decoder);
    EXPECT_THROW(model.setPreferableDevice("opencl"), cv::Exception);
}

TEST(Vlm_LocalModelBase, SetDeviceCpuAndCudaDoNotThrow)
{
    TestLocalVLMModel model;
    dnn::Net vision, embed, decoder;
    model.registerNets(vision, embed, decoder);
    EXPECT_NO_THROW(model.setPreferableDevice("cpu"));
    EXPECT_NO_THROW(model.setPreferableDevice("cuda"));
}

}} // namespace opencv_test::(anonymous)
