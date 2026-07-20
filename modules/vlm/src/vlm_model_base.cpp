// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "vlm_model_base.hpp"

namespace cv { namespace vlm {

VLMModel::~VLMModel() {}

void VLMModelBase::registerNets(dnn::Net& visionNet, dnn::Net& embedNet, dnn::Net& decoderNet)
{
    visionNet_ = &visionNet;
    embedNet_ = &embedNet;
    decoderNet_ = &decoderNet;
}

void VLMModelBase::reset()
{
    CV_Assert(decoderNet_);
    decoderNet_->resetKVCache();
}

void VLMModelBase::setPreferableDevice(const String& device)
{
    CV_Assert(visionNet_ && embedNet_ && decoderNet_);

    int backendId, targetId;
    if (device == "cpu")
    {
        backendId = dnn::DNN_BACKEND_DEFAULT;
        targetId = dnn::DNN_TARGET_CPU;
    }
    else if (device == "cuda")
    {
        backendId = dnn::DNN_BACKEND_CUDA;
        targetId = dnn::DNN_TARGET_CUDA;
    }
    else
    {
        CV_Error(Error::StsBadArg, "vlm: unknown device '" + device + "' (expected 'cpu' or 'cuda')");
    }

    dnn::Net* nets[] = {visionNet_, embedNet_, decoderNet_};
    for (dnn::Net* net : nets)
    {
        net->setPreferableBackend(backendId);
        net->setPreferableTarget(targetId);
    }
}

std::vector<String> VLMModelBase::inferDocument(const String& input_path, const String& prompt,
                                                 int max_new_tokens)
{
    Mat image = imread(input_path, IMREAD_COLOR);
    if (image.empty())
        CV_Error(Error::StsError, "vlm: could not read input file: " + input_path);

    reset();
    return { infer(image, prompt, max_new_tokens) };
}

}} // namespace cv::vlm
