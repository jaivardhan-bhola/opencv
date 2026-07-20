// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_VLM_LOCAL_MODEL_BASE_HPP__
#define __OPENCV_VLM_LOCAL_MODEL_BASE_HPP__

#include "opencv2/dnn.hpp"
#include "vlm_model_base.hpp"

namespace cv { namespace vlm {

// Adds the parts shared by engines that run locally from ONNX weights: per-page KV-cache
// reset and device forwarding to the three underlying nets. Local engine subclasses only
// need to implement infer() plus their own config loading and net construction, then call
// registerNets() once all three nets exist.
class LocalVLMModelBase : public VLMModelBase
{
public:
    void reset() CV_OVERRIDE;
    void setPreferableDevice(const String& device) CV_OVERRIDE;

protected:
    void registerNets(dnn::Net& visionNet, dnn::Net& embedNet, dnn::Net& decoderNet);

private:
    dnn::Net* visionNet_ = nullptr;
    dnn::Net* embedNet_ = nullptr;
    dnn::Net* decoderNet_ = nullptr;
};

}} // namespace cv::vlm

#endif // __OPENCV_VLM_LOCAL_MODEL_BASE_HPP__
