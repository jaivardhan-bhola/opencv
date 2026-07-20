// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_VLM_MODEL_BASE_HPP__
#define __OPENCV_VLM_MODEL_BASE_HPP__

#include "opencv2/dnn.hpp"
#include "opencv2/vlm.hpp"

namespace cv { namespace vlm {

// Implements the parts of VLMModel that are identical across engines: document-level
// input handling (currently single-image only), per-page KV-cache reset, and
// device forwarding to the three underlying nets. Engine subclasses only need to
// implement infer() plus their own config loading and net construction, then call
// registerNets() once all three nets exist.
class VLMModelBase : public VLMModel
{
public:
    std::vector<String> inferDocument(const String& input_path, const String& prompt,
                                       int max_new_tokens) CV_OVERRIDE;
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

#endif // __OPENCV_VLM_MODEL_BASE_HPP__
