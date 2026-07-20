// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_VLM_MODEL_BASE_HPP__
#define __OPENCV_VLM_MODEL_BASE_HPP__

#include "opencv2/vlm.hpp"

namespace cv { namespace vlm {

// Implements the one part of VLMModel that's identical across every engine, local or
// cloud: document-level input handling (currently single-image only). reset() and
// setPreferableDevice() stay pure virtual here since their meaning differs between
// on-device engines (see LocalVLMModelBase) and cloud engines (no-ops, see cloud_engine.cpp).
class VLMModelBase : public VLMModel
{
public:
    std::vector<String> inferDocument(const String& input_path, const String& prompt,
                                       int max_new_tokens) CV_OVERRIDE;
};

}} // namespace cv::vlm

#endif // __OPENCV_VLM_MODEL_BASE_HPP__
