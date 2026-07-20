// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_VLM_CLOUD_ENGINE_HPP__
#define __OPENCV_VLM_CLOUD_ENGINE_HPP__

#include "opencv2/vlm.hpp"

namespace cv { namespace vlm {

Ptr<VLMModel> createCloudModel(VLMModelType model_type, const String& model_name, const String& api_key);

}} // namespace cv::vlm

#endif // __OPENCV_VLM_CLOUD_ENGINE_HPP__
