// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_VLM_CONFIG_JSON_HPP__
#define __OPENCV_VLM_CONFIG_JSON_HPP__

#include "opencv2/core.hpp"

namespace cv { namespace vlm {

FileStorage openJsonConfigOrThrow(const String& path);

int getInt(const FileNode& node, const String& name, int fallback);
float getFloat(const FileNode& node, const String& name, float fallback);
void getVec3f(const FileNode& node, const String& name, Vec3f& value, const Vec3f& fallback);

// Looks up `name` at the top level of `config`, falling back to `config["text_config"][name]`,
// matching the config layout used by both PaddleOCR-VL and Granite-Docling-258M.
int getIntWithTextConfigFallback(const FileStorage& config, const String& name, int fallback);

}} // namespace cv::vlm

#endif // __OPENCV_VLM_CONFIG_JSON_HPP__
