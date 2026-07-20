// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_VLM_HTTP_CLIENT_HPP__
#define __OPENCV_VLM_HTTP_CLIENT_HPP__

#include <string>
#include <vector>

namespace cv { namespace vlm {

struct HttpResponse
{
    long statusCode;
    std::string body;
};

// POSTs jsonBody to url with the given "Name: Value" headers over HTTPS via libcurl.
// Throws cv::Exception if the transport itself fails (DNS/connect/TLS/etc.), or if OpenCV
// was built without libcurl available. Non-2xx HTTP responses are NOT thrown here -- the
// caller inspects HttpResponse::statusCode/body.
HttpResponse httpPostJson(const std::string& url, const std::string& jsonBody,
                          const std::vector<std::string>& headers);

}} // namespace cv::vlm

#endif // __OPENCV_VLM_HTTP_CLIENT_HPP__
