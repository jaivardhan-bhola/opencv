// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_VLM_GENERATION_HPP__
#define __OPENCV_VLM_GENERATION_HPP__

#include "opencv2/core.hpp"
#include "opencv2/dnn.hpp"

namespace cv { namespace vlm {

int argmaxLastToken(const Mat& logits);

// Runs the shared KV-cache prefill/decode loop common to the PaddleOCR-VL and
// Granite-Docling-258M decoders: prefill with promptInputsEmbeds, then greedily decode
// (argmax) one token at a time via embedNet + decoderNet until eosTokenId is produced or
// maxNewTokens is reached. Calls decoderNet.enableKVCache() itself; caller is responsible
// for resetting decoderNet's KV-cache (resetKVCache()) between independent generations.
std::vector<int> generateWithKVCache(dnn::Net& embedNet, dnn::Net& decoderNet,
                                      const Mat& promptInputsEmbeds, int promptLen,
                                      int maxNewTokens, int eosTokenId);

}} // namespace cv::vlm

#endif // __OPENCV_VLM_GENERATION_HPP__
