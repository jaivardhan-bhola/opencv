// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "engines/paddleocr_vl_engine.hpp"
#include "engines/granite_docling_engine.hpp"

namespace cv { namespace vlm {

namespace {

int engineFromString(const String& engine)
{
    if (engine == "new")
        return dnn::ENGINE_NEW;
    if (engine == "ort")
        return dnn::ENGINE_ORT;
    CV_Error(Error::StsBadArg, "vlm: unknown engine '" + engine + "' (expected 'new' or 'ort')");
}

} // namespace

Ptr<VLMModel> create(VLMModelType model_type, const String& model_dir,
                      const String& engine, const String& device)
{
    int engineId = engineFromString(engine);
    switch (model_type)
    {
    case VLM_MODEL_PADDLEOCR_VL:
        return createPaddleOCRVLModel(model_dir, engineId, device);
    case VLM_MODEL_GRANITE_DOCLING:
        return createGraniteDoclingModel(model_dir, engineId, device);
    default:
        CV_Error(Error::StsBadArg, cv::format("vlm: unknown VLMModelType: %d", (int)model_type));
    }
}

}} // namespace cv::vlm
