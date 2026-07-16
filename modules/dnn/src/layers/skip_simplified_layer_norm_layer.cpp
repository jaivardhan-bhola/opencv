// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"

namespace cv { namespace dnn {

// com.microsoft.SkipSimplifiedLayerNormalization: sum = input + skip [+ bias];
// output = RMSNorm(sum) * gamma; also emits sum as input_skip_bias_sum (the residual
// carried into the next block). The RMSNorm math itself is delegated to the existing,
// already-tested RMSNormLayer (same op used for the standalone RMSNormalization/
// SimplifiedLayerNormalization nodes elsewhere in this graph).
class SkipSimplifiedLayerNormalizationLayerImpl CV_FINAL : public SkipSimplifiedLayerNormalizationLayer {
public:
    float epsilon;

    SkipSimplifiedLayerNormalizationLayerImpl(const LayerParams& params) {
        setParamsFrom(params);
        epsilon = params.get<float>("epsilon", 1e-5f);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
        return backendId == DNN_BACKEND_OPENCV;
    }

    virtual void getTypes(const std::vector<MatType>& inputs,
                          const int requiredOutputs,
                          const int requiredInternals,
                          std::vector<MatType>& outputs,
                          std::vector<MatType>& internals) const CV_OVERRIDE {
        outputs.assign(requiredOutputs, inputs[0]);
        internals.clear();
    }

    virtual bool getMemoryShapes(const std::vector<MatShape>& inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape>& outputs,
                                 std::vector<MatShape>& internals) const CV_OVERRIDE {
        CV_CheckGE((int)inputs.size(), 3, "SkipSimplifiedLayerNormalization: expects input, skip, gamma [, bias]");
        // All declared outputs (output, mean, inv_std_var, input_skip_bias_sum) share
        // input's shape; only outputs[0] (normalized) and outputs.back() (residual sum)
        // are ever actually used downstream -- the middle ones (mean/inv_std_var), if
        // requested at all, are unused placeholders.
        outputs.assign(requiredOutputs, inputs[0]);
        internals.clear();
        return false;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE {
        CV_TRACE_FUNCTION();

        if (inputs_arr.depth() == CV_16F) {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        const Mat& input = inputs[0];
        const Mat& skip = inputs[1];
        const Mat& gamma = inputs[2];
        const bool hasBias = inputs.size() > 3 && !inputs[3].empty();

        Mat& sumOut = outputs.back();  // input_skip_bias_sum
        cv::add(input, skip, sumOut);
        if (hasBias) {
            // bias is (hidden_size,), sumOut is (..., hidden_size) -- cv::add's scalar
            // broadcast only covers checkScalar()'s Size(1,1)/Size(1,cn)/Size(cn,1) cases,
            // none of which match a real hidden_size-wide bias, so broadcast explicitly.
            CV_CheckEQ((int)inputs[3].total(), (int)gamma.total(),
                       "SkipSimplifiedLayerNormalization: bias/gamma size mismatch");
            const int hidden = (int)gamma.total();
            const int rows = (int)(sumOut.total() / hidden);
            float* s = sumOut.ptr<float>();
            const float* b = inputs[3].ptr<float>();
            parallel_for_(Range(0, rows), [&](const Range& r) {
                for (int i = r.start; i < r.end; ++i) {
                    float* row = s + (size_t)i * hidden;
                    for (int c = 0; c < hidden; ++c) row[c] += b[c];
                }
            });
        }

        LayerParams rmsParams;
        rmsParams.set("axis", -1);
        rmsParams.set("epsilon", epsilon);
        Ptr<RMSNormLayer> rms = RMSNormLayer::create(rmsParams);
        std::vector<Mat> rmsInputs = {sumOut, gamma};
        std::vector<Mat> rmsOutputs = {outputs[0]};
        std::vector<Mat> rmsInternals;
        rms->forward(rmsInputs, rmsOutputs, rmsInternals);
    }
};

Ptr<SkipSimplifiedLayerNormalizationLayer> SkipSimplifiedLayerNormalizationLayer::create(const LayerParams& params) {
    return makePtr<SkipSimplifiedLayerNormalizationLayerImpl>(params);
}

}} // namespace cv::dnn
