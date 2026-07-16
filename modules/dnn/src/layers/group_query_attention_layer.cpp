// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <cmath>

namespace cv { namespace dnn {

// com.microsoft.GroupQueryAttention (fused causal GQA + internal RoPE + KV-cache concat).
// https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.GroupQueryAttention
//
// Inputs: query (B,S,H*D), key (B,S,kvH*D), value (B,S,kvH*D),
//         past_key (B,kvH,Sp,D) [optional], past_value (B,kvH,Sp,D) [optional],
//         seqlens_k (B) [int, = valid_length-1 per batch row], total_sequence_length (scalar, unused here --
//         derived directly from past/new seq lengths instead), cos_cache (max_pos, D/2), sin_cache (max_pos, D/2).
// Outputs: output (B,S,H*D), present_key (B,kvH,Sp+S,D), present_value (B,kvH,Sp+S,D).
//
// seqlens_k/attention_mask convention: ORT's GQA expects any padding to be at the FRONT of the
// KV buffer (left-padding) -- valid_length = seqlens_k[b]+1, padding_offset = total_kv_len -
// valid_length. A query at new-sequence index i has absolute (unpadded) position
// valid_length - S + i, used both for its RoPE angle and as the causal attention boundary.
class GroupQueryAttentionLayerImpl CV_FINAL : public GroupQueryAttentionLayer {
public:
    int num_heads = 0;
    int kv_num_heads = 0;
    float scale = 0.f;
    int local_window_size = -1;
    float softcap = 0.f;
    bool do_rotary = false;
    bool rotary_interleaved = false;

    GroupQueryAttentionLayerImpl(const LayerParams& params) {
        setParamsFrom(params);
        num_heads = params.get<int>("num_heads");
        kv_num_heads = params.get<int>("kv_num_heads");
        scale = params.get<float>("scale", 0.f);
        local_window_size = params.get<int>("local_window_size", -1);
        softcap = params.get<float>("softcap", 0.f);
        do_rotary = params.get<int>("do_rotary", 0) != 0;
        rotary_interleaved = params.get<int>("rotary_interleaved", 0) != 0;
        CV_CheckGT(num_heads, 0, "GroupQueryAttention: num_heads must be > 0");
        CV_CheckGT(kv_num_heads, 0, "GroupQueryAttention: kv_num_heads must be > 0");
        CV_CheckEQ(num_heads % kv_num_heads, 0, "GroupQueryAttention: num_heads must be a multiple of kv_num_heads");
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
        return backendId == DNN_BACKEND_OPENCV;
    }

    virtual void getTypes(const std::vector<MatType>& inputs,
                          const int requiredOutputs,
                          const int requiredInternals,
                          std::vector<MatType>& outputs,
                          std::vector<MatType>& internals) const CV_OVERRIDE {
        outputs.assign(3, CV_32F);
        internals.clear();
    }

    virtual bool getMemoryShapes(const std::vector<MatShape>& inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape>& outputs,
                                 std::vector<MatShape>& internals) const CV_OVERRIDE {
        CV_CheckGE((int)inputs.size(), 9, "GroupQueryAttention: expects 9 inputs");
        const MatShape& q = inputs[0];
        CV_CheckEQ(q.dims, 3, "GroupQueryAttention: query must be 3D (B,S,H*D)");
        int B = q[0], S = q[1];
        int D = q[2] / num_heads;
        int Sp = 0;
        const MatShape& pastKey = inputs[3];
        if (pastKey.dims == 4) Sp = pastKey[2];

        outputs.resize(3);
        outputs[0] = MatShape{B, S, num_heads * D};
        outputs[1] = MatShape{B, kv_num_heads, Sp + S, D};
        outputs[2] = MatShape{B, kv_num_heads, Sp + S, D};
        internals.clear();
        return false;
    }

    // (B,S,nH*D) -> flat (B,nH,S,D) row-major buffer.
    static void splitHeads(const Mat& x, int B, int S, int nH, int D, std::vector<float>& out) {
        out.resize((size_t)B * nH * S * D);
        const float* src = x.ptr<float>();
        for (int b = 0; b < B; ++b) {
            for (int s = 0; s < S; ++s) {
                const float* row = src + ((size_t)b * S + s) * nH * D;
                for (int h = 0; h < nH; ++h) {
                    float* dst = out.data() + (((size_t)b * nH + h) * S + s) * D;
                    std::memcpy(dst, row + (size_t)h * D, sizeof(float) * D);
                }
            }
        }
    }

    // Applies RoPE in place to a flat (B,nH,S,D) buffer by delegating to the existing,
    // already-tested RotaryEmbeddingLayer (reused via its public forward() API).
    void applyRotary(std::vector<float>& buf, int B, int nH, int S, int D,
                     const Mat& cosCache, const Mat& sinCache, const Mat& positionIds) const {
        int sizes4[4] = {B, nH, S, D};
        Mat x(4, sizes4, CV_32F, buf.data());

        LayerParams lp;
        lp.set("num_heads", nH);
        lp.set("interleaved", rotary_interleaved ? 1 : 0);
        Ptr<RotaryEmbeddingLayer> rope = RotaryEmbeddingLayer::create(lp);

        std::vector<Mat> ropeInputs = {x, cosCache, sinCache, positionIds};
        std::vector<Mat> ropeOutputs = {Mat(4, sizes4, CV_32F)};
        int dhalf = static_cast<int>(cosCache.size[cosCache.dims - 1]);
        std::vector<Mat> ropeInternals = {
            Mat(std::vector<int>{B, S, dhalf}, CV_32F),
            Mat(std::vector<int>{B, S, dhalf}, CV_32F),
        };
        rope->forward(ropeInputs, ropeOutputs, ropeInternals);
        std::memcpy(buf.data(), ropeOutputs[0].ptr<float>(), sizeof(float) * buf.size());
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE {
        CV_TRACE_FUNCTION();

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        const Mat& query = inputs[0];
        const Mat& key = inputs[1];
        const Mat& value = inputs[2];
        const Mat& pastKey = inputs[3];
        const Mat& pastValue = inputs[4];
        const Mat& seqlensK = inputs[5];
        const Mat& cosCache = inputs[7];
        const Mat& sinCache = inputs[8];

        const int B = query.size[0];
        const int S = query.size[1];
        const int D = query.size[2] / num_heads;
        const int Sp = (pastKey.dims == 4) ? pastKey.size[2] : 0;
        const int Skv = Sp + S;
        const int groupSize = num_heads / kv_num_heads;

        std::vector<float> Q, Knew, Vnew;
        splitHeads(query, B, S, num_heads, D, Q);
        splitHeads(key, B, S, kv_num_heads, D, Knew);
        splitHeads(value, B, S, kv_num_heads, D, Vnew);

        // Absolute (unpadded) position of each new token, per batch: seqlens_k[b] - S + 1 + i.
        Mat positionIds(std::vector<int>{B, S}, CV_MAKETYPE(CV_64S, 1));
        std::vector<int> validLen(B), padOffset(B);
        {
            int64_t* posPtr = positionIds.ptr<int64_t>();
            for (int b = 0; b < B; ++b) {
                int64_t sk = 0;
                if (seqlensK.depth() == CV_32S) sk = seqlensK.ptr<int32_t>()[b];
                else sk = seqlensK.ptr<int64_t>()[b];
                validLen[b] = static_cast<int>(sk) + 1;
                padOffset[b] = Skv - validLen[b];
                int64_t base = sk - S + 1;
                for (int i = 0; i < S; ++i) {
                    int64_t pos = base + i;
                    posPtr[(size_t)b * S + i] = std::max<int64_t>(pos, 0);
                }
            }
        }

        if (do_rotary) {
            applyRotary(Q, B, num_heads, S, D, cosCache, sinCache, positionIds);
            applyRotary(Knew, B, kv_num_heads, S, D, cosCache, sinCache, positionIds);
        }

        // present_key/value = concat(past, new) along the sequence axis, per (b, kv-head).
        Mat& presentKey = outputs[1];
        Mat& presentValue = outputs[2];
        {
            float* pk = presentKey.ptr<float>();
            float* pv = presentValue.ptr<float>();
            const float* pastKeyPtr = (Sp > 0) ? pastKey.ptr<float>() : nullptr;
            const float* pastValuePtr = (Sp > 0) ? pastValue.ptr<float>() : nullptr;
            for (int b = 0; b < B; ++b) {
                for (int h = 0; h < kv_num_heads; ++h) {
                    float* dstK = pk + (((size_t)b * kv_num_heads + h) * Skv) * D;
                    float* dstV = pv + (((size_t)b * kv_num_heads + h) * Skv) * D;
                    if (Sp > 0) {
                        const float* srcK = pastKeyPtr + (((size_t)b * kv_num_heads + h) * Sp) * D;
                        const float* srcV = pastValuePtr + (((size_t)b * kv_num_heads + h) * Sp) * D;
                        std::memcpy(dstK, srcK, sizeof(float) * Sp * D);
                        std::memcpy(dstV, srcV, sizeof(float) * Sp * D);
                    }
                    const float* newK = Knew.data() + (((size_t)b * kv_num_heads + h) * S) * D;
                    const float* newV = Vnew.data() + (((size_t)b * kv_num_heads + h) * S) * D;
                    std::memcpy(dstK + (size_t)Sp * D, newK, sizeof(float) * S * D);
                    std::memcpy(dstV + (size_t)Sp * D, newV, sizeof(float) * S * D);
                }
            }
        }

        const float effScale = (scale > 0.f) ? scale : (1.f / std::sqrt(static_cast<float>(D)));

        // Causal grouped-query attention: query head h reads KV head h / groupSize.
        std::vector<float> outHeadsMajor((size_t)B * num_heads * S * D);
        parallel_for_(Range(0, B * num_heads), [&](const Range& r) {
            std::vector<float> scores(Skv);
            for (int bh = r.start; bh < r.end; ++bh) {
                const int b = bh / num_heads;
                const int h = bh % num_heads;
                const int kvh = h / groupSize;
                const int validLenB = validLen[b];
                const int padOffB = padOffset[b];

                const float* Qbh = Q.data() + (size_t)bh * S * D;
                const float* Kbh = presentKey.ptr<float>() + (((size_t)b * kv_num_heads + kvh) * Skv) * D;
                const float* Vbh = presentValue.ptr<float>() + (((size_t)b * kv_num_heads + kvh) * Skv) * D;
                float* outBh = outHeadsMajor.data() + (size_t)bh * S * D;

                for (int i = 0; i < S; ++i) {
                    const int queryPos = validLenB - S + i;             // absolute, unpadded position
                    int hi = padOffB + queryPos;                        // inclusive causal boundary (buffer index)
                    int lo = padOffB;                                   // start of valid (non-padding) region
                    if (local_window_size >= 0) lo = std::max(lo, hi - local_window_size);

                    const float* Qi = Qbh + (size_t)i * D;
                    float mx = -FLT_MAX;
                    for (int j = lo; j <= hi; ++j) {
                        const float* Kj = Kbh + (size_t)j * D;
                        float s = 0.f;
                        for (int d = 0; d < D; ++d) s += Qi[d] * Kj[d];
                        s *= effScale;
                        if (softcap > 0.f) s = softcap * std::tanh(s / softcap);
                        scores[j] = s;
                        if (s > mx) mx = s;
                    }
                    float sum = 0.f;
                    for (int j = lo; j <= hi; ++j) {
                        float e = std::exp(scores[j] - mx);
                        scores[j] = e;
                        sum += e;
                    }
                    const float invSum = 1.f / sum;
                    float* outRow = outBh + (size_t)i * D;
                    for (int d = 0; d < D; ++d) outRow[d] = 0.f;
                    for (int j = lo; j <= hi; ++j) {
                        const float a = scores[j] * invSum;
                        const float* Vj = Vbh + (size_t)j * D;
                        for (int d = 0; d < D; ++d) outRow[d] += a * Vj[d];
                    }
                }
            }
        });

        // (B,H,S,D) -> (B,S,H*D)
        Mat& output = outputs[0];
        float* outPtr = output.ptr<float>();
        for (int b = 0; b < B; ++b) {
            for (int h = 0; h < num_heads; ++h) {
                const float* src = outHeadsMajor.data() + (((size_t)b * num_heads + h) * S) * D;
                for (int s = 0; s < S; ++s) {
                    float* dst = outPtr + (((size_t)b * S + s) * num_heads + h) * D;
                    std::memcpy(dst, src + (size_t)s * D, sizeof(float) * D);
                }
            }
        }
    }
};

Ptr<GroupQueryAttentionLayer> GroupQueryAttentionLayer::create(const LayerParams& params) {
    return makePtr<GroupQueryAttentionLayerImpl>(params);
}

}} // namespace cv::dnn
