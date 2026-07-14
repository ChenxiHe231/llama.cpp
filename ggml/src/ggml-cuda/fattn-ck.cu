// CK ck_tile FMHA forward integration for the RDNA3.5 D=256 GQA=8 path (spike).
// This TU is compiled against CK's own ck_tile headers and calls the pre-built
// dispatched host entry `fmha_fwd(...)`. The generated instance objects and the
// fmha_fwd_api object are linked in separately (see ggml-hip/CMakeLists.txt).
//
// Only built when GGML_HIP_CK_FATTN is defined.

#ifdef GGML_HIP_CK_FATTN

#include <hip/hip_runtime.h>
#include "fmha_fwd.hpp"   // from CK example include path

#include "fattn-ck.h"

extern "C" float ggml_cuda_fa_ck_d256_fp16(
        const void * q_ptr, const void * k_ptr, const void * v_ptr, void * o_ptr,
        int batch, int nhead_q, int nhead_k, int seqlen_q, int seqlen_k, int hdim,
        float scale,
        // strides in ELEMENTS ([b,h,s,d] layout: row stride, head stride, batch stride)
        long stride_q, long nhead_stride_q, long batch_stride_q,
        long stride_k, long nhead_stride_k, long batch_stride_k,
        long stride_v, long nhead_stride_v, long batch_stride_v,
        long stride_o, long nhead_stride_o, long batch_stride_o,
        int  mask_type, // 1 = top-left causal, 2 = bottom-right causal, 0 = none
        void * stream) {

    fmha_fwd_traits traits{};
    traits.hdim_q              = hdim;
    traits.hdim_v              = hdim;
    traits.data_type           = "fp16";
    traits.is_group_mode       = false;
    traits.is_v_rowmajor       = true;
    traits.has_logits_soft_cap = false;
    traits.mask_type           = static_cast<mask_enum>(mask_type);
    traits.bias_type           = bias_enum::no_bias;
    traits.has_lse             = false;
    traits.has_dropout         = false;
    traits.qscale_type         = quant_scale_enum::no_scale;
    traits.skip_min_seqlen_q   = false;
    traits.has_sink            = false;

    fmha_fwd_args a{};
    a.q_ptr = q_ptr;
    a.k_ptr = k_ptr;
    a.v_ptr = v_ptr;
    a.o_ptr = o_ptr;
    a.bias_ptr     = nullptr;
    a.lse_ptr      = nullptr;
    a.rand_val_ptr = nullptr;

    a.seqlen_q     = seqlen_q;
    a.seqlen_k     = seqlen_k;
    a.batch        = batch;
    a.max_seqlen_q = seqlen_q;
    a.hdim_q       = hdim;
    a.hdim_v       = hdim;
    a.nhead_q      = nhead_q;
    a.nhead_k      = nhead_k;
    a.num_head_q_total = nhead_q;
    a.head_start       = 0;

    a.scale_s         = scale;
    a.logits_soft_cap = 0.0f;

    a.stride_q = stride_q;  a.nhead_stride_q = nhead_stride_q;  a.batch_stride_q = batch_stride_q;
    a.stride_k = stride_k;  a.nhead_stride_k = nhead_stride_k;  a.batch_stride_k = batch_stride_k;
    a.stride_v = stride_v;  a.nhead_stride_v = nhead_stride_v;  a.batch_stride_v = batch_stride_v;
    a.stride_o = stride_o;  a.nhead_stride_o = nhead_stride_o;  a.batch_stride_o = batch_stride_o;

    // Causal encoded as an lr window: left = -1 (unbounded past), right = 0.
    a.window_size_left  = -1;
    a.window_size_right =  0;
    a.sink_size         =  0;
    a.mask_type         = mask_type;
    a.min_seqlen_q      = 0;

    a.p_drop    = 0.0f;
    a.s_randval = false;
    a.drop_seed_offset = std::make_pair(static_cast<uint64_t>(0), static_cast<uint64_t>(0));

    a.block_scale_size_q  = 0;
    a.block_scale_size_kv = 0;

    ck_tile::stream_config sc{};
    sc.stream_id_ = reinterpret_cast<hipStream_t>(stream);

    return fmha_fwd(traits, a, sc);
}

#endif // GGML_HIP_CK_FATTN
