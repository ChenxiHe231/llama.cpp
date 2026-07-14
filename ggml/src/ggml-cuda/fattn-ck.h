#pragma once

#ifdef GGML_HIP_CK_FATTN

#ifdef __cplusplus
extern "C" {
#endif

// Thin C entry into CK ck_tile FMHA fwd. Returns kernel time in ms (>=0) on
// success, or a negative value if CK's dispatcher found no matching instance.
float ggml_cuda_fa_ck_d256_fp16(
        const void * q_ptr, const void * k_ptr, const void * v_ptr, void * o_ptr,
        int batch, int nhead_q, int nhead_k, int seqlen_q, int seqlen_k, int hdim,
        float scale,
        long stride_q, long nhead_stride_q, long batch_stride_q,
        long stride_k, long nhead_stride_k, long batch_stride_k,
        long stride_v, long nhead_stride_v, long batch_stride_v,
        long stride_o, long nhead_stride_o, long batch_stride_o,
        int  mask_type,
        void * stream);

#ifdef __cplusplus
}
#endif

#endif // GGML_HIP_CK_FATTN
