// fattn-wmma-d256-rdna35.cuh —  WMMA flash-attention for gfx1151/RDNA3.5, D=256.
//
// Embedded from protos/wmma-fa-proto.cu (validated: Bc=32/NW=4 = 23.22 TFLOPS = 39.1% peak;
// causal probe PASS diff=1.7e-4). The mma primitives below are inline-copied verbatim from the
// validated prototype (private namespace to avoid clashing with mma.cuh) — do not drift.
//
// Adaptations vs prototype for real ggml tensors:
//  - Q is F32 with strides (read float2, make_half2; scale applied in softmax as in prototype).
//  - K/V are contiguous f16 from the fattn workspace (byte-stride indexed).
//  - Causal mask via the mask tensor: set unscaled KQ_C = -inf where mask is -inf (masks are 0/-inf
//    on this path: gqa_opt_applies requires mask present, max_bias==0). Validated equivalent to the
//    position-based causal probe (diff 1.7e-4).
//  - Per-Q-block causal KV-loop bound kvb_stop=(r0+BR_BLOCK-1)/Bc+1: skips upper-triangle chunks
//    (valid for causal/sliding-window; prefill is causal). ~halves work vs full-range.
//  - Multi-sequence via blockIdx.z; GQA via qhead=blockIdx.y*NWARPS+warp, kvh=qhead/gqa.
//  - Direct F32 dst write (no stream-K, no dst_meta). Boundary guard for partial last Q-block.

#pragma once
#include "fattn-common.cuh"

namespace wmma_d256 {

enum data_layout {
    DATA_LAYOUT_I_MAJOR          =  0,
    DATA_LAYOUT_I_MAJOR_MIRRORED = 20,
};

template <int I_, int J_, typename T, data_layout dl> struct tile {};

template <int I_, int J_>
struct tile<I_, J_, float, DATA_LAYOUT_I_MAJOR> {
    static constexpr int         I  = I_;
    static constexpr int         J  = J_;
    static constexpr data_layout dl_ = DATA_LAYOUT_I_MAJOR;
    static constexpr int         ne = I * J / 32;
    float x[ne] = {0};
    static __device__ __forceinline__ int get_i(const int l) { (void)l; return threadIdx.x % 16; }
    static __device__ __forceinline__ int get_j(const int l) { return 2 * l + (threadIdx.x / 16); }
};

template <int I_, int J_>
struct tile<I_, J_, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> {
    static constexpr int         I  = I_;
    static constexpr int         J  = J_;
    static constexpr data_layout dl_ = DATA_LAYOUT_I_MAJOR_MIRRORED;
    static constexpr int         ne = I * J / 32 * 2;
    half2 x[ne] = {{0.0f, 0.0f}};
    static __device__ __forceinline__ int get_i(const int /*l*/) { return threadIdx.x % 16; }
    static __device__ __forceinline__ int get_j(const int l) { return l; }
};

template <data_layout dl_ab, data_layout dl_d>
static __device__ __forceinline__ void mma(
        tile<16, 16, float, dl_d> & D,
        const tile<16, 8, half2, dl_ab> & A,
        const tile<16, 8, half2, dl_ab> & B) {
    using halfx16_t = __attribute__((ext_vector_type(16))) _Float16;
    using floatx8_t  = __attribute__((ext_vector_type(8)))  float;
    floatx8_t      & acc_frag = reinterpret_cast<floatx8_t      &>(D.x[0]);
    const halfx16_t& a_frag   = reinterpret_cast<const halfx16_t&>(A.x[0]);
    const halfx16_t& b_frag   = reinterpret_cast<const halfx16_t&>(B.x[0]);
    acc_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, acc_frag);
}

static __device__ __forceinline__ tile<16, 8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>
get_half2(const tile<16, 16, float, DATA_LAYOUT_I_MAJOR> & tile_float) {
    tile<16, 8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> ret;
    #pragma unroll
    for (int l = 0; l < tile_float.ne; ++l) {
        float tmp[2];
        int i = threadIdx.x / 16;
        tmp[i] = tile_float.x[l];
        i ^= 1;
        // permlanex16 swaps lane <-> lane^16 in-register (v_permlanex16_b32), avoiding
        // the ds_bpermute LDS round-trip that __shfl_xor_sync(...,16,32) lowers to on gfx1151.
        tmp[i] = __builtin_bit_cast(float,
            __builtin_amdgcn_permlanex16(0, __builtin_bit_cast(int, tile_float.x[l]),
                                         0x76543210, 0xfedcba98, false, true));
        ret.x[l] = make_half2(tmp[0], tmp[1]);
    }
    return ret;
}

static __device__ __forceinline__ void
load_ldmatrix(tile<16, 8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> & t,
              const half2 * __restrict__ xs0, const int stride) {
    const int row = threadIdx.x % 16;
    #pragma unroll
    for (int l = 0; l < t.ne; ++l) t.x[l] = xs0[row * stride + l];
}

static __device__ __forceinline__ void
load_ldmatrix_trans(tile<16, 8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> & t,
                    const half2 * __restrict__ xs0, const int stride) {
    const half * xh_src = (const half *) xs0;
    half * xh = (half *) t.x;
    const int stride_h = 2 * stride;
    const int d = threadIdx.x % 16;
    #pragma unroll
    for (int l = 0; l < t.ne; ++l) {
        xh[2 * l + 0] = xh_src[(2 * l + 0) * stride_h + d];
        xh[2 * l + 1] = xh_src[(2 * l + 1) * stride_h + d];
    }
}

} // namespace wmma_d256

template<int D, int Bc, int BR_BLOCK, int NWARPS>
__global__ __launch_bounds__(NWARPS * 32, 1)
void flash_attn_wmma_d256_rdna35(
        const char * __restrict__ Q,      // F32 [ne03, ne02, ne01, D]
        const half * __restrict__ K,      // F16 contiguous [ne03, ne12, ne11, D]
        const half * __restrict__ V,      // F16 contiguous [ne03, ne12, ne11, D]
        const half * __restrict__ mask,   // F16 [ne33, ne01, ne11] (0 / -inf) or nullptr
        const int  * __restrict__ KV_max, // [ne03, ntiles_x] upper KV bound per Q-tile (or nullptr)
        float       * __restrict__ dst,   // F32 [ne03, ne02, ne01, D] contiguous
        const float scale,
        const int32_t ne01, const int32_t ne02, const int32_t ne11, const int32_t ne12,
        const int32_t ne03, const int32_t ne33,
        const int32_t nb01, const int32_t nb02, const int64_t nb03,
        const int32_t nb11, const int32_t nb12, const int64_t nb13,
        const int32_t nb21, const int32_t nb22, const int64_t nb23,
        const int32_t nb31, const int64_t nb33) {
    static_assert(D == 256, "D256 only");
    constexpr int DTILES = D / 16;
    constexpr int NKSUB  = Bc / 16;
    constexpr int NSLOT  = Bc <= 16 ? 2 : 1;
    constexpr int NTHREADS = NWARPS * 32;
    constexpr int HEADS_PER_BLOCK = NWARPS;
    using T_A_KQ  = wmma_d256::tile<16, 8, half2, wmma_d256::DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using T_B_KQ  = wmma_d256::tile<16, 8, half2, wmma_d256::DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using T_C_KQ  = wmma_d256::tile<16, 16, float, wmma_d256::DATA_LAYOUT_I_MAJOR>;
    using T_A_VKQ = wmma_d256::tile<16, 8, half2, wmma_d256::DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using T_B_VKQ = wmma_d256::tile<16, 8, half2, wmma_d256::DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using T_C_VKQ = wmma_d256::tile<16, 16, float, wmma_d256::DATA_LAYOUT_I_MAJOR>;

    const int tid   = threadIdx.y * 32 + threadIdx.x;
    const int warp  = threadIdx.y;
    const int lane  = threadIdx.x;
    const int qrow  = lane % 16;
    const int halfb = lane / 16;
    const int r0    = blockIdx.x * BR_BLOCK;
    const int qhead = blockIdx.y * HEADS_PER_BLOCK + warp;
    const int seq   = blockIdx.z;
    if (qhead >= ne02) return;
    const int gqa   = ne02 / ne12;
    const int kvh   = qhead / gqa;
    const int nchunks = (ne11 + Bc - 1) / Bc;  // #7: ceil so the partial tail KV chunk is processed
    constexpr int RS = (Bc >= 64) ? D : (D + 8);
    __shared__ half sK[NSLOT][Bc * RS];
    __shared__ half sV[NSLOT][Bc * RS];

    // Per-warp LDS scratch to broadcast the Q pack from lanes [0,16) to lanes [16,32).
    // Lane l and lane l+16 have the same qrow (=l%16) and thus the same Qrow address, so only
    // the low half-warp fetches from global and both halves consume from LDS. This shrinks the
    // Q-pack live-range from 32 lanes worth of loads/half2 packs down to 16, freeing VGPR that
    // the register allocator was spilling on the baseline (VGPR 256, spill 741, private 1748 B).
    // Layout [warp][l][qrow]: 32-bit half2 stride; qrow varies with lane so lanes 0..15 write to
    // distinct banks (no conflict) and lanes 16..31 read the same address as their qrow-peer
    // (LDS broadcast, no conflict).
    constexpr int Q_PACK_HALF2 = 8;                // == T_B_KQ::ne for D=256, Bc=32
    static_assert(T_B_KQ::ne == Q_PACK_HALF2, "Q_PACK_HALF2 must match T_B_KQ::ne");
    __shared__ half2 sQ_bcast[NWARPS][Q_PACK_HALF2][16]; // 4 warps * 8 * 16 * 4B = 2 KB

    const bool qrow_valid = (r0 + qrow) < ne01;

    float KQ_max    = -INFINITY;
    float KQ_rowsum = 0.0f;
    T_C_VKQ VKQ_C[DTILES];
    #pragma unroll
    for (int dt = 0; dt < DTILES; ++dt)
        #pragma unroll
        for (int l = 0; l < T_C_VKQ::ne; ++l) VKQ_C[dt].x[l] = 0.0f;

    // Q (F32) base for this seq+head
    const float2 * Q_f2  = (const float2 *)(Q + nb03*seq + nb02*qhead);
    const int stride_Q1  = nb01 / (int)sizeof(float2);   // float2 units per Q row

    // K/V (contiguous f16) base for this seq+kvhead
    const half * Kseq = (const half *)((const char *)K + nb13*seq + nb12*kvh);
    const half * Vseq = (const half *)((const char *)V + nb23*seq + nb22*kvh);

    // mask base for this seq
    const half * mask_seq  = mask ? (const half *)((const char *)mask + nb33*(seq % ne33)) : nullptr;
    const int    stride_m = mask ? (nb31 / (int)sizeof(half)) : 0;
    const int    stride_Kh = nb11 / (int)sizeof(half);   // K row stride (half units)
    const int    stride_Vh = nb21 / (int)sizeof(half);   // V row stride (half units)

    // Cooperative global->LDS load of one KV chunk into `slot`. Zero-fills rows beyond ne11 so
    // the partial tail chunk is safe. Factored into a lambda so the double-buffer pipeline can
    // EARLY-ISSUE the next chunk's global loads before the current chunk's WMMA/softmax, letting
    // the DRAM latency overlap with compute (CK qr_ks_vs software-pipeline; no async DMA needed
    // on gfx11 -- ordinary buffer_load retires in the background of the WMMA that reads the
    // OTHER slot).
    auto load_chunk = [&] __device__ (int kvb_ld, int slot) {
        const half * Ksrc = Kseq + (size_t)kvb_ld * Bc * stride_Kh;
        const half * Vsrc = Vseq + (size_t)kvb_ld * Bc * stride_Vh;
        half * dk = &sK[slot][0];
        half * dv = &sV[slot][0];
        const int base_row = kvb_ld * Bc;
        for (int i = tid; i < Bc * D; i += NTHREADS) {
            const int r = i / D, c = i % D;
            if (base_row + r < ne11) {
                dk[r*RS + c] = Ksrc[r*stride_Kh + c]; dv[r*RS + c] = Vsrc[r*stride_Vh + c];
            } else {
                dk[r*RS + c] = (half)0.0f; dv[r*RS + c] = (half)0.0f;
            }
        }
    };

    // Prologue: load chunk 0 into slot 0.
    load_chunk(0, 0);
    __syncthreads();

    // KV-loop bound: use the per-tile KV_max from flash_attn_mask_to_KV_max when available
    // (correctly accounts for the ubatch Q-offset via the mask scan). Falls back to full range;
    // the mask-add below handles per-element causal masking either way.
    int kvb_stop = nchunks;
    if (KV_max) {
        const int ntiles_x = (ne01 + BR_BLOCK - 1) / BR_BLOCK;
        const int km = KV_max[seq * ntiles_x + blockIdx.x];  // multiple of 256, upper edge of valid region
        kvb_stop = (km + Bc - 1) / Bc;                        // 256 % 32 == 0 => exact
        if (kvb_stop > nchunks) kvb_stop = nchunks;
        if (kvb_stop < 0) kvb_stop = 0;
    }

    for (int kvb = 0; kvb < kvb_stop; ++kvb) {
        const int buf = kvb % NSLOT;
        const half * sKb = &sK[buf][0];
        const half * sVb = &sV[buf][0];
        // EARLY ISSUE: with a 2-slot buffer, kick off the next chunk's global->LDS load into the
        // OTHER slot BEFORE this chunk's WMMA/softmax. The buffer_load VMEM ops retire in the
        // background while the compute below reads the disjoint slot `buf`, giving load/compute
        // overlap (the single barrier at the loop end gates both the WAR on `buf` and the RAW on
        // `buf^1`). No-op when NSLOT==1 (falls back to the serial load-after-compute path).
        // Note: deliberately NOT bracketed by sched_barrier -- the overlap requires the compiler
        // to be free to interleave the next-chunk VMEM issue with this chunk's WMMA; a hard
        // sched_barrier(0) would isolate the load into its own region and serialize it.
        if (NSLOT > 1 && kvb + 1 < kvb_stop) {
            load_chunk(kvb + 1, buf ^ 1);
        }
        T_C_KQ KQ_C[NKSUB];
        #pragma unroll
        for (int kb = 0; kb < NKSUB; ++kb)
            #pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) KQ_C[kb].x[l] = 0.0f;
        // KQ mma: K-as-A, Q-as-B
        #pragma unroll
        for (int kb = 0; kb < NKSUB; ++kb) {
            const int kr = kb * 16;
            #pragma unroll
            for (int dt = 0; dt < DTILES; ++dt) {
                const int d0 = dt * 16;
                T_A_KQ K_A;
                wmma_d256::load_ldmatrix(K_A, (const half2*)(sKb + kr*RS + d0), RS / 2);
                T_B_KQ Q_B;
                {
                    // Only the low half-warp (halfb == 0) issues the global Q load and packs
                    // half2; both halves then read from LDS. This preserves the qrow_valid
                    // zero-fill fallback for the tail-partial Q block.
                    if (halfb == 0) {
                        if (qrow_valid) {
                            const float2 * Qrow_f2 = Q_f2 + (r0 + qrow) * stride_Q1 + d0 / 2;
                            #pragma unroll
                            for (int l = 0; l < Q_B.ne; ++l) {
                                const float2 tmp = Qrow_f2[l];
                                sQ_bcast[warp][l][qrow] = make_half2(tmp.x, tmp.y);
                            }
                        } else {
                            #pragma unroll
                            for (int l = 0; l < Q_B.ne; ++l) {
                                sQ_bcast[warp][l][qrow] = make_half2(0.0f, 0.0f);
                            }
                        }
                    }
                    // Wave-scope LDS visibility: HIP disables __syncwarp() (see vendors/hip.h),
                    // so call the underlying builtins directly. The release+wave_barrier+acquire
                    // triple emits s_waitcnt lgkmcnt(0) around a hardware wave barrier, which
                    // is the minimum needed to make lane 0..15's stores visible to lane 16..31's
                    // loads within the same wave (32 lanes, wave32). No __syncthreads() -- this
                    // is per-warp broadcast, cross-warp sync would cost 4-way barrier overhead.
                    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
                    __builtin_amdgcn_wave_barrier();
                    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
                    #pragma unroll
                    for (int l = 0; l < Q_B.ne; ++l) {
                        Q_B.x[l] = sQ_bcast[warp][l][qrow];
                    }
                }
                wmma_d256::mma<wmma_d256::DATA_LAYOUT_I_MAJOR_MIRRORED, wmma_d256::DATA_LAYOUT_I_MAJOR>(KQ_C[kb], K_A, Q_B);
            }
        }
        // Causal mask: set unscaled KQ_C = -inf where mask is -inf (masks are 0/-inf on this path).
        // Mapping (locked by causal probe, diff 1.7e-4): qrow=lane%16, kvcol=vkb*Bc+kb*16+2l+lane/16.
        {
            const int qg = r0 + qrow;
            #pragma unroll
            for (int kb = 0; kb < NKSUB; ++kb) {
                #pragma unroll
                for (int l = 0; l < T_C_KQ::ne; ++l) {
                    const int kg = kvb*Bc + kb*16 + 2*l + halfb;
                    if (kg >= ne11) { KQ_C[kb].x[l] = -INFINITY; continue; }
                    if (!qrow_valid) { KQ_C[kb].x[l] = -INFINITY; continue; }  // #5: qrow OOB must not read mask
                    if (mask_seq) {
                        const float mv = __half2float(mask_seq[(size_t)qg * stride_m + kg]);
                        // DIAG p19: add finite mask bias (reference adds mask to the SCALED score).
                        // KQ_C holds the unscaled QK; scale is applied downstream as KQ_C*scale, so
                        // fold the bias as mv/scale here => (KQ_C + mv/scale)*scale = KQ_C*scale + mv.
                        if (mv == -INFINITY) KQ_C[kb].x[l] = -INFINITY;
                        else                 KQ_C[kb].x[l] += mv / scale;
                    } else if (kg > qg) {
                        KQ_C[kb].x[l] = -INFINITY;
                    }
                }
            }
        }
        // Softmax (online, scale applied here as in prototype). FAST_EXP2: track the running max
        // and take exp in the log2 domain (scale_log2e = scale*log2(e)); exp2f -> single v_exp_f32.
        const float scale_log2e = scale * 1.4426950408889634f;
        float m_chunk = -INFINITY;
        #pragma unroll
        for (int kb = 0; kb < NKSUB; ++kb)
            #pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l)
                m_chunk = fmaxf(m_chunk, KQ_C[kb].x[l] * scale_log2e);
        m_chunk = fmaxf(m_chunk, __shfl_xor_sync(0xFFFFFFFFFFFFFFFFull, m_chunk, 16, 32));
        // #6: NaN guard. If this chunk is fully masked for the row (m_chunk == -inf) AND this is the
        // first valid chunk (KQ_max == -inf), then m_new = -inf and expf(-inf - -inf) = NaN would
        // poison KQ_rowsum/VKQ_C. Guard: chunk_active flags a fully-masked chunk; rescale folds to 0
        // when KQ_max is still -inf; exp is only taken when m_new is finite, else 0. The PV mma is
        // skipped for inactive chunks (P would be 0 anyway). chunk_active is uniform within a warp
        // (m_chunk is reduced by shfl_xor across the 16 lanes sharing a qrow), so no divergence; all
        // warps still reach the block-level __syncthreads and next-KV load below.
        const bool  chunk_active = (m_chunk != -INFINITY);
        const float m_new    = chunk_active ? fmaxf(KQ_max, m_chunk) : KQ_max;
        const float rescale  = (KQ_max == -INFINITY) ? 0.0f : exp2f(KQ_max - m_new);
        float rowsum_add = 0.0f;
        #pragma unroll
        for (int kb = 0; kb < NKSUB; ++kb)
            #pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                KQ_C[kb].x[l] = (m_new != -INFINITY) ? exp2f(KQ_C[kb].x[l] * scale_log2e - m_new) : 0.0f;
                rowsum_add += KQ_C[kb].x[l];
            }
        rowsum_add += __shfl_xor_sync(0xFFFFFFFFFFFFFFFFull, rowsum_add, 16, 32);
        KQ_rowsum = KQ_rowsum * rescale + rowsum_add;
        KQ_max = m_new;
        // Conditional rescale: skip the DTILES*ne no-op multiplies when the running max is unchanged
        // (rescale==1.0 in the exp2 domain). Bit-identical (x*1==x).
        if (rescale != 1.0f) {
            #pragma unroll
            for (int dt = 0; dt < DTILES; ++dt)
                #pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l)
                    VKQ_C[dt].x[l] *= rescale;
        }
        // VKQ mma: V-transposed-as-A, P-as-B. DIAG p19: run UNCONDITIONALLY.
        // chunk_active is NOT warp-uniform (only reduced across the 2 lanes of a qrow-pair via
        // shfl_xor 16, not across the 16 qrows), so `if (chunk_active)` diverges on diagonal
        // causal chunks -> cooperative get_half2 shfl + wmma run under partial EXEC -> NaN.
        // Inactive qrows have KQ_C==0 (P==0) so contribute nothing; unconditional is correct.
        {
            #pragma unroll
            for (int kb = 0; kb < NKSUB; ++kb) {
                const int kr = kb * 16;
                T_B_VKQ P_B = wmma_d256::get_half2(KQ_C[kb]);
                #pragma unroll
                for (int dt = 0; dt < DTILES; ++dt) {
                    const int d0 = dt * 16;
                    T_A_VKQ V_A;
                    wmma_d256::load_ldmatrix_trans(V_A, (const half2*)(sVb + kr*RS + d0), RS / 2);
                    wmma_d256::mma<wmma_d256::DATA_LAYOUT_I_MAJOR_MIRRORED, wmma_d256::DATA_LAYOUT_I_MAJOR>(VKQ_C[dt], V_A, P_B);
                }
            }
        }
        // Loop-end barrier. In the double-buffered path (NSLOT>1) the next chunk's load was already
        // early-issued into `buf^1` above; this single barrier gates both hazards (WAR on `buf` read
        // by compute, RAW on `buf^1` written by the early load) -- one sync per iteration instead of
        // the two-sync serial load-after-compute. In the NSLOT==1 fallback the load must happen here.
        __syncthreads();
        if (NSLOT == 1 && kvb + 1 < kvb_stop) {
            load_chunk(kvb + 1, 0);
            __syncthreads();
        }
    }
    if (!qrow_valid) return;
    const float inv = (KQ_rowsum > 0.0f) ? (1.0f / KQ_rowsum) : 0.0f;
    // DIAG p19: ggml dst layout is [D, ne02(heads), ne01(rows), seq]; offset =
    //   d + qhead*D + row*ne02*D + seq*ne01*ne02*D  (matches fattn-tile.cuh:1098).
    float * Orow = (float *)dst + ((size_t)(seq * ne01 + (r0 + qrow)) * ne02 + qhead) * D;
    #pragma unroll
    for (int dt = 0; dt < DTILES; ++dt) {
        #pragma unroll
        for (int l = 0; l < T_C_VKQ::ne; ++l) {
            const int d = dt * 16 + 2 * l + halfb;
            Orow[d] = VKQ_C[dt].x[l] * inv;
        }
    }
}
