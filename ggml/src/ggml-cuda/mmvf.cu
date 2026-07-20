#include "ggml.h"
#include "common.cuh"
#include "unary.cuh"
#include "mmvf.cuh"
#include "convert.cuh"

// hcx rows-per-block A/B: each block computes ROWS_PER_BLOCK output rows so that
// small-N decode (shexp/attn/MoE per-expert) launches enough blocks to fill all CUs,
// and so the shared activation vector y is loaded once and reused across R rows while
// R independent weight load streams stay in flight (raises per-wave MLP). R only takes
// effect for ncols_dst==1 && !is_multi_token_id; every other path falls back to R=1.
// *** Change R here (main agent will A/B R=2 vs R=4). ***
#ifndef GGML_MMVF_ROWS_PER_BLOCK
#define GGML_MMVF_ROWS_PER_BLOCK 2
#endif

template <typename T, typename type_acc, int ncols_dst, int block_size, int rows_per_block, bool has_fusion = false, bool is_multi_token_id = false>
static __global__ void mul_mat_vec_f(
        const T * x_ptr, const float * y_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const int ncols2, const int nrows, const uint3 nchannels_y, const int stride_row, const int stride_col_y2, const int stride_col_dst,
        const uint3 channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride) {
    const T       * GGML_CUDA_RESTRICT x   = x_ptr;
    const float   * GGML_CUDA_RESTRICT y   = y_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;
    // rows-per-block: this block owns output rows [row_base, row_base + rows_per_block)
    const int row_base    = blockIdx.x * rows_per_block;
    // for MUL_MAT_ID - blockIdx.y = n_expert_used, blockIdx.z = ncols_dst (tokens)
    const int channel_dst = blockIdx.y;
    const int tid         = threadIdx.x;

    int token_idx;
    int channel_x;
    int channel_y;
    int sample_dst;

    ggml_cuda_pdl_sync();
    if constexpr (is_multi_token_id) {
        // Multi-token MUL_MAT_ID path, adding these in the normal path causes a perf regression for n_tokens=1 case
        token_idx  = blockIdx.z;
        channel_x  = ids[channel_dst + token_idx * ids_stride];
        channel_y  = fastmodulo(channel_dst, nchannels_y);
        sample_dst = 0;
    } else {
        token_idx  = ids ? blockIdx.z                                          : 0;
        channel_x  = ids ? ids[blockIdx.y + token_idx * ids_stride]            : fastdiv((uint32_t) channel_dst, channel_ratio);
        channel_y  = ids ? fastmodulo(blockIdx.y, nchannels_y)                 : channel_dst;
        sample_dst = ids ? 0                                                   : blockIdx.z;
    }

    const int sample_x    = fastdiv((uint32_t) sample_dst, sample_ratio);
    const int sample_y    = sample_dst;

    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();

    // Base pointers land on row_base; per-row weight pointers add drow[r]*stride_row below.
    x   += int64_t(sample_x)  *stride_sample_x   + channel_x  *stride_channel_x   + row_base*stride_row;
    y   += int64_t(sample_y)  *stride_sample_y   + channel_y  *stride_channel_y;
    dst += int64_t(sample_dst)*stride_sample_dst + channel_dst*stride_channel_dst;
    if constexpr (is_multi_token_id) {
        y   += token_idx*stride_col_y2*2;
        dst += token_idx*stride_col_dst;
    }

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    ggml_glu_op glu_op = ggml_glu_op::GGML_GLU_OP_SWIGLU;
    const T * gate_x = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;

    if constexpr (has_fusion) {
        use_gate = fusion.gate != nullptr;
        use_bias = fusion.x_bias != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr;
        glu_op = fusion.glu_op;

        if (use_gate) {
            gate_x = static_cast<const T *>(fusion.gate);
        }
        if (use_bias) {
            x_bias = static_cast<const float *>(fusion.x_bias);
        }
        if (use_gate_bias) {
            gate_bias = static_cast<const float *>(fusion.gate_bias);
            use_gate_bias = use_gate;
        } else {
            use_gate_bias = false;
        }
    }

    if (use_gate) {
        gate_x += int64_t(sample_x)  *stride_sample_x   + channel_x  *stride_channel_x   + row_base*stride_row;
    }

    if constexpr (has_fusion) {
        const int channel_bias = ids ? channel_x : channel_dst;
        if (use_bias) {
            x_bias += int64_t(sample_dst)*stride_sample_dst + channel_bias*stride_channel_dst;
        }
        if (use_gate_bias) {
            gate_bias += int64_t(sample_dst)*stride_sample_dst + channel_bias*stride_channel_dst;
        }
    }

    const float2 * y2 = (const float2 *) y;

    extern __shared__ char data_mmv[];
    float * buf_iw = (float *) data_mmv;
    [[maybe_unused]] float * buf_iw_gate = nullptr;
    if constexpr (has_fusion) {
        buf_iw_gate = (float *) (data_mmv + warp_size*sizeof(float));
    }

    if (block_size > warp_size) {
        if (tid < warp_size) {
            buf_iw[tid] = 0.0f;
            if constexpr (has_fusion) {
                if (use_gate) {
                    buf_iw_gate[tid] = 0.0f;
                }
            }
        }
        __syncthreads();
    }

    // Per-row weight offset. Rows [row_base, row_base+R) may exceed nrows on the last
    // block when nrows % R != 0; clamp the READ of those rows to the last valid row
    // (row_base is always < nrows, so nrows-1-row_base is in [0, R-1]) so no OOB weight
    // load happens. The clamped rows compute a harmless duplicate sum that the write
    // guard (row < nrows) discards. This keeps the K-loop branch-free.
    int drow[rows_per_block];
#pragma unroll
    for (int r = 0; r < rows_per_block; ++r) {
        drow[r] = (row_base + r < nrows) ? r : (nrows - 1 - row_base);
    }

    float sumf[rows_per_block][ncols_dst];
#pragma unroll
    for (int r = 0; r < rows_per_block; ++r) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            sumf[r][j] = 0.0f;
        }
    }
    float sumf_gate[rows_per_block][ncols_dst];
    if constexpr (has_fusion) {
#pragma unroll
        for (int r = 0; r < rows_per_block; ++r) {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                sumf_gate[r][j] = 0.0f;
            }
        }
    }

    if constexpr (std::is_same_v<T, float>) {
        const float2 * x2[rows_per_block];
        [[maybe_unused]] const float2 * gate_x2[rows_per_block];
#pragma unroll
        for (int r = 0; r < rows_per_block; ++r) {
            x2[r] = (const float2 *) (x + drow[r]*stride_row);
            if constexpr (has_fusion) {
                gate_x2[r] = use_gate ? (const float2 *) (gate_x + drow[r]*stride_row) : nullptr;
            }
        }

        for (int col2 = tid; col2 < ncols2; col2 += block_size) {
            // load R independent weight streams first
            float2 tmpx[rows_per_block];
            float2 tmpx_gate[rows_per_block];
#pragma unroll
            for (int r = 0; r < rows_per_block; ++r) {
                tmpx[r] = x2[r][col2];
                tmpx_gate[r] = make_float2(0.0f, 0.0f);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmpx_gate[r] = gate_x2[r][col2];
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const float2 tmpy = y2[j*stride_col_y2 + col2];   // shared across R rows
#pragma unroll
                for (int r = 0; r < rows_per_block; ++r) {
                    ggml_cuda_mad(sumf[r][j], tmpx[r].x, tmpy.x);
                    ggml_cuda_mad(sumf[r][j], tmpx[r].y, tmpy.y);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            ggml_cuda_mad(sumf_gate[r][j], tmpx_gate[r].x, tmpy.x);
                            ggml_cuda_mad(sumf_gate[r][j], tmpx_gate[r].y, tmpy.y);
                        }
                    }
                }
            }
        }
    } else if constexpr (std::is_same_v<T, half>) {
        const half2 * x2[rows_per_block];
        [[maybe_unused]] const half2 * gate_x2[rows_per_block];
#pragma unroll
        for (int r = 0; r < rows_per_block; ++r) {
            x2[r] = (const half2 *) (x + drow[r]*stride_row);
            if constexpr (has_fusion) {
                gate_x2[r] = use_gate ? (const half2 *) (gate_x + drow[r]*stride_row) : nullptr;
            }
        }

        if (std::is_same_v<type_acc, float>) {
            for (int col2 = tid; col2 < ncols2; col2 += block_size) {
                float2 tmpx[rows_per_block];
                float2 tmpx_gate[rows_per_block];
#pragma unroll
                for (int r = 0; r < rows_per_block; ++r) {
                    tmpx[r] = __half22float2(x2[r][col2]);
                    tmpx_gate[r] = make_float2(0.0f, 0.0f);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            tmpx_gate[r] = __half22float2(gate_x2[r][col2]);
                        }
                    }
                }
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const float2 tmpy = y2[j*stride_col_y2 + col2];
#pragma unroll
                    for (int r = 0; r < rows_per_block; ++r) {
                        ggml_cuda_mad(sumf[r][j], tmpx[r].x, tmpy.x);
                        ggml_cuda_mad(sumf[r][j], tmpx[r].y, tmpy.y);
                        if constexpr (has_fusion) {
                            if (use_gate) {
                                ggml_cuda_mad(sumf_gate[r][j], tmpx_gate[r].x, tmpy.x);
                                ggml_cuda_mad(sumf_gate[r][j], tmpx_gate[r].y, tmpy.y);
                            }
                        }
                    }
                }
            }
        } else {
#ifdef FP16_AVAILABLE
            half2 sumh2[rows_per_block][ncols_dst];
            half2 sumh2_gate[rows_per_block][ncols_dst];
#pragma unroll
            for (int r = 0; r < rows_per_block; ++r) {
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    sumh2[r][j] = make_half2(0.0f, 0.0f);
                    sumh2_gate[r][j] = make_half2(0.0f, 0.0f);
                }
            }

            for (int col2 = tid; col2 < ncols2; col2 += block_size) {
                half2 tmpx[rows_per_block];
                half2 tmpx_gate[rows_per_block];
#pragma unroll
                for (int r = 0; r < rows_per_block; ++r) {
                    tmpx[r] = x2[r][col2];
                    tmpx_gate[r] = make_half2(0.0f, 0.0f);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            tmpx_gate[r] = gate_x2[r][col2];
                        }
                    }
                }
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const float2 tmpy = y2[j*stride_col_y2 + col2];
                    const half2 tmpy_h = make_half2(tmpy.x, tmpy.y);
#pragma unroll
                    for (int r = 0; r < rows_per_block; ++r) {
                        sumh2[r][j] += tmpx[r] * tmpy_h;
                        if constexpr (has_fusion) {
                            if (use_gate) {
                                sumh2_gate[r][j] += tmpx_gate[r] * tmpy_h;
                            }
                        }
                    }
                }
            }

#pragma unroll
            for (int r = 0; r < rows_per_block; ++r) {
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    sumf[r][j] = __low2float(sumh2[r][j]) + __high2float(sumh2[r][j]);
                }
            }

            if constexpr (has_fusion) {
                if (use_gate) {
#pragma unroll
                    for (int r = 0; r < rows_per_block; ++r) {
#pragma unroll
                        for (int j = 0; j < ncols_dst; ++j) {
                            sumf_gate[r][j] = __low2float(sumh2_gate[r][j]) + __high2float(sumh2_gate[r][j]);
                        }
                    }
                }
            }
#else
            NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
        }
    } else if constexpr (std::is_same_v<T, nv_bfloat16>) {
//TODO: add support for ggml_cuda_mad for hip_bfloat162
#if defined(GGML_USE_HIP)
        // hcx wide-load + rows-per-block: int4 (=4 int =8 bf16) per thread per row to raise
        // per-wave in-flight weight bytes (16B coalesced); y (activation) loaded once, reused
        // across R rows. ncols2 semantics kept; ncols4 derived internally.
        const int ncols4 = ncols2 / 4;
        const int4 * x4[rows_per_block];
        const int4 * gate_x4[rows_per_block];
        const int  * x2_tail[rows_per_block];
        const int  * gate_x2_tail[rows_per_block];
#pragma unroll
        for (int r = 0; r < rows_per_block; ++r) {
            const int * xr = (const int *) (x + drow[r]*stride_row);
            x4[r]      = (const int4 *) xr;
            x2_tail[r] = xr;
            gate_x4[r]      = nullptr;
            gate_x2_tail[r] = nullptr;
            if constexpr (has_fusion) {
                if (use_gate) {
                    const int * gr = (const int *) (gate_x + drow[r]*stride_row);
                    gate_x4[r]      = (const int4 *) gr;
                    gate_x2_tail[r] = gr;
                }
            }
        }

        for (int c4 = tid; c4 < ncols4; c4 += block_size) {
            int4 tmpx[rows_per_block];
            int4 tmpx_gate[rows_per_block];
#pragma unroll
            for (int r = 0; r < rows_per_block; ++r) {
                tmpx[r] = x4[r][c4];
                tmpx_gate[r] = make_int4(0, 0, 0, 0);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmpx_gate[r] = gate_x4[r][c4];
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const float2 tmpy = y2[j*stride_col_y2 + c4*4 + p];   // shared across R rows
#pragma unroll
                    for (int r = 0; r < rows_per_block; ++r) {
                        const int * tmpx_i = reinterpret_cast<const int *>(&tmpx[r]);
                        const float tmpx0 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx_i[p])[0]);
                        const float tmpx1 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx_i[p])[1]);
                        ggml_cuda_mad(sumf[r][j], tmpx0, tmpy.x);
                        ggml_cuda_mad(sumf[r][j], tmpx1, tmpy.y);

                        if constexpr (has_fusion) {
                            if (use_gate) {
                                const int * tmpx_gate_i = reinterpret_cast<const int *>(&tmpx_gate[r]);
                                const float g0 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx_gate_i[p])[0]);
                                const float g1 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx_gate_i[p])[1]);
                                ggml_cuda_mad(sumf_gate[r][j], g0, tmpy.x);
                                ggml_cuda_mad(sumf_gate[r][j], g1, tmpy.y);
                            }
                        }
                    }
                }
            }
        }
        // tail: int-pairs not covered by the int4 groups (ncols2 % 4 != 0)
        for (int col2 = ncols4*4 + tid; col2 < ncols2; col2 += block_size) {
            int tmpx[rows_per_block];
            int tmpx_gate[rows_per_block];
#pragma unroll
            for (int r = 0; r < rows_per_block; ++r) {
                tmpx[r] = x2_tail[r][col2];
                tmpx_gate[r] = 0;
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmpx_gate[r] = gate_x2_tail[r][col2];
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const float2 tmpy = y2[j*stride_col_y2 + col2];
#pragma unroll
                for (int r = 0; r < rows_per_block; ++r) {
                    const float tmpx0 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx[r])[0]);
                    const float tmpx1 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx[r])[1]);
                    ggml_cuda_mad(sumf[r][j], tmpx0, tmpy.x);
                    ggml_cuda_mad(sumf[r][j], tmpx1, tmpy.y);

                    if constexpr (has_fusion) {
                        if (use_gate) {
                            const float tmpx0_gate = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx_gate[r])[0]);
                            const float tmpx1_gate = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx_gate[r])[1]);
                            ggml_cuda_mad(sumf_gate[r][j], tmpx0_gate, tmpy.x);
                            ggml_cuda_mad(sumf_gate[r][j], tmpx1_gate, tmpy.y);
                        }
                    }
                }
            }
        }
#else
        const nv_bfloat162 * x2[rows_per_block];
        [[maybe_unused]] const nv_bfloat162 * gate_x2[rows_per_block];
#pragma unroll
        for (int r = 0; r < rows_per_block; ++r) {
            x2[r] = (const nv_bfloat162 *) (x + drow[r]*stride_row);
            if constexpr (has_fusion) {
                gate_x2[r] = use_gate ? (const nv_bfloat162 *) (gate_x + drow[r]*stride_row) : nullptr;
            }
        }
        for (int col2 = tid; col2 < ncols2; col2 += block_size) {
            nv_bfloat162 tmpx[rows_per_block];
            [[maybe_unused]] nv_bfloat162 tmpx_gate[rows_per_block];
#pragma unroll
            for (int r = 0; r < rows_per_block; ++r) {
                tmpx[r] = x2[r][col2];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmpx_gate[r] = gate_x2[r][col2];
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const float2 tmpy = y2[j*stride_col_y2 + col2];
#pragma unroll
                for (int r = 0; r < rows_per_block; ++r) {
                    ggml_cuda_mad(sumf[r][j], tmpx[r].x, tmpy.x);
                    ggml_cuda_mad(sumf[r][j], tmpx[r].y, tmpy.y);

                    if constexpr (has_fusion) {
                        if (use_gate) {
                            ggml_cuda_mad(sumf_gate[r][j], tmpx_gate[r].x, tmpy.x);
                            ggml_cuda_mad(sumf_gate[r][j], tmpx_gate[r].y, tmpy.y);
                        }
                    }
                }
            }
        }
#endif
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }

    ggml_cuda_pdl_lc();
    // Reduce each (row r, col j) independently. buf_iw / buf_iw_gate hold one float per
    // warp (block_size <= 256 => <= 8 warps) and are reused serially across all (r,j);
    // the __syncthreads() at the end of each iteration keeps the reuse race-free, so no
    // extra shared memory (no R-factor on nbytes_shared) is needed.
#pragma unroll
    for (int r = 0; r < rows_per_block; ++r) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            sumf[r][j] = warp_reduce_sum<warp_size>(sumf[r][j]);

            if constexpr (has_fusion) {
                if (use_gate) {
                    sumf_gate[r][j] = warp_reduce_sum<warp_size>(sumf_gate[r][j]);
                }
            }

            if (block_size > warp_size) {
                buf_iw[tid/warp_size] = sumf[r][j];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        buf_iw_gate[tid/warp_size] = sumf_gate[r][j];
                    }
                }
                __syncthreads();
                if (tid < warp_size) {
                    sumf[r][j] = buf_iw[tid];
                    sumf[r][j] = warp_reduce_sum<warp_size>(sumf[r][j]);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            sumf_gate[r][j] = buf_iw_gate[tid];
                            sumf_gate[r][j] = warp_reduce_sum<warp_size>(sumf_gate[r][j]);
                        }
                    }
                }

                // serial reuse of buf_iw across the next (r,j): every thread must finish
                // reading before the next iteration overwrites it.
                __syncthreads();
            }
        }
    }

    // Write-out: threads [0, ncols_dst) each own one dst column, for every row r.
    if (tid >= ncols_dst) {
        return;
    }

#pragma unroll
    for (int r = 0; r < rows_per_block; ++r) {
        const int row = row_base + r;
        if (row >= nrows) {
            continue;   // clamped duplicate row (nrows % R != 0); discard
        }

        float value = sumf[r][tid];

        if constexpr (has_fusion) {
            if (use_bias) {
                value += x_bias[tid*stride_col_dst + row];
            }

            if (use_gate) {
                float gate_value = sumf_gate[r][tid];
                if (use_gate_bias) {
                    gate_value += gate_bias[tid*stride_col_dst + row];
                }
                switch (glu_op) {
                    case GGML_GLU_OP_SWIGLU:
                        value *= ggml_cuda_op_silu_single(gate_value);
                        break;
                    case GGML_GLU_OP_GEGLU:
                        value *= ggml_cuda_op_gelu_single(gate_value);
                        break;
                    case GGML_GLU_OP_SWIGLU_OAI: {
                        value = ggml_cuda_op_swiglu_oai_single(gate_value, value);
                        break;
                    }
                    default:
                        break;
                }
            }
        }

        dst[tid*stride_col_dst + row] = value;
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, glu_op, gate_x, x_bias, gate_bias, sumf_gate);
    }
}


template<typename T, typename type_acc, int ncols_dst, int block_size, int rows_per_block, bool is_multi_token_id = false>
static void mul_mat_vec_f_switch_fusion(
        const T * x, const float * y, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int64_t ncols, const int64_t nrows, const uint3 nchannels_y,
        const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const uint3 channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const dim3 & block_dims, const dim3 & block_nums, const int nbytes_shared, const int ids_stride, const cudaStream_t stream) {

    const ggml_cuda_kernel_launch_params launch_params = {block_nums, block_dims, nbytes_shared, stream};

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
    if constexpr (ncols_dst == 1) {
        if (has_fusion) {
            ggml_cuda_kernel_launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, rows_per_block, true, is_multi_token_id>, launch_params,
                x, y, ids, fusion, dst, ncols, (int) nrows, nchannels_y, stride_row, stride_col_y, stride_col_dst,
                channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
            return;
       }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    ggml_cuda_kernel_launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, rows_per_block, false, is_multi_token_id>, launch_params,
        x, y, ids, fusion, dst, ncols, (int) nrows, nchannels_y, stride_row, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);

}

template <typename T, typename type_acc, int ncols_dst, bool is_multi_token_id = false>
void launch_mul_mat_vec_f_cuda(
        const T * x, const float * y, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int64_t ncols, const int64_t nrows,
        const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
        const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        const int64_t nsamples_or_ntokens, const int64_t ids_stride, cudaStream_t stream) {
    GGML_ASSERT(ncols        % 2 == 0);
    GGML_ASSERT(stride_row   % 2 == 0);
    GGML_ASSERT(stride_col_y % 2 == 0);
    GGML_ASSERT(ids || nchannels_dst % nchannels_x == 0);
    GGML_ASSERT(       nsamples_dst  % nsamples_x  == 0);
    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0) : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[device].warp_size;

    int64_t block_size_best = warp_size;
    int64_t niter_best      = (ncols + 2*warp_size - 1) / (2*warp_size);
    int64_t max_block_size  = 256;
    if(ggml_cuda_info().devices[device].cc > GGML_CUDA_CC_OFFSET_AMD && ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_RDNA1) {
        max_block_size = 128;
    }
    for (int64_t block_size = 2*warp_size; block_size <= max_block_size; block_size += warp_size) {
        const int64_t niter = (ncols + 2*block_size - 1) / (2*block_size);
        if (niter < niter_best) {
            niter_best      = niter;
            block_size_best = block_size;
        }
    }

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;

    const int nbytes_shared = warp_size*sizeof(float) + (has_fusion ? warp_size*sizeof(float) : 0);

    // hcx rows-per-block: R only for the single-token ncols_dst==1 path (decode / MoE
    // per-expert). Multi-token MUL_MAT_ID and ncols_dst>1 keep R=1 for correctness/no
    // regression. Grid x-dim shrinks to ceil(nrows / R); kernel guards row < nrows.
    constexpr int rows_per_block = (ncols_dst == 1 && !is_multi_token_id) ? GGML_MMVF_ROWS_PER_BLOCK : 1;
    const int64_t nrows_blocks = (nrows + rows_per_block - 1) / rows_per_block;

    const dim3 block_nums(nrows_blocks, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(block_size_best, 1, 1);
    switch (block_size_best) {
        case   32: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 32, rows_per_block, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nrows, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        case   64: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 64, rows_per_block, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nrows, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        case   96: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 96, rows_per_block, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nrows, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        case  128: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 128, rows_per_block, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nrows, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        case  160: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 160, rows_per_block, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nrows, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        case  192: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 192, rows_per_block, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nrows, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        case  224: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 224, rows_per_block, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nrows, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        case  256: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 256, rows_per_block, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nrows, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        default: {
            GGML_ABORT("fatal error");
        } break;
    }
}
template <typename T, typename type_acc>
static void mul_mat_vec_f_cuda_switch_ncols_dst(
        const T * x, const float * y, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int64_t ncols, const int64_t nrows, const int64_t ncols_dst,
        const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
        const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        const int64_t ids_stride, cudaStream_t stream) {

    const bool has_ids = ids != nullptr;

    if (has_ids && ncols_dst > 1) {
        // Multi-token MUL_MAT_ID path only - single-token goes through regular path below
        constexpr int c_ncols_dst = 1;
        launch_mul_mat_vec_f_cuda<T, type_acc, c_ncols_dst, true>
            (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
             nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
             stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
             ncols_dst, ids_stride, stream);
        return;
    }

    if (has_ids) {
        // Single-token MUL_MAT_ID path
        constexpr int c_ncols_dst = 1;
        launch_mul_mat_vec_f_cuda<T, type_acc, c_ncols_dst>
            (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
             nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
             stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
             ncols_dst, ids_stride, stream);
        return;
    }

    switch (ncols_dst) {
        case 1:
            launch_mul_mat_vec_f_cuda<T, type_acc, 1>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 2:
            launch_mul_mat_vec_f_cuda<T, type_acc, 2>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 3:
            launch_mul_mat_vec_f_cuda<T, type_acc, 3>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 4:
            launch_mul_mat_vec_f_cuda<T, type_acc, 4>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 5:
            launch_mul_mat_vec_f_cuda<T, type_acc, 5>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 6:
            launch_mul_mat_vec_f_cuda<T, type_acc, 6>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 7:
            launch_mul_mat_vec_f_cuda<T, type_acc, 7>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 8:
            launch_mul_mat_vec_f_cuda<T, type_acc, 8>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

template<typename T>
static void mul_mat_vec_f_cuda(
        const T * x, const float * y, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int64_t ncols, const int64_t nrows, const int64_t ncols_dst,
        const int64_t stride_row, const int64_t stride_col_y, const int stride_col_dst,
        const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
        const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        const int64_t ids_stride, enum ggml_prec prec, cudaStream_t stream) {

    if constexpr(std::is_same_v<T, half>) {
        if (prec == GGML_PREC_DEFAULT) {
            mul_mat_vec_f_cuda_switch_ncols_dst<T, half>
                (x, y, ids, fusion, dst, ncols, nrows, ncols_dst, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            return;
        }
    }
    mul_mat_vec_f_cuda_switch_ncols_dst<T, float>
        (x, y, ids, fusion, dst, ncols, nrows, ncols_dst, stride_row, stride_col_y, stride_col_dst,
        nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
        stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
}

void ggml_cuda_mul_mat_vec_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(!ids ||  ids->type == GGML_TYPE_I32);
    GGML_ASSERT(         dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(!ids || ne12 <= MMVF_MAX_BATCH_SIZE);
    GGML_ASSERT(ne13 == ne3);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));
    GGML_ASSERT(        nb0        == ts_dst);

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const enum ggml_prec prec = fast_fp16_available(cc) ? ggml_prec(dst->op_params[0]) : GGML_PREC_F32;

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_device fusion_local{};

    if (fusion) {
        GGML_ASSERT( !ids || dst->ne[2] == 1);
        GGML_ASSERT(  ids || dst->ne[1] == 1);
        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            GGML_ASSERT(fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0));
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            fusion_local.gate_bias = fusion->gate_bias->data;
        }
        fusion_local.glu_op = fusion->glu_op;
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = src1->nb[1] / ts_src1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s12 = src1->nb[2] / ts_src1;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s13 = src1->nb[3] / ts_src1;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    switch (src0->type) {
        case GGML_TYPE_F32: {
            const float * src0_d = (const float *) src0->data;
            mul_mat_vec_f_cuda(src0_d, src1_d, ids_d, fusion_local, dst_d, ne00, ne01, ncols_dst, s01, stride_col_y, stride_col_dst,
                ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
                ne03,              ne3,           s03, s13,              s3,                 ids_stride, prec, ctx.stream());
        } break;
        case GGML_TYPE_F16: {
            const half * src0_d = (const half *) src0->data;
            mul_mat_vec_f_cuda(src0_d, src1_d, ids_d, fusion_local, dst_d, ne00, ne01, ncols_dst, s01, stride_col_y, stride_col_dst,
                ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
                ne03,              ne3,           s03, s13,              s3,                 ids_stride, prec, ctx.stream());
        } break;
        case GGML_TYPE_BF16: {
            const nv_bfloat16 * src0_d = (const nv_bfloat16 *) src0->data;
            mul_mat_vec_f_cuda(src0_d, src1_d, ids_d, fusion_local, dst_d, ne00, ne01, ncols_dst, s01, stride_col_y, stride_col_dst,
                ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
                ne03,              ne3,           s03, s13,              s3,                 ids_stride, prec, ctx.stream());
        } break;
        default:
            GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
    }
}

void ggml_cuda_op_mul_mat_vec_f(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne0  =  dst->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const enum ggml_prec prec = fast_fp16_available(cc) ? ggml_prec(dst->op_params[0]) : GGML_PREC_F32;

    // ggml_cuda_op provides single, contiguous matrices
    const int64_t stride_row         = ne00;
    const int64_t stride_col_y       = ne10;
    const int64_t stride_col_dst     = id == ctx.device ? ne0 : row_diff; // main device has larger memory buffer
    const int64_t nchannels_x        = 1;
    const int64_t nchannels_y        = 1;
    const int64_t nchannels_dst      = 1;
    const int64_t stride_channel_x   = 0;
    const int64_t stride_channel_y   = 0;
    const int64_t stride_channel_dst = 0;
    const int64_t nsamples_x         = 1;
    const int64_t nsamples_dst       = 1;
    const int64_t stride_sample_x    = 0;
    const int64_t stride_sample_y    = 0;
    const int64_t stride_sample_dst  = 0;

    ggml_cuda_mm_fusion_args_device empty{};
    switch (src0->type) {
        case GGML_TYPE_F32: {
            const float * src0_d = (const float *) src0_dd_i;
            mul_mat_vec_f_cuda(src0_d, src1_ddf_i, nullptr, empty, dst_dd_i, ne00, row_diff, src1_ncols, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, 0, prec, stream);
        } break;
        case GGML_TYPE_F16: {
            const half * src0_d = (const half *) src0_dd_i;
            mul_mat_vec_f_cuda(src0_d, src1_ddf_i, nullptr, empty, dst_dd_i, ne00, row_diff, src1_ncols, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, 0, prec, stream);
        } break;
        case GGML_TYPE_BF16: {
            const nv_bfloat16 * src0_d = (const nv_bfloat16 *) src0_dd_i;
            mul_mat_vec_f_cuda(src0_d, src1_ddf_i, nullptr, empty, dst_dd_i, ne00, row_diff, src1_ncols, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, 0, prec, stream);
        } break;
        default:
            GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
    }

    GGML_UNUSED_VARS(ctx, src1, dst, src1_ddq_i, src1_ncols, src1_padded_row_size);
}

bool ggml_cuda_should_use_mmvf(enum ggml_type type, int cc, const int64_t * src0_ne, const size_t * src0_nb, int64_t ne11) {
    if (src0_ne[0] % 2 != 0) {
        return false;
    }

    const size_t ts = ggml_type_size(type);
    if (src0_nb[0] != ts) {
        return false;
    }

    // Pointers not aligned to the size of half2/nv_bfloat162/float2 would result in a crash:
    for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
        if (src0_nb[i] % (2*ts) != 0) {
            return false;
        }
    }

    switch (type) {
        case GGML_TYPE_F32:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                if (ampere_mma_available(cc)) {
                    return ne11 <= 3;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    return ne11 <= 4;
                }
                return ne11 <= 3;
            } else if (GGML_CUDA_CC_IS_AMD(cc)) {
                if (fp32_mma_hardware_available(cc)) {
                    return ne11 <= 3;
                }
                return ne11 <= 8;
            }
            return ne11 <= 8;
        case GGML_TYPE_F16:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                const bool src0_small = (src0_ne[1] <= 512 || src0_ne[2]*src0_ne[3] == 1);
                if (ampere_mma_available(cc)) {
                    return src0_small && ne11 == 1;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    return src0_small && ne11 <= 4;
                }
                if (fp16_mma_hardware_available(cc)) {
                    return src0_small && ne11 <= 3;
                }
                return ne11 <= 8;
            } else if (GGML_CUDA_CC_IS_AMD(cc)) {
                if (fp16_mma_hardware_available(cc)) {
                    if (GGML_CUDA_CC_IS_RDNA3(cc)) {
                        return ne11 <= 3;
                    }
                    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
                        return ne11 <= 5;
                    }
                    return ne11 <= 2;
                }
                return ne11 <= 8;
            }
            return ne11 <= 8;
        case GGML_TYPE_BF16:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                const bool src0_small = (src0_ne[1] <= 512 || src0_ne[2]*src0_ne[3] == 1);
                if (ampere_mma_available(cc)) {
                    return src0_small && ne11 == 1;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    return src0_small && ne11 <= 4;
                }
                if (bf16_mma_hardware_available(cc)) {
                    return src0_small && ne11 <= 3;
                }
                return ne11 <= 8;
            } else if (GGML_CUDA_CC_IS_AMD(cc)) {
                if (bf16_mma_hardware_available(cc)) {
                    return ne11 <= 3;
                }
                return ne11 <= 8;
            }
            return ne11 <= 8;
        default:
            return false;
    }
}
