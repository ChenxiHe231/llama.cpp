#!/bin/bash
# ============================================================================
# llama.cpp HIP/ROCm 算子优化 profiler 脚本 — Linux 端
# ============================================================================
# 用法:
#   source scripts/profile_ops.sh                # 加载函数到当前 shell
#   profile_ops_setup                             # 首次使用: 配置构建
#   profile_ops_perf MUL_MAT                      # 算子级 perf (rocprofv3 --stats)
#   profile_ops_pmc MUL_MAT "F16,F32"            # 硬件 PMC 计数器 (rocprofv3 -i pmc)
#   profile_ops_omni MUL_MAT                      # 全功能 omniperf 分析
#   profile_ops_bench SOCKS 32K 128               # 端到端 llama-bench
# ============================================================================

export LLAMA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LLAMA_BUILD="$LLAMA_ROOT/build"
export PROFILE_OUT="$LLAMA_ROOT/profiles"
export GPU_TARGET="gfx1151"  # Strix Halo / Radeon 8060S

# ---------------------------------------------------------------------------
# 初始化: 构建 + 输出目录
# ---------------------------------------------------------------------------
profile_ops_setup() {
    echo "=== Building llama.cpp for $GPU_TARGET ==="
    mkdir -p "$PROFILE_OUT"

    cmake -B "$LLAMA_BUILD" \
        -DGGML_HIP=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DAMDGPU_TARGETS="$GPU_TARGET" \
        -DGGML_HIP_EXPORT_METRICS=ON \
        -G Ninja

    cmake --build "$LLAMA_BUILD" -j --target test-backend-ops llama-bench

    echo "=== Binaries ==="
    ls -lh "$LLAMA_BUILD/bin/test-backend-ops" "$LLAMA_BUILD/bin/llama-bench"

    echo "=== GPU ==="
    rocminfo | grep -E "Name:|gfx"
}

# ---------------------------------------------------------------------------
# 方法 A: rocprofv3 --stats — per-kernel 耗时 + 调用次数 (轻量)
# ---------------------------------------------------------------------------
profile_ops_perf() {
    local OP="${1:-MUL_MAT}"
    local PARAMS="${2:-}"
    local NAME="${OP}_perf"
    local OUTDIR="$PROFILE_OUT/$NAME"

    mkdir -p "$OUTDIR"

    local PARAM_FLAG=""
    [ -n "$PARAMS" ] && PARAM_FLAG="-p \"$PARAMS\""

    echo "=== rocprofv3 --stats for $OP (params: ${PARAMS:-all}) ==="

    rocprofv3 --stats \
        --output-dir "$OUTDIR" \
        "$LLAMA_BUILD/bin/test-backend-ops" perf -b HIP -o "$OP" $PARAM_FLAG --output csv

    echo "=== Results in $OUTDIR ==="
    ls -la "$OUTDIR/"
    echo "=== Per-kernel summary ==="
    cat "$OUTDIR/results.stats.csv" 2>/dev/null | head -30
}

# ---------------------------------------------------------------------------
# 方法 B: rocprofv3 自定义 PMC 计数器 — 硬件 stall/cache miss/occupancy
# ---------------------------------------------------------------------------
# 生成 pmc 配置文件，然后运行 rocprofv3 并汇总结果
# 参考: https://rocm.docs.amd.com/projects/rocprofiler/en/latest/
profile_ops_pmc() {
    local OP="${1:-MUL_MAT}"
    local PARAMS="${2:-}"
    local NAME="${OP}_pmc"
    local OUTDIR="$PROFILE_OUT/$NAME"
    local PMCFILE="$PROFILE_OUT/pmc_counters.txt"

    mkdir -p "$OUTDIR"

    # 关键硬件计数器 (RDNA3/gfx11)
    cat > "$PMCFILE" << 'EOF'
# === 计算单元占用率 ===
pmc: SQ_WAVES                 # 活跃 wavefront 数
pmc: SQ_VALU_BUSY_CYCLES      # VALU 忙碌周期
pmc: SQ_SALU_BUSY_CYCLES      # SALU 忙碌周期
# === 内存延迟来源 ===
pmc: SQ_INST_STALL_LDS        # LDS 冲突 stall
pmc: SQ_INST_STALL_VALU       # VALU 依赖 stall
pmc: SQ_INST_STALL_SMEM       # 标量内存 stall
pmc: SQ_INST_STALL_VMEM       # 向量内存 stall
# === 缓存效率 ===
pmc: TCP_TCC_READ_REQ_SUM     # L1 读请求
pmc: TCP_TCC_READ_HIT_SUM     # L1 读命中
pmc: TCP_TCC_WRITE_REQ_SUM    # L1 写请求
pmc: TCP_TCC_WRITE_HIT_SUM    # L1 写命中
# PCIe/带宽
pmc: TCC_EA_RDREQ_32B_SUM     # 显存读请求
pmc: TCC_EA_WRREQ_SUM         # 显存写请求
EOF

    local PARAM_FLAG=""
    [ -n "$PARAMS" ] && PARAM_FLAG="-p \"$PARAMS\""

    echo "=== rocprofv3 with PMC counters for $OP ==="

    rocprofv3 \
        --pmc "$PMCFILE" \
        --output-dir "$OUTDIR" \
        "$LLAMA_BUILD/bin/test-backend-ops" perf -b HIP -o "$OP" $PARAM_FLAG --output csv

    echo "=== PMC results in $OUTDIR ==="
    ls -la "$OUTDIR/"
}

# ---------------------------------------------------------------------------
# 方法 C: omniperf — 全功能 UI 分析 (推荐, 数据最全)
# ---------------------------------------------------------------------------
profile_ops_omni() {
    local OP="${1:-MUL_MAT}"
    local PARAMS="${2:-}"
    local NAME="${OP}_omni"

    echo "=== omniperf profile for $OP ==="

    local PARAM_FLAG=""
    [ -n "$PARAMS" ] && PARAM_FLAG="-p \"$PARAMS\""

    omniperf profile \
        --name "$NAME" \
        --workdir "$PROFILE_OUT" \
        -- "$LLAMA_BUILD/bin/test-backend-ops" perf -b HIP -o "$OP" $PARAM_FLAG --output csv

    echo "=== Analyze with: omniperf analyze -p $PROFILE_OUT/workloads/$NAME/ ==="
}

# ---------------------------------------------------------------------------
# 方法 D: 端到端 llama-bench
# ---------------------------------------------------------------------------
profile_ops_bench() {
    local MODEL="${1:-models/llama-8b-Q4_K_M.gguf}"
    local PROMPT="${2:-512,4096,8192,32768}"
    local GEN="${3:-128}"
    local NAME="bench"
    local OUTDIR="$PROFILE_OUT/$NAME"

    mkdir -p "$OUTDIR"

    echo "=== llama-bench: $MODEL ==="

    # 纯 prefill
    "$LLAMA_BUILD/bin/llama-bench" \
        -m "$MODEL" \
        -p "$PROMPT" -n 0 \
        -ngl 99 -fa 1 \
        -r 5 --no-warmup \
        -o csv 2>&1 | tee "$OUTDIR/bench_prefill.csv"

    # 纯 decode
    "$LLAMA_BUILD/bin/llama-bench" \
        -m "$MODEL" \
        -p 512 -n "$GEN" \
        -ngl 99 -fa 1 \
        -r 5 --no-warmup \
        -o csv 2>&1 | tee "$OUTDIR/bench_decode.csv"

    echo "=== Results in $OUTDIR ==="
}

# ---------------------------------------------------------------------------
# 快捷命令: 一键 profile 单个算子 (perf + pmc 两步)
# ---------------------------------------------------------------------------
profile_ops_all() {
    local OP="${1:-MUL_MAT}"
    local PARAMS="${2:-}"

    echo ">>> Profiling $OP (params: ${PARAMS:-all})"
    profile_ops_perf "$OP" "$PARAMS"
    profile_ops_pmc "$OP" "$PARAMS"
    echo ">>> Done. Run: profile_ops_omni $OP \"$PARAMS\" for omniperf UI"
}

# ---------------------------------------------------------------------------
# 快速看真实推理中最耗时的 kernel (不改代码, rocprofv3 --stats)
# ---------------------------------------------------------------------------
profile_ops_real_breakdown() {
    local MODEL="${1}"
    local PROMPT="${2:-hello}"
    local N_GEN="${3:-20}"

    if [ -z "$MODEL" ]; then
        echo "Usage: profile_ops_real_breakdown <model.gguf> [prompt] [n_gen]"
        return 1
    fi

    local NAME="real_breakdown"
    local OUTDIR="$PROFILE_OUT/$NAME"
    mkdir -p "$OUTDIR"

    echo "=== rocprofv3 --stats on real inference: $MODEL ==="
    echo "    prompt: '$PROMPT', n_gen: $N_GEN"

    rocprofv3 --stats \
        --output-dir "$OUTDIR" \
        "$LLAMA_BUILD/bin/llama-cli" -m "$MODEL" -p "$PROMPT" -n "$N_GEN" --no-display-prompt

    echo "=== Top 20 most expensive kernels ==="
    if [ -f "$OUTDIR/results.stats.csv" ]; then
        cat "$OUTDIR/results.stats.csv" | sort -t',' -k3 -rn | head -20
    fi
    echo "=== Full results in $OUTDIR/ ==="
}

echo "[profile_ops.sh] loaded. Run: profile_ops_setup && profile_ops_all MUL_MAT"

# ============================================================================
# 便携参考卡 (切到 Linux 后不需要 Claude memory 也能看到)
# ============================================================================
# 环境:
#   GPU: AMD Radeon 8060S (gfx1151, Strix Halo APU, 20CU, 110GB UMA VRAM)
#   ROCm: Linux 端需 >= 6.x, 推荐 7.x
#   Repo: git@github.com:runboo-fly/llama.cpp.git (origin)
#         https://github.com/ggml-org/llama.cpp.git (upstream)
#
# 首次 setup:
#   git clone git@github.com:runboo-fly/llama.cpp.git
#   cd llama.cpp
#   source scripts/profile_ops.sh
#   profile_ops_setup
#
# 快速看真实推理中最耗时的 kernel:
#   profile_ops_real_breakdown <model.gguf> "hello" 20
#   (本质是 rocprofv3 --stats 包住 llama-cli, 不改代码, 立即可用)
#
# 工作流:
#   Linux:  profile_ops_real_breakdown → 找到最耗时 kernel
#           → profile_ops_omni <OP> → 看 PMC 瓶颈
#           → 改 kernel 源码 → commit & push
#   Windows: git pull → cmake --build build-amd --config Release -j
#            → test-backend-ops.exe perf -b ROCm0 -o <OP> --output csv  (A/B 对比)
#            → llama-bench.exe -m <model> -p 4096,8192 -n 128 -ngl 99 -fa 1  (最终判据)
#
# 注意: Windows 上 backend 名是 "ROCm0" 不是 "HIP"