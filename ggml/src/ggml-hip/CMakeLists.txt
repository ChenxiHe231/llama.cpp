if (NOT EXISTS $ENV{ROCM_PATH})
    if (NOT EXISTS /opt/rocm)
        set(ROCM_PATH /usr)
    else()
        set(ROCM_PATH /opt/rocm)
    endif()
else()
    set(ROCM_PATH $ENV{ROCM_PATH})
endif()

list(APPEND CMAKE_PREFIX_PATH  ${ROCM_PATH})
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}/lib64/cmake")

if (NOT DEFINED CMAKE_HIP_FLAGS_DEBUG)
    set(CMAKE_HIP_FLAGS_DEBUG "-g -O2")
endif()

# CMake on Windows doesn't support the HIP language yet
if (WIN32)
    set(CXX_IS_HIPCC TRUE)
else()
    string(REGEX MATCH "hipcc(\.bat)?$" CXX_IS_HIPCC "${CMAKE_CXX_COMPILER}")
endif()

if (CXX_IS_HIPCC)
    if (LINUX)
        if (NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
            message(WARNING "Only LLVM is supported for HIP, hint: CXX=/opt/rocm/llvm/bin/clang++")
        endif()

        message(WARNING "Setting hipcc as the C++ compiler is legacy behavior."
                " Prefer setting the HIP compiler directly. See README for details.")
    endif()
else()
    # Forward (AMD)GPU_TARGETS to CMAKE_HIP_ARCHITECTURES.
    if(AMDGPU_TARGETS AND NOT GPU_TARGETS)
        set(GPU_TARGETS ${AMDGPU_TARGETS})
    endif()
    if(GPU_TARGETS AND NOT CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES ${GPU_TARGETS})
    endif()
    cmake_minimum_required(VERSION 3.21)
    enable_language(HIP)
endif()

find_package(hip     REQUIRED)
find_package(hipblas REQUIRED)
find_package(rocblas REQUIRED)

if (GGML_HIP_RCCL)
    find_package(rccl REQUIRED)
endif()

if (${hip_VERSION} VERSION_LESS 6.1)
    message(FATAL_ERROR "At least ROCM/HIP V6.1 is required")
endif()

message(STATUS "HIP and hipBLAS found")

file(GLOB   GGML_HEADERS_ROCM "../ggml-cuda/*.cuh")
list(APPEND GGML_HEADERS_ROCM "../../include/ggml-cuda.h")

file(GLOB   GGML_SOURCES_ROCM "../ggml-cuda/*.cu")
file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-tile*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})
file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-mma*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})
file(GLOB   SRCS "../ggml-cuda/template-instances/mmq*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})
file(GLOB   SRCS "../ggml-cuda/template-instances/mmf*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})

if (GGML_CUDA_FA_ALL_QUANTS)
    file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-vec*.cu")
    list(APPEND GGML_SOURCES_ROCM ${SRCS})
    add_compile_definitions(GGML_CUDA_FA_ALL_QUANTS)
else()
    list(APPEND GGML_SOURCES_ROCM
        ../ggml-cuda/template-instances/fattn-vec-instance-f16-f16.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-q4_0-q4_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-q8_0-q8_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-bf16-bf16.cu)
endif()

ggml_add_backend_library(ggml-hip
                         ${GGML_HEADERS_ROCM}
                         ${GGML_SOURCES_ROCM}
                        )

# TODO: do not use CUDA definitions for HIP
if (NOT GGML_BACKEND_DL)
    target_compile_definitions(ggml PUBLIC GGML_USE_CUDA)
endif()

add_compile_definitions(GGML_USE_HIP)

if (GGML_CUDA_FORCE_MMQ)
    add_compile_definitions(GGML_CUDA_FORCE_MMQ)
endif()

if (GGML_CUDA_FORCE_CUBLAS)
    add_compile_definitions(GGML_CUDA_FORCE_CUBLAS)
endif()

if (GGML_CUDA_NO_PEER_COPY)
    add_compile_definitions(GGML_CUDA_NO_PEER_COPY)
endif()

if (GGML_HIP_GRAPHS)
    add_compile_definitions(GGML_HIP_GRAPHS)
endif()

if (GGML_HIP_NO_VMM)
    add_compile_definitions(GGML_HIP_NO_VMM)
endif()

if (GGML_HIP_ROCWMMA_FATTN)
    add_compile_definitions(GGML_HIP_ROCWMMA_FATTN)
endif()

if (NOT GGML_HIP_MMQ_MFMA)
    add_compile_definitions(GGML_HIP_NO_MMQ_MFMA)
endif()

if (GGML_HIP_RCCL)
    add_compile_definitions(GGML_USE_NCCL) # RCCL has the same interface as NCCL.
endif()

if (GGML_HIP_EXPORT_METRICS)
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Rpass-analysis=kernel-resource-usage --save-temps")
endif()

if (NOT GGML_CUDA_FA)
    add_compile_definitions(GGML_CUDA_NO_FA)
endif()

if (CXX_IS_HIPCC)
    set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES LANGUAGE CXX)
    if (WIN32 AND CMAKE_BUILD_TYPE STREQUAL "Debug")
        # CMake on Windows doesn't support the HIP language yet.
        # Therefore we workaround debug build's failure on HIP backend this way.
        set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES COMPILE_FLAGS "-O2 -g")
    endif()
    target_link_libraries(ggml-hip PRIVATE hip::device)
else()
    set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES LANGUAGE HIP)
endif()

if (GGML_STATIC)
    message(FATAL_ERROR "Static linking not supported for HIP/ROCm")
endif()

if (GGML_HIP_RCCL)
    target_link_libraries(ggml-hip PRIVATE ggml-base roc::rccl)
endif()

target_link_libraries(ggml-hip PRIVATE ggml-base hip::host roc::rocblas roc::hipblas)

# --- CK ck_tile FMHA integration (spike) ---------------------------------
# Compile fattn-ck.cu against CK's own ck_tile headers and link the pre-built
# generated instance objects + the fmha_fwd_api object. Enabled by -DGGML_HIP_CK_FATTN=ON.
#
# This path is a spike: it links PRE-GENERATED CK ck_tile FMHA instance objects
# rather than building CK from source in-tree. You must point it at a CK checkout
# and a directory of pre-built (-fPIC) instance .o files via cache variables:
#
#   -DGGML_HIP_CK_FATTN=ON
#   -DCK_SRC=/path/to/composable_kernel            (CK source checkout)
#   -DCK_INSTDIR=/path/to/prebuilt-ck-instances    (fmha_fwd_d256_fp16_batch_*.o + fmha_fwd_api.cpp.o, all -fPIC)
#   [-DCK_BUILD=/path/to/ck/build]                 (defaults to ${CK_SRC}/build)
#
# The instance objects are produced by CK's codegen (example/ck_tile/01_fmha) filtered
# to the d256/fp16 case and recompiled with -fPIC (ggml-hip is a shared lib). See
# stage25 notes for the generation recipe. TODO(productionize): replace this pre-built
# link with a proper vendored-CK ExternalProject/codegen step so the build is
# self-contained and portable (Windows/other machines).
if (GGML_HIP_CK_FATTN)
    set(CK_SRC     "" CACHE PATH "Composable Kernel source checkout (contains include/, library/, example/ck_tile)")
    set(CK_INSTDIR "" CACHE PATH "Directory of pre-built (-fPIC) CK ck_tile FMHA instance .o files")
    set(CK_BUILD   "" CACHE PATH "CK build dir with generated headers (defaults to \${CK_SRC}/build)")

    if (NOT CK_SRC OR NOT IS_DIRECTORY "${CK_SRC}")
        message(FATAL_ERROR "GGML_HIP_CK_FATTN=ON requires -DCK_SRC=<composable_kernel checkout>; got '${CK_SRC}'")
    endif()
    if (NOT CK_INSTDIR OR NOT IS_DIRECTORY "${CK_INSTDIR}")
        message(FATAL_ERROR "GGML_HIP_CK_FATTN=ON requires -DCK_INSTDIR=<dir of pre-built CK instance .o>; got '${CK_INSTDIR}'")
    endif()
    if (NOT CK_BUILD)
        set(CK_BUILD "${CK_SRC}/build")
    endif()

    add_compile_definitions(GGML_HIP_CK_FATTN)

    set_source_files_properties(
        ../ggml-cuda/fattn-ck.cu PROPERTIES
        INCLUDE_DIRECTORIES
        "${CK_SRC}/include;${CK_BUILD}/include;${CK_SRC}/library/include;${CK_SRC}/example/ck_tile;${CK_SRC}/example/ck_tile/01_fmha"
        COMPILE_OPTIONS
        "-DCK_TILE_USE_WMMA=1;-DCK_USE_WMMA;-DCK_ENABLE_FP16;-DCK_TILE_FMHA_FWD_FAST_EXP2=1;-fbracket-depth=1024;-Wno-everything")

    file(GLOB CK_INSTANCE_OBJS "${CK_INSTDIR}/fmha_fwd_d256_fp16_batch_*.o")
    list(APPEND CK_INSTANCE_OBJS "${CK_INSTDIR}/fmha_fwd_api.cpp.o")
    if (NOT CK_INSTANCE_OBJS)
        message(FATAL_ERROR "No CK instance objects found in CK_INSTDIR='${CK_INSTDIR}' (expected fmha_fwd_d256_fp16_batch_*.o)")
    endif()
    target_link_libraries(ggml-hip PRIVATE ${CK_INSTANCE_OBJS})
endif()

