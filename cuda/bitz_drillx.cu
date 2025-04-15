#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "bitz_drillx.h"
#include "equix.h"
#include "hashx.h"
#include "equix/src/context.h"
#include "equix/src/solver.h"
#include "equix/src/solver_heap.h"
#include "hashx/src/context.h"

// GPU批处理大小，可根据GPU性能调整，默认值
// 实际运行时会从环境变量读取
const int BATCH_SIZE = 1024;

// 从环境变量读取整数参数，如果未设置则使用默认值
int get_env_int(const char* name, int default_value) {
    const char* value = getenv(name);
    if (value == NULL) {
        return default_value;
    }
    return atoi(value);
}

extern "C" void gpu_hash(uint8_t *challenge, uint8_t *nonce, uint64_t *out) {
    // 从环境变量读取批处理大小和线程参数
    int batch_size = get_env_int("BITZ_GPU_BATCH_SIZE", BATCH_SIZE);
    int hash_threads = get_env_int("BITZ_GPU_HASH_THREADS", 384);
    
    // 分配固定内存，提高传输效率
    hashx_ctx** ctxs;
    uint64_t** hash_space;

    cudaMallocHost(&ctxs, batch_size * sizeof(hashx_ctx*));
    cudaMallocHost(&hash_space, batch_size * sizeof(uint64_t*));

    for (int i = 0; i < batch_size; i++) {
        cudaMalloc(&hash_space[i], INDEX_SPACE * sizeof(uint64_t));
    }

    // 准备种子和哈希上下文
    uint8_t seed[40];
    memcpy(seed, challenge, 32);
    for (int i = 0; i < batch_size; i++) {
        uint64_t nonce_offset = *((uint64_t*)nonce) + i;
        memcpy(seed + 32, &nonce_offset, 8);
        ctxs[i] = hashx_alloc(HASHX_INTERPRETED);
        if (!ctxs[i] || !hashx_make(ctxs[i], seed, 40)) {
            cudaFreeHost(ctxs);
            return;
        }
    }

    // 启动内核进行并行哈希计算 - 使用动态线程设置
    dim3 threadsPerBlock(hash_threads);
    dim3 blocksPerGrid((65536 * batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    gpu_hash_kernel<<<blocksPerGrid, threadsPerBlock>>>(ctxs, hash_space, batch_size);
    
    // 将哈希结果复制回CPU
    for (int i = 0; i < batch_size; i++) {
        cudaMemcpy(out + i * INDEX_SPACE, hash_space[i], INDEX_SPACE * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    }

    // 释放内存
    for (int i = 0; i < batch_size; i++) {
        hashx_free(ctxs[i]);
        cudaFree(hash_space[i]);
    }
    cudaFreeHost(hash_space);
    cudaFreeHost(ctxs);
}

__global__ void gpu_hash_kernel(hashx_ctx** ctxs, uint64_t** hash_space, int batch_size) {
    uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t batch_idx = item / INDEX_SPACE;
    uint32_t i = item % INDEX_SPACE;
    if (batch_idx < batch_size) {
        hash_stage0i(ctxs[batch_idx], hash_space[batch_idx], i);
    }
}

extern "C" void gpu_solve_stages(uint64_t *hashes, uint8_t *out, uint32_t *sols, int num_sets) {
    // 从环境变量读取解算阶段的线程参数
    int solve_threads = get_env_int("BITZ_GPU_SOLVE_THREADS", 192);
    
    // 分配设备内存
    uint64_t *d_hashes;
    solver_heap *d_heaps;
    equix_solution *d_solutions;
    uint32_t *d_num_sols;

    cudaMalloc(&d_hashes, num_sets * INDEX_SPACE * sizeof(uint64_t));
    cudaMalloc(&d_heaps, num_sets * sizeof(solver_heap));
    cudaMalloc(&d_solutions, num_sets * EQUIX_MAX_SOLS * sizeof(equix_solution));
    cudaMalloc(&d_num_sols, num_sets * sizeof(uint32_t));

    // 分配固定主机内存
    equix_solution *h_solutions;
    uint32_t *h_num_sols;
    cudaHostAlloc(&h_solutions, num_sets * EQUIX_MAX_SOLS * sizeof(equix_solution), cudaHostAllocDefault);
    cudaHostAlloc(&h_num_sols, num_sets * sizeof(uint32_t), cudaHostAllocDefault);

    // 复制输入数据到设备
    cudaMemcpy(d_hashes, hashes, num_sets * INDEX_SPACE * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // 启动内核 - 使用动态线程设置
    int threadsPerBlock = solve_threads;
    int blocksPerGrid = (num_sets + threadsPerBlock - 1) / threadsPerBlock;
    solve_all_stages_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_hashes, d_heaps, d_solutions, d_num_sols);

    // 使用固定内存将结果复制回主机
    cudaMemcpy(h_solutions, d_solutions, num_sets * EQUIX_MAX_SOLS * sizeof(equix_solution), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_num_sols, d_num_sols, num_sets * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // 处理结果
    for (int i = 0; i < num_sets; i++) {
        sols[i] = h_num_sols[i];
        if (h_num_sols[i] > 0) {
            memcpy(out + i * sizeof(equix_solution), &h_solutions[i * EQUIX_MAX_SOLS], sizeof(equix_solution));
        }
    }

    // 释放设备内存
    cudaFree(d_hashes);
    cudaFree(d_heaps);
    cudaFree(d_solutions);
    cudaFree(d_num_sols);

    // 释放固定主机内存
    cudaFreeHost(h_solutions);
    cudaFreeHost(h_num_sols);
} 