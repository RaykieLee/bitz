/*
 * CUDA优化挖矿实现
 * 
 * 主要优化点：
 * 1. 避免在GPU内核中使用动态内存分配（malloc/free），改用预分配的全局内存
 * 2. 将哈希计算和解算过程分成两个阶段，分别使用不同的并行策略
 * 3. 哈希计算阶段：一个线程计算一个哈希值，充分利用GPU并行能力
 * 4. 解算阶段：一个线程处理一个批次的全部解算过程
 * 5. 移除不必要的同步点，减少线程等待
 * 6. 优化线程数和块大小配置，减少资源冲突
 */

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
#include <hashx_endian.h>

// 索引空间大小定义
#ifndef INDEX_SPACE
#define INDEX_SPACE 65536
#endif

// GPU批处理大小，可根据GPU性能调整，默认值
// 实际运行时会从环境变量读取
const int BATCH_SIZE = 2048;  // 提高默认批处理大小

// 从环境变量读取整数参数，如果未设置则使用默认值
int get_env_int(const char* name, int default_value) {
    const char* value = getenv(name);
    if (value == NULL) {
        return default_value;
    }
    return atoi(value);
}

// 前向声明GPU内核函数和其他函数
__global__ void gpu_hash_kernel(hashx_ctx** ctxs, uint64_t** hash_space, int batch_size);
__global__ void gpu_solve_all_stages_kernel(uint64_t* hashes, solver_heap* heaps, equix_solution* solutions, uint32_t* num_sols);

// 从equix/src/solver.cu导入的函数，这些函数已在外部定义
__device__ void hash_stage0i(hashx_ctx* hash_func, uint64_t* out, uint32_t i);
__device__ void solve_stage0(uint64_t* hashes, solver_heap* heap);
__device__ void solve_stage1(solver_heap* heap);
__device__ void solve_stage2(solver_heap* heap);
__device__ uint32_t solve_stage3(solver_heap* heap, equix_solution output[EQUIX_MAX_SOLS]);

// 优化的哈希计算内核 - 一个批次使用多个线程
__global__ void optimized_hash_kernel(hashx_ctx** ctxs, uint64_t* hash_space, int batch_size) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = global_idx / INDEX_SPACE;
    int hash_idx = global_idx % INDEX_SPACE;
    
    // 确保只有有效范围内的线程工作
    if (batch_idx < batch_size && hash_idx < INDEX_SPACE) {
        uint64_t* batch_hash_space = hash_space + (batch_idx * INDEX_SPACE);
        hash_stage0i(ctxs[batch_idx], batch_hash_space, hash_idx);
    }
}

// 合并的挖矿内核 - 在GPU上完成所有步骤，但拆分为多个阶段以提高效率
__global__ void unified_mining_kernel(hashx_ctx** ctxs, uint64_t* nonce_base, uint64_t* global_hash_space, solver_heap* heaps, equix_solution* solutions, uint32_t* num_sols, int batch_size) {
    // 获取线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 只处理有效范围内的批次
    if (idx >= batch_size) return;
    
    // 使用预分配的全局内存，避免动态内存分配
    uint64_t* thread_hashes = global_hash_space + (idx * INDEX_SPACE);
    
    // 阶段2-5: 在GPU上解算所有阶段
    solver_heap* thread_heap = &heaps[idx];
    equix_solution* thread_solutions = &solutions[idx * EQUIX_MAX_SOLS];
    
    // 执行所有阶段的解算
    solve_stage0(thread_hashes, thread_heap);
    solve_stage1(thread_heap);
    solve_stage2(thread_heap);
    num_sols[idx] = solve_stage3(thread_heap, thread_solutions);
}

// 替换原有的两个分离函数，使用单一的函数调用完成所有GPU工作
extern "C" void gpu_unified_mine(uint8_t *challenge, uint8_t *nonce, uint8_t *out, uint32_t *sols, int num_sets) {
    // 使用传入的num_sets作为批处理大小
    int batch_size = num_sets;
    // 如果num_sets为0或负数，则使用默认值
    if (batch_size <= 0) {
        batch_size = get_env_int("BITZ_GPU_BATCH_SIZE", BATCH_SIZE);
    }
    
    // 哈希阶段和解算阶段使用不同的线程配置
    int hash_threads_per_block = get_env_int("BITZ_GPU_HASH_THREADS", 256);
    int solve_threads_per_block = get_env_int("BITZ_GPU_SOLVE_THREADS", 256);
    
    // 分配设备内存
    hashx_ctx** d_ctxs;
    uint64_t* d_nonce_base;
    uint64_t* d_hash_space;  // 预分配哈希空间
    solver_heap* d_heaps;
    equix_solution* d_solutions;
    uint32_t* d_num_sols;

    // 分配固定主机内存和设备内存
    cudaMalloc(&d_ctxs, batch_size * sizeof(hashx_ctx*));
    cudaMalloc(&d_nonce_base, batch_size * sizeof(uint64_t));
    cudaMalloc(&d_hash_space, batch_size * INDEX_SPACE * sizeof(uint64_t));  // 全局预分配内存
    cudaMalloc(&d_heaps, batch_size * sizeof(solver_heap));
    cudaMalloc(&d_solutions, batch_size * EQUIX_MAX_SOLS * sizeof(equix_solution));
    cudaMalloc(&d_num_sols, batch_size * sizeof(uint32_t));
    
    equix_solution* h_solutions;
    uint32_t* h_num_sols;
    cudaHostAlloc(&h_solutions, batch_size * EQUIX_MAX_SOLS * sizeof(equix_solution), cudaHostAllocDefault);
    cudaHostAlloc(&h_num_sols, batch_size * sizeof(uint32_t), cudaHostAllocDefault);

    // 准备种子和哈希上下文
    hashx_ctx** h_ctxs = (hashx_ctx**)malloc(batch_size * sizeof(hashx_ctx*));
    uint64_t nonce_value = *((uint64_t*)nonce);
    uint64_t* h_nonce_base = (uint64_t*)malloc(batch_size * sizeof(uint64_t));
    
    uint8_t seed[40];
    memcpy(seed, challenge, 32);
    
    for (int i = 0; i < batch_size; i++) {
        h_nonce_base[i] = nonce_value + i;
        memcpy(seed + 32, &h_nonce_base[i], 8);
        h_ctxs[i] = hashx_alloc(HASHX_INTERPRETED);
        if (!h_ctxs[i] || !hashx_make(h_ctxs[i], seed, 40)) {
            free(h_ctxs);
            free(h_nonce_base);
            return;
        }
    }
    
    // 初始化设备内存
    cudaMemset(d_hash_space, 0, batch_size * INDEX_SPACE * sizeof(uint64_t));
    cudaMemset(d_num_sols, 0, batch_size * sizeof(uint32_t));
    
    // 复制数据到GPU
    cudaMemcpy(d_ctxs, h_ctxs, batch_size * sizeof(hashx_ctx*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nonce_base, h_nonce_base, batch_size * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    // 第一阶段：哈希计算 - 每个哈希值分配一个线程
    int total_hashes = batch_size * INDEX_SPACE;
    int hash_blocks = (total_hashes + hash_threads_per_block - 1) / hash_threads_per_block;
    optimized_hash_kernel<<<hash_blocks, hash_threads_per_block>>>(d_ctxs, d_hash_space, batch_size);
    
    // 确保哈希计算完成后再进行解算
    cudaDeviceSynchronize();
    
    // 第二阶段：解算 - 每个批次一个线程
    int solve_blocks = (batch_size + solve_threads_per_block - 1) / solve_threads_per_block;
    unified_mining_kernel<<<solve_blocks, solve_threads_per_block>>>(d_ctxs, d_nonce_base, d_hash_space, d_heaps, d_solutions, d_num_sols, batch_size);
    
    // 直接将结果复制回主机 - 只有最终结果需要传回CPU
    cudaMemcpy(h_solutions, d_solutions, batch_size * EQUIX_MAX_SOLS * sizeof(equix_solution), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_num_sols, d_num_sols, batch_size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    // 处理结果
    for (int i = 0; i < batch_size; i++) {
        sols[i] = h_num_sols[i];
        if (h_num_sols[i] > 0) {
            memcpy(out + i * sizeof(equix_solution), &h_solutions[i * EQUIX_MAX_SOLS], sizeof(equix_solution));
        }
    }
    
    // 释放资源
    for (int i = 0; i < batch_size; i++) {
        hashx_free(h_ctxs[i]);
    }
    free(h_ctxs);
    free(h_nonce_base);
    
    cudaFree(d_ctxs);
    cudaFree(d_nonce_base);
    cudaFree(d_hash_space);  // 释放预分配的哈希空间
    cudaFree(d_heaps);
    cudaFree(d_solutions);
    cudaFree(d_num_sols);
    
    cudaFreeHost(h_solutions);
    cudaFreeHost(h_num_sols);
}

// 保留原来的函数以兼容旧代码，但内部调用新的统一函数
extern "C" void gpu_hash(uint8_t *challenge, uint8_t *nonce, uint64_t *out) {
    printf("WARNING: 使用了旧的gpu_hash函数，建议使用gpu_unified_mine以获得更好性能\n");
    
    // 实际实现保持不变，用于兼容性
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 计算批次索引和项目索引
    int batch_idx = idx / INDEX_SPACE;
    int i = idx % INDEX_SPACE;
    
    // 只处理有效范围内的索引
    if (batch_idx < batch_size && i < INDEX_SPACE) {
        // 使用hash_stage0i函数进行哈希计算
        hash_stage0i(ctxs[batch_idx], hash_space[batch_idx], i);
    }
}

extern "C" void gpu_solve_stages(uint64_t *hashes, uint8_t *out, uint32_t *sols, int num_sets) {
    printf("WARNING: 使用了旧的gpu_solve_stages函数，建议使用gpu_unified_mine以获得更好性能\n");
    
    // 实际实现保持不变，用于兼容性
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
    gpu_solve_all_stages_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_hashes, d_heaps, d_solutions, d_num_sols);

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

// 自定义的解决方案求解内核 - 处理所有阶段（改名以避免冲突）
__global__ void gpu_solve_all_stages_kernel(uint64_t* hashes, solver_heap* heaps, equix_solution* solutions, uint32_t* num_sols) {
    // 获取线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 检查线程索引是否有效
    if (idx >= gridDim.x * blockDim.x) return;
    
    // 指向当前批次的数据
    uint64_t* thread_hashes = hashes + idx * INDEX_SPACE;
    solver_heap* thread_heap = &heaps[idx];
    equix_solution* thread_solutions = &solutions[idx * EQUIX_MAX_SOLS];
    
    // 执行所有求解阶段
    solve_stage0(thread_hashes, thread_heap);
    __syncthreads();
    solve_stage1(thread_heap);
    __syncthreads();
    solve_stage2(thread_heap);
    __syncthreads();
    num_sols[idx] = solve_stage3(thread_heap, thread_solutions);
} 