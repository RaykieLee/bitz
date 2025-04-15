#ifndef BITZ_DRILLX_H
#define BITZ_DRILLX_H

#include "equix.h"
#include "hashx.h"
#include "equix/src/context.h"
#include "equix/src/solver_heap.h"
#include "hashx/src/context.h"

extern "C" const int BATCH_SIZE;

extern "C" void gpu_hash(uint8_t *challenge, uint8_t *nonce, uint64_t *out);

extern "C" void gpu_solve_stages(uint64_t *hashes, uint8_t *out, uint32_t *sols, int num_sets);

__global__ void gpu_hash_kernel(hashx_ctx** ctxs, uint64_t** hash_space);

#endif 