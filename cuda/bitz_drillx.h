#ifndef BITZ_DRILLX_H
#define BITZ_DRILLX_H

#include <stdint.h>
#include "equix.h"
#include "hashx.h"

#ifdef __cplusplus
extern "C" {
#endif

// 批处理大小常量
extern const int BATCH_SIZE;

// 获取环境变量中的整数值
int get_env_int(const char* name, int default_value);

// 旧的分离函数
void gpu_hash(uint8_t *challenge, uint8_t *nonce, uint64_t *out);
void gpu_solve_stages(uint64_t *hashes, uint8_t *out, uint32_t *sols, int num_sets);

// 新的统一挖矿函数 - 所有步骤在GPU内完成，无中间数据传输
void gpu_unified_mine(uint8_t *challenge, uint8_t *nonce, uint8_t *out, uint32_t *sols, int num_sets);

#ifdef __cplusplus
}
#endif

#endif 