// 生成失败时的应急值：RB_DEPTH_CNTL 按 0 处理（深度改写不生效）
// 修正方法：PC 上跑  python3 06_gen_regs.py --gen a7xx --out adreno_regs.h
// 或者从 IB dump 反查寄存器地址：python3 04_ib_parse.py ib.bin 看「寄存器写次数 Top25」
#ifndef ADRENO_REGS_H
#define ADRENO_REGS_H
#define RB_DEPTH_CNTL            0u
#define RB_DEPTH_CNTL_TEST_EN    0u
#define RB_DEPTH_CNTL_WRITE_EN   0u
#endif
