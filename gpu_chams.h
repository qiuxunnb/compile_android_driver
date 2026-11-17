#ifndef GPU_CHAMS_H
#define GPU_CHAMS_H
#include <linux/types.h>
#include <linux/ioctl.h>

struct gpu_chams_cfg {
    __u32 enable;       /* 总开关，默认 0（装载后不动手） */
    __u32 mode;         /* 1=内透 2=线框 3=两者 */
    __u32 depth_off;    /* 1=关深度测试/写入 */
    __u32 ctx_filter;   /* 只处理指定 context_id，0=全 */
    __u32 min_idx;      /* 索引数下限 */
    __u32 max_idx;      /* 索引数上限 */
    __s32 tgid;         /* 目标进程 */
    __s32 target_mode;  /* 0=全部 1=只改区间内 2=只改区间外 */
};

struct gpu_chams_stat {
    __u32 submits;
    __u32 cache_hit;
    __u32 cache_miss;
    __u32 draws_seen;
    __u32 depth_patched;
    __u32 filtered;
    __u32 last_opcode;
    __u32 cache_used;
};

struct gpu_chams_hist {
    __u32 draws;
    __u32 n;
    __u32 reg[8];
    __u32 cnt[8];
    __u32 reserved[2];
};

#define GPUCHAMS_IOC_MAGIC  'G'
#define GPUCHAMS_SET_CFG    _IOW(GPUCHAMS_IOC_MAGIC, 1, struct gpu_chams_cfg)
#define GPUCHAMS_GET_STAT   _IOR(GPUCHAMS_IOC_MAGIC, 2, struct gpu_chams_stat)
#define GPUCHAMS_FLUSH      _IOR(GPUCHAMS_IOC_MAGIC, 3, struct gpu_chams_stat)
#define GPUCHAMS_GET_HIST   _IOR(GPUCHAMS_IOC_MAGIC, 4, struct gpu_chams_hist)
#define GPUCHAMS_RESET      _IO (GPUCHAMS_IOC_MAGIC, 5)
#define GPUCHAMS_DEV_NAME   "gpuchams"
#endif