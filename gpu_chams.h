#ifndef GPU_CHAMS_H
#define GPU_CHAMS_H
#include <linux/types.h>
#include <linux/ioctl.h>

struct gpu_chams_cfg {
    __u32 enable;
    __u32 mode;         /* 1=内透 2=线框 3=两者 */
    __u32 depth_off;    /* 关深度测试/写 */
    __u32 ctx_filter;   /* 只处理指定 context_id，0=全 */
    __u32 min_idx;
    __u32 max_idx;
    __s32 tgid;
    __s32 target_mode;  /* 0=全部 1=只改区间内(min_idx..max_idx) 2=只改区间外 */
};

struct gpu_chams_stat {
    __u32 submits;
    __u32 ib_scanned;
    __u32 draws_seen;
    __u32 depth_patched;
    __u32 fail;
    __u32 last_opcode;
    __u32 skip_cached;
    __u32 reserved;
};

struct gpu_chams_dump {
    __u32 enable;
    __u32 segments;
    __u32 dwords;
    __u32 reserved;
};

#define GPUCHAMS_IOC_MAGIC  'G'
#define GPUCHAMS_SET_CFG    _IOW(GPUCHAMS_IOC_MAGIC, 1, struct gpu_chams_cfg)
#define GPUCHAMS_GET_STAT   _IOR(GPUCHAMS_IOC_MAGIC, 2, struct gpu_chams_stat)
#define GPUCHAMS_DUMP_CTRL  _IOWR(GPUCHAMS_IOC_MAGIC, 3, struct gpu_chams_dump)
#define GPUCHAMS_RESET      _IO (GPUCHAMS_IOC_MAGIC, 4)
#define GPUCHAMS_DEV_NAME   "gpuchams"
#endif
