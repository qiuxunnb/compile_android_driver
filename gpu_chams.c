// ============================================================
//  gpu_chams.c —— B 方案：零内核结构体版（探针 + 翻译器）
//
//  【为什么要这么写】
//  之前崩的根因猜测：模块用 6.6.144 头文件编译，设备跑 6.6.89，
//  传给内核的结构体（miscdevice / kprobe / kernel_param_ops）字段布局
//  可能不同 -> 内核读到垃圾指针 -> panic。
//
//  【本版原则】
//  · 只使用 module_param(ulong/int) —— 内核内置的 param_ops_ulong
//  · 不建设备节点（无 misc_register）
//  · 不注册 kprobe（无 register_kprobe）
//  · 只用「函数指针调用」：把用户态查到的内核符号地址当函数指针用
//  · 用完主动 return -EINVAL，让 insmod 失败回滚，不留任何状态
//
//  【用法】
//  1) 用户态拿地址：
//       su -c "echo 0 > /proc/sys/kernel/kptr_restrict"
//       su -c "grep -E ' (kgsl_sharedmem_find|kgsl_get_device)$' /proc/kallsyms"
//  2) 带参数装载：
//       insmod gpu_chams.ko p_find=0x.... p_getdev=0x.... p_gpuaddr=0x....
//  3) 看结果：
//       dmesg | grep GPUC
// ============================================================
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/sched.h>
#include <linux/string.h>
#include <linux/errno.h>
#include <linux/types.h>

#define GC_TAG "GPUC"

/* ── 全部参数都是基本类型，零结构体 ── */
static unsigned long p_find    = 0;   /* kgsl_sharedmem_find 地址 */
static unsigned long p_getdev  = 0;   /* kgsl_get_device 地址 */
static unsigned long p_gpuaddr = 0;   /* 要翻译的 GPU 地址 */
static unsigned long p_devaddr = 0;   /* 可选：已知的 kgsl_device 指针 */
static int p_verbose = 1;

module_param(p_find,    ulong, 0644);
module_param(p_getdev,  ulong, 0644);
module_param(p_gpuaddr, ulong, 0644);
module_param(p_devaddr, ulong, 0644);
module_param(p_verbose, int,   0644);

MODULE_PARM_DESC(p_find,    "address of kgsl_sharedmem_find");
MODULE_PARM_DESC(p_getdev,  "address of kgsl_get_device");
MODULE_PARM_DESC(p_gpuaddr, "gpuaddr to translate");
MODULE_PARM_DESC(p_devaddr, "optional kgsl_device pointer");

static int __init gc_init(void)
{
    void *(*fn_getdev)(int);
    void *(*fn_find)(void *, unsigned long);
    void *dev = NULL;
    void *entry = NULL;

    pr_info(GC_TAG ": ==== enter ====\n");
    pr_info(GC_TAG ": p_find=0x%lx p_getdev=0x%lx p_gpuaddr=0x%lx\n",
            p_find, p_getdev, p_gpuaddr);

    if (!p_find || !p_getdev) {
        pr_info(GC_TAG ": 缺少参数。用法：\n");
        pr_info(GC_TAG ":   insmod gpu_chams.ko p_find=0x.. p_getdev=0x.. p_gpuaddr=0x..\n");
        pr_info(GC_TAG ": 取地址：\n");
        pr_info(GC_TAG ":   echo 0 > /proc/sys/kernel/kptr_restrict\n");
        pr_info(GC_TAG ":   grep ' kgsl_sharedmem_find$' /proc/kallsyms\n");
        pr_info(GC_TAG ":   grep ' kgsl_get_device$'     /proc/kallsyms\n");
        return -EINVAL;
    }

    /* ① 拿 device 句柄 —— 纯函数指针调用，不传任何结构体 */
    if (p_devaddr) {
        dev = (void *)p_devaddr;
        pr_info(GC_TAG ": 用传入的 dev=%px\n", dev);
    } else {
        fn_getdev = (void *)p_getdev;
        dev = fn_getdev(0);
        pr_info(GC_TAG ": kgsl_get_device(0) -> %px\n", dev);
    }

    if (!dev) {
        pr_err(GC_TAG ": dev 为空，退出\n");
        return -EINVAL;
    }

    /* ② 翻译 gpuaddr -> hostptr */
    if (p_gpuaddr) {
        fn_find = (void *)p_find;
        entry = fn_find(dev, p_gpuaddr);
        pr_info(GC_TAG ": kgsl_sharedmem_find(dev, 0x%lx) -> %px\n",
                p_gpuaddr, entry);
        if (entry) {
            pr_info(GC_TAG ": ✅ 翻译成功！hostptr(entry)=%px\n", entry);
        } else {
            pr_info(GC_TAG ": ❌ 返回 NULL（这个 gpuaddr 不在 kgsl 内存表里）\n");
        }
    } else {
        pr_info(GC_TAG ": 没给 p_gpuaddr，只验证了 dev 获取\n");
    }

    pr_info(GC_TAG ": ==== 探针完成，主动返回 -EINVAL 不驻留 ====\n");
    /* 主动返回错误：insmod 会报错退出，内核自动回滚，不留任何状态 */
    return -EINVAL;
}

static void __exit gc_exit(void)
{
    pr_info(GC_TAG ": bye\n");
}

module_init(gc_init);
module_exit(gc_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("gpu chams probe (no struct, no misc, no kprobe)");