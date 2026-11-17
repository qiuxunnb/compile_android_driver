// ============================================================
//  gc_test.c —— 极简测试模块（只 printk / 可选 misc_register）
//
//  stage=0  只有 pr_info（最纯）
//  stage=1  额外做 misc_register，建 /dev/gct 节点
//
//  用途：判断崩在哪一层
//    · stage=0 就崩            -> 设备不允许加载第三方模块（风驰内核/完整性校验）
//    · stage=0 OK、stage=1 崩  -> misc_register 或 /dev 节点创建被拦
//    · 两个都 OK               -> 问题在我们的 kprobe 代码
// ============================================================
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/uaccess.h>

static int stage = 0;
module_param(stage, int, 0644);
MODULE_PARM_DESC(stage, "0=only printk, 1=also misc_register");

static long gct_ioctl(struct file *f, unsigned int c, unsigned long a)
{
    return 0;
}

static const struct file_operations gct_fops = {
    .owner          = THIS_MODULE,
    .unlocked_ioctl = gct_ioctl,
};

static struct miscdevice gct_misc = {
    .minor = MISC_DYNAMIC_MINOR,
    .name  = "gct",
    .fops  = &gct_fops,
    .mode  = 0666,
};

static int __init gct_init(void)
{
    pr_info("GC_TEST: stage=%d enter\n", stage);

    if (stage >= 1) {
        int ret;
        pr_info("GC_TEST: calling misc_register...\n");
        ret = misc_register(&gct_misc);
        pr_info("GC_TEST: misc_register ret=%d\n", ret);
    }

    pr_info("GC_TEST: ===== load OK =====\n");
    return 0;
}

static void __exit gct_exit(void)
{
    if (stage >= 1)
        misc_deregister(&gct_misc);
    pr_info("GC_TEST: ===== bye =====\n");
}

module_init(gct_init);
module_exit(gct_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("gpu chams load-test");