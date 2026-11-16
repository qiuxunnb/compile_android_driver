// ============================================================
//  gpu_chams.c —— 三角洲 网格体内透（方案B 内核 IB 改写）
//  阶段一：hook kgsl 提交点，关深度测试 -> 穿墙实色内透
//  附带：IB dump 通道（给 04_ib_parse.py 校准用）
// ============================================================
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>
#include <linux/mutex.h>
#include <linux/sched.h>
#include <linux/kprobes.h>
#include <linux/ktime.h>
#include "gpu_chams.h"
#include "adreno_regs.h"

#define GC_TAG "gpuchams"

#ifndef RB_DEPTH_CNTL
#define RB_DEPTH_CNTL   0u
#endif
#ifndef RB_DEPTH_CNTL_TEST_EN
#define RB_DEPTH_CNTL_TEST_EN   0u
#endif
#ifndef RB_DEPTH_CNTL_WRITE_EN
#define RB_DEPTH_CNTL_WRITE_EN  0u
#endif

#define CP_TYPE3_PKT 0x40000000u
#define CP_TYPE4_PKT 0x80000000u
#define CP_TYPE7_PKT 0xC0000000u
#define CP_DUMP_MAGIC 0x49424458u

struct kgsl_command_object {
    __u64 offset;
    __u64 gpuaddr;
    __u64 size;
    __u32 flags;
    __u32 id;
};

struct kgsl_gpu_command {
    __u64 flags;
    __u64 cmdlist;
    __u32 cmdsize;
    __u32 numcmds;
    __u64 objlist;
    __u32 objsize;
    __u32 numobjs;
    __u64 synclist;
    __u32 syncsize;
    __u32 numsyncs;
    __u32 context_id;
    __u32 timestamp;
};

static struct gpu_chams_cfg  g_cfg;
static struct gpu_chams_stat g_stat;
static DEFINE_MUTEX(g_lock);
static struct kprobe g_kp;
static bool g_hooked;

/* ── kallsyms ─────────────────────────────────────────── */
static unsigned long (*g_ksym)(const char *name);
static unsigned long gc_ksym(const char *name)
{
    if (g_ksym) return g_ksym(name);
    return 0;
}

/* ── kgsl 符号 ────────────────────────────────────────── */
static void *(*g_find)(void *dev, u64 gpuaddr);
static void *(*g_getdev)(int minor);
static int g_off_host = -1, g_off_size = -1;

static bool is_kernel_ptr(u64 p)
{
    return (p >> 48) == 0xffff;
}

static bool kva_ok(void *host, u32 size)
{
    u8 tmp[16];
    if (!host || !is_kernel_ptr((u64)(uintptr_t)host)) return false;
    if (size == 0 || size > (512u << 20)) return false;
    if (probe_kernel_read(tmp, host, sizeof(tmp))) return false;
    return true;
}

/* memdesc 里 hostptr/size 位置逐版本不同：自动扫一遍并把偏移固定下来 */
static bool gc_calibrate(struct kgsl_mem_entry_placeholder *e);
struct kgsl_mem_entry_placeholder;

static void *gpuaddr_to_kva(u64 gpuaddr, u32 *out_size)
{
    void *dev, *entry;
    if (!g_find || !g_getdev) return NULL;

    dev = g_getdev(0);
    if (!dev) return NULL;
    entry = g_find(dev, gpuaddr);
    if (!entry) return NULL;

    if (g_off_host >= 0) {
        void *host = *(void **)((char *)entry + g_off_host);
        u32 sz = *(u32 *)((char *)entry + g_off_size);
        if (kva_ok(host, sz)) { if (out_size) *out_size = sz; return host; }
        g_off_host = g_off_size = -1;   /* 失效重扫 */
    }

    /* 自动扫描：找"内核指针 + 紧随/前邻一个合理 size"的窗口 */
    for (int off = 0; off <= 0x200; off += 8) {
        void *host = *(void **)((char *)entry + off);
        u32 cand[2];
        cand[0] = (off >= 8)  ? *(u32 *)((char *)entry + off - 8) : 0;
        cand[1] = *(u32 *)((char *)entry + off + 8);
        for (int k = 0; k < 2; k++) {
            if (!kva_ok(host, cand[k])) continue;
            g_off_host = off;
            g_off_size = (k == 0) ? off - 8 : off + 8;
            if (out_size) *out_size = cand[k];
            pr_info(GC_TAG ": memdesc 校准 host@0x%x size@0x%x\n",
                    g_off_host, g_off_size);
            return host;
        }
    }
    g_stat.fail++;
    return NULL;
}

/* ── IB dump ──────────────────────────────────────────── */
#define DUMP_BYTES (4u << 20)
static u32 *g_dump;
static u32  g_dump_dw, g_dump_cap;
static bool g_dump_on;

static void dump_seg(u64 gpuaddr, const u32 *ib, u32 dwords)
{
    if (!g_dump_on || !g_dump) return;
    if (g_dump_dw + dwords + 4 > g_dump_cap) return;
    g_dump[g_dump_dw++] = CP_DUMP_MAGIC;
    g_dump[g_dump_dw++] = (u32)(gpuaddr & 0xffffffffu);
    g_dump[g_dump_dw++] = (u32)(gpuaddr >> 32);
    g_dump[g_dump_dw++] = dwords;
    memcpy(&g_dump[g_dump_dw], ib, (size_t)dwords * 4);
    g_dump_dw += dwords;
}

/* ── IB 改写：关深度 ──────────────────────────────────── */
static bool is_draw_op(u32 op)
{
    /* 三角洲按 0x35(CP_DRAW_INDX_OFFSET_2) 居多，其余一并兼容 */
    return op == 0x22 || op == 0x34 || op == 0x35 || op == 0x36 || op == 0x2b;
}

static u32 g_filtered;      /* 被过滤掉的 draw 数（调试用） */

/* ── 寄存器直方图：自动找出"每条 draw 前都写一次"的那个寄存器 ── */
#define HIST_MAX 128
static struct { u32 reg, cnt; } g_hist[HIST_MAX];
static u32 g_hist_n, g_draws_frame, g_submits_frame;

static void hist_add(u32 reg)
{
    for (u32 i = 0; i < g_hist_n; i++) {
        if (g_hist[i].reg == reg) { g_hist[i].cnt++; return; }
    }
    if (g_hist_n < HIST_MAX) {
        g_hist[g_hist_n].reg = reg;
        g_hist[g_hist_n].cnt = 1;
        g_hist_n++;
    }
}

static void hist_report(void)
{
    u32 best[6] = {0}, bestc[6] = {0};
    for (u32 i = 0; i < g_hist_n; i++) {
        for (int k = 0; k < 6; k++) {
            if (g_hist[i].cnt > bestc[k]) {
                for (int m = 5; m > k; m--) {
                    bestc[m] = bestc[m - 1];
                    best[m]  = best[m - 1];
                }
                bestc[k] = g_hist[i].cnt;
                best[k]  = g_hist[i].reg;
                break;
            }
        }
    }
    pr_info(GC_TAG ": draws=%u  寄存器Top6:", g_draws_frame);
    for (int k = 0; k < 6; k++) {
        if (best[k]) pr_cont(" 0x%04x x%u", best[k], bestc[k]);
    }
    pr_cont("\n");
    g_hist_n = 0;
    g_draws_frame = 0;
}

/* ── 过滤：决定这条 draw 要不要改 ───────────────────────── */
static bool pass_filter(u32 idx_count)
{
    if (g_cfg.target_mode == 0) return true;          /* 全改 */

    const bool in_range =
        (g_cfg.min_idx == 0 && g_cfg.max_idx == 0) ||
        (idx_count >= g_cfg.min_idx && idx_count <= g_cfg.max_idx);

    return (g_cfg.target_mode == 1) ? in_range : !in_range;
}

/* draw 包里的索引数字段：位置逐档位不同，取不到就当作"不过滤" */
static u32 draw_idx_count(u32 hdr)
{
    const u32 n = (hdr >> 16) & 0x3fffu;
    if (n < 100 || n > 200000) return 0;              /* 明显不是 count */
    return n;
}

static void patch_ib(u32 *ib, u32 dwords)
{
    u32 i = 0;
    int depth_at = -1;

    while (i < dwords) {
        const u32 hdr = ib[i];
        const u32 type = hdr & 0xC0000000u;

        if (type == CP_TYPE3_PKT || type == CP_TYPE7_PKT) {
            const u32 cnt = (hdr >> 16) & 0x3fffu;
            const u32 off = (hdr & 0xffffu) << 2;
            if (i + 1 + cnt > dwords) break;

            for (u32 k = 0; k < cnt; k++) {
                const u32 rg = off + k * 4;
                hist_add(rg);
                if (RB_DEPTH_CNTL && rg == RB_DEPTH_CNTL)
                    depth_at = (int)(i + 1 + k);
            }
            i += 1 + cnt + ((type == CP_TYPE7_PKT) ? 2u : 0u);
            continue;
        }
        if (type == CP_TYPE4_PKT) { i += 2; continue; }

        const u32 op = hdr & 0xffu;
        if (op) g_stat.last_opcode = op;

        if (is_draw_op(op)) {
            g_stat.draws_seen++;
            g_draws_frame++;

            const u32 idx_count = draw_idx_count(hdr);
            if (idx_count && !pass_filter(idx_count)) {
                g_filtered++;
                g_stat.skip_cached = g_filtered;
                depth_at = -1;
                i += 5;
                continue;
            }

            if (g_cfg.depth_off && depth_at > 0 && (u32)depth_at < dwords) {
                u32 v = ib[depth_at];
                if (v & (RB_DEPTH_CNTL_TEST_EN | RB_DEPTH_CNTL_WRITE_EN)) {
                    ib[depth_at] = v & ~(RB_DEPTH_CNTL_TEST_EN |
                                         RB_DEPTH_CNTL_WRITE_EN);
                    g_stat.depth_patched++;
                }
            }
            depth_at = -1;
            i += 5;
            continue;
        }
        i++;
    }
}

/* ── hook 主体 ────────────────────────────────────────── */
static int gc_pre(struct kprobe *p, struct pt_regs *regs)
{
    void __user *udata;
    struct kgsl_gpu_command cmd;
    struct kgsl_command_object *objs;

    if (!g_cfg.enable) return 0;
    if (g_cfg.tgid && current->tgid != g_cfg.tgid) return 0;

    udata = (void __user *)(uintptr_t)regs->regs[2];   /* x2 = data */
    if (!udata) return 0;
    if (copy_from_user(&cmd, udata, sizeof(cmd))) return 0;
    if (g_cfg.ctx_filter && cmd.context_id != g_cfg.ctx_filter) return 0;
    if (!cmd.numcmds || !cmd.cmdlist || cmd.numcmds > 256) return 0;

    objs = kvmalloc((size_t)cmd.numcmds * sizeof(*objs), GFP_KERNEL);
    if (!objs) return 0;
    if (copy_from_user(objs, (void __user *)(uintptr_t)cmd.cmdlist,
                       (size_t)cmd.numcmds * sizeof(*objs)))
        goto out;

    mutex_lock(&g_lock);
    g_stat.submits++;

    for (u32 n = 0; n < cmd.numcmds; n++) {
        u32 isz = 0, dwords;
        u32 *ib;

        ib = gpuaddr_to_kva(objs[n].gpuaddr, &isz);
        if (!ib) { g_stat.fail++; continue; }

        dwords = (u32)(objs[n].size / 4);
        if (dwords > isz / 4) dwords = isz / 4;
        if (dwords < 4) continue;

        g_stat.ib_scanned++;
        dump_seg(objs[n].gpuaddr, ib, dwords);
        patch_ib(ib, dwords);
    }
    mutex_unlock(&g_lock);

    if (++g_submits_frame % 120 == 0) hist_report();
out:
    kvfree(objs);
    return 0;
}

/* ── 设备节点 ─────────────────────────────────────────── */
static long gc_ioctl(struct file *f, unsigned int c, unsigned long a)
{
    switch (c) {
    case GPUCHAMS_SET_CFG: {
        struct gpu_chams_cfg t;
        if (copy_from_user(&t, (void __user *)a, sizeof(t))) return -EFAULT;
        mutex_lock(&g_lock);
        g_cfg = t;
        pr_info(GC_TAG ": cfg enable=%u mode=%u depth=%u tmode=%d tgid=%d idx[%u,%u]\n",
                t.enable, t.mode, t.depth_off, (int)t.target_mode, t.tgid,
                t.min_idx, t.max_idx);
        mutex_unlock(&g_lock);
        return 0;
    }
    case GPUCHAMS_GET_STAT:
        return copy_to_user((void __user *)a, &g_stat, sizeof(g_stat)) ? -EFAULT : 0;
    case GPUCHAMS_DUMP_CTRL: {
        struct gpu_chams_dump d;
        if (copy_from_user(&d, (void __user *)a, sizeof(d))) return -EFAULT;
        mutex_lock(&g_lock);
        g_dump_on = d.enable ? true : false;
        if (d.enable) g_dump_dw = 0;
        d.segments = 0;
        d.dwords = g_dump_dw;
        mutex_unlock(&g_lock);
        return copy_to_user((void __user *)a, &d, sizeof(d)) ? -EFAULT : 0;
    }
    case GPUCHAMS_RESET:
        mutex_lock(&g_lock);
        memset(&g_stat, 0, sizeof(g_stat));
        g_dump_dw = 0;
        mutex_unlock(&g_lock);
        return 0;
    }
    return -EINVAL;
}

static ssize_t gc_read(struct file *f, char __user *buf, size_t n, loff_t *off)
{
    size_t left, take;
    if (!g_dump) return 0;
    if (*off < 0) return -EINVAL;
    left = (size_t)g_dump_dw * 4;
    if ((size_t)*off >= left) return 0;
    take = left - (size_t)*off;
    if (take > n) take = n;
    if (copy_to_user(buf, (char *)g_dump + *off, take)) return -EFAULT;
    *off += take;
    return take;
}

static const struct file_operations gc_fops = {
    .owner = THIS_MODULE,
    .unlocked_ioctl = gc_ioctl,
    .read = gc_read,
};

static struct miscdevice gc_misc = {
    .minor = MISC_DYNAMIC_MINOR,
    .name  = GPUCHAMS_DEV_NAME,
    .fops  = &gc_fops,
    .mode  = 0666,
};

static int __init gc_init(void)
{
    struct kprobe kp = { .symbol_name = "kallsyms_lookup_name" };

    if (register_kprobe(&kp) == 0) {
        g_ksym = (void *)kp.addr;
        unregister_kprobe(&kp);
    }
    if (!g_ksym) { pr_err(GC_TAG ": 找不到 kallsyms_lookup_name\n"); return -ENOENT; }

    g_find   = (void *)gc_ksym("kgsl_sharedmem_find");
    g_getdev = (void *)gc_ksym("kgsl_get_device");
    if (!g_find || !g_getdev) {
        pr_err(GC_TAG ": kgsl 符号缺失 find=%px getdev=%px\n", g_find, g_getdev);
        return -ENOENT;
    }

    g_dump_cap = DUMP_BYTES / 4;
    g_dump = kvmalloc(DUMP_BYTES, GFP_KERNEL | __GFP_ZERO);
    if (!g_dump) return -ENOMEM;

    if (misc_register(&gc_misc)) {
        pr_err(GC_TAG ": /dev 节点注册失败\n");
        kvfree(g_dump); g_dump = NULL;
        return -ENODEV;
    }

    g_kp.symbol_name = "kgsl_ioctl_gpu_command";
    g_kp.pre_handler = gc_pre;
    if (register_kprobe(&g_kp)) {
        g_kp.symbol_name = "adreno_ringbuffer_submitcmd";
        if (register_kprobe(&g_kp)) {
            pr_err(GC_TAG ": hook 装不上（两个提交点都失败）\n");
            misc_deregister(&gc_misc);
            kvfree(g_dump); g_dump = NULL;
            return -ENOENT;
        }
    }
    g_hooked = true;

    pr_info(GC_TAG ": 就绪，hook=%s dev=/dev/%s\n",
            g_kp.symbol_name, gc_misc.name);
    return 0;
}

static void __exit gc_exit(void)
{
    if (g_hooked) unregister_kprobe(&g_kp);
    misc_deregister(&gc_misc);
    if (g_dump) kvfree(g_dump);
    pr_info(GC_TAG ": 卸载\n");
}

module_init(gc_init);
module_exit(gc_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("gpu chams (IB patch)");
