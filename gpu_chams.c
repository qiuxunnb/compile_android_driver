// ============================================================
//  gpu_chams.c —— 三角洲 网格体内透（安全版 / 无睡眠 kprobe）
//
//  上一版崩溃根因：kprobe pre_handler 里做了 4 件会睡眠的事
//    ① kvmalloc(GFP_KERNEL)   ② copy_from_user
//    ③ mutex_lock             ④ kgsl_sharedmem_find()（内部有锁）
//  本版把 ①②③④ 全部从 kprobe 里搬走：
//    · kprobe 内只做 copy_from_user_nofault + 查缓存表 + 纯内存写
//    · gpuaddr → hostptr 的翻译放到 ioctl(GPUCHAMS_FLUSH)，进程上下文
//
//  默认 enable=0：装载后不碰任何 IB，先确认不崩再开
// ============================================================
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include <linux/slab.h>
#include <linux/mutex.h>
#include <linux/spinlock.h>
#include <linux/sched.h>
#include <linux/kprobes.h>
#include <linux/string.h>
#include <linux/errno.h>
#include <linux/atomic.h>
#include "gpu_chams.h"
#include "adreno_regs.h"

#define GC_TAG "gpuchams"

#ifndef RB_DEPTH_CNTL
#define RB_DEPTH_CNTL            0u
#endif
#ifndef RB_DEPTH_CNTL_TEST_EN
#define RB_DEPTH_CNTL_TEST_EN    0u
#endif
#ifndef RB_DEPTH_CNTL_WRITE_EN
#define RB_DEPTH_CNTL_WRITE_EN   0u
#endif

#define CP_TYPE3_PKT 0x40000000u
#define CP_TYPE4_PKT 0x80000000u
#define CP_TYPE7_PKT 0xC0000000u

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
static DEFINE_MUTEX(g_ioctl_lock);

static unsigned long (*g_ksym)(const char *name);
static void *(*g_find)(void *dev, u64 gpuaddr);
static void *(*g_getdev)(int minor);

/* ── gpuaddr -> hostptr 缓存（kprobe 内只读）── */
#define CACHE_MAX 512
struct kv_ent {
    u64  gpuaddr;
    void *host;
    u32  size;
    u32  used;
};
static struct kv_ent g_cache[CACHE_MAX];
static DEFINE_SPINLOCK(g_cache_lock);
static u32 g_cache_used;

/* ── 未命中 ring ── */
#define RING_SZ 512
static u64 g_ring[RING_SZ];
static atomic_t g_ring_w = ATOMIC_INIT(0);

/* ── 寄存器直方图 ── */
#define HIST_MAX 64
static u32 g_hist_reg[HIST_MAX];
static u32 g_hist_cnt[HIST_MAX];
static u32 g_hist_n;
static u32 g_draws_period;

/* ── memdesc 偏移（首次 FLUSH 校准）── */
static int g_off_host = -1;
static int g_off_size = -1;

static bool is_kernel_ptr(u64 p)
{
    return (p >> 48) == 0xffff;
}

static void *cache_lookup(u64 gpuaddr, u32 *size)
{
    void *ret = NULL;
    unsigned long flags;
    u32 i;

    spin_lock_irqsave(&g_cache_lock, flags);
    for (i = 0; i < g_cache_used; i++) {
        if (g_cache[i].used && g_cache[i].gpuaddr == gpuaddr) {
            ret = g_cache[i].host;
            if (size) *size = g_cache[i].size;
            break;
        }
    }
    spin_unlock_irqrestore(&g_cache_lock, flags);
    return ret;
}

static void cache_store(u64 gpuaddr, void *host, u32 size)
{
    unsigned long flags;
    u32 i;

    spin_lock_irqsave(&g_cache_lock, flags);
    for (i = 0; i < g_cache_used; i++) {
        if (g_cache[i].gpuaddr == gpuaddr) {
            g_cache[i].host = host;
            g_cache[i].size = size;
            spin_unlock_irqrestore(&g_cache_lock, flags);
            return;
        }
    }
    if (g_cache_used < CACHE_MAX) {
        g_cache[g_cache_used].gpuaddr = gpuaddr;
        g_cache[g_cache_used].host = host;
        g_cache[g_cache_used].size = size;
        g_cache[g_cache_used].used = 1;
        g_cache_used++;
    }
    spin_unlock_irqrestore(&g_cache_lock, flags);
}

static void hist_add(u32 reg)
{
    u32 i;
    if (g_hist_n >= HIST_MAX) return;
    for (i = 0; i < g_hist_n; i++) {
        if (g_hist_reg[i] == reg) { g_hist_cnt[i]++; return; }
    }
    g_hist_reg[g_hist_n] = reg;
    g_hist_cnt[g_hist_n] = 1;
    g_hist_n++;
}

static bool pass_filter(u32 idx_count)
{
    bool in;
    if (g_cfg.target_mode == 0) return true;
    if (g_cfg.min_idx == 0 && g_cfg.max_idx == 0) return true;
    in = (idx_count >= g_cfg.min_idx && idx_count <= g_cfg.max_idx);
    return (g_cfg.target_mode == 1) ? in : !in;
}

static u32 draw_idx_count(u32 hdr)
{
    u32 n = (hdr >> 16) & 0x3fffu;
    if (n < 100 || n > 200000) return 0;
    return n;
}

static bool is_draw_op(u32 op)
{
    return op == 0x22 || op == 0x34 || op == 0x35 || op == 0x36 || op == 0x2b;
}

/* 纯内存操作 —— 可在 kprobe 上下文安全执行 */
static void patch_ib(u32 *ib, u32 dwords)
{
    u32 i = 0;
    int depth_at = -1;
    const u32 mask = RB_DEPTH_CNTL_TEST_EN | RB_DEPTH_CNTL_WRITE_EN;

    while (i < dwords) {
        u32 hdr  = ib[i];
        u32 type = hdr & 0xC0000000u;

        if (type == CP_TYPE3_PKT || type == CP_TYPE7_PKT) {
            u32 cnt = (hdr >> 16) & 0x3fffu;
            u32 off = (hdr & 0xffffu) << 2;
            u32 k;

            if (i + 1 + cnt > dwords) break;
            for (k = 0; k < cnt; k++) {
                u32 rg = off + k * 4;
                hist_add(rg);
                if (RB_DEPTH_CNTL && rg == RB_DEPTH_CNTL)
                    depth_at = (int)(i + 1 + k);
            }
            i += 1 + cnt + ((type == CP_TYPE7_PKT) ? 2u : 0u);
            continue;
        }
        if (type == CP_TYPE4_PKT) { i += 2; continue; }

        {
            u32 op = hdr & 0xffu;
            if (op) g_stat.last_opcode = op;

            if (is_draw_op(op)) {
                u32 ic = draw_idx_count(hdr);
                g_stat.draws_seen++;
                g_draws_period++;

                if (ic && !pass_filter(ic)) {
                    g_stat.filtered++;
                    depth_at = -1;
                    i += 5;
                    continue;
                }
                if (g_cfg.depth_off && depth_at > 0 && (u32)depth_at < dwords) {
                    u32 v = ib[depth_at];
                    if (mask && (v & mask)) {
                        ib[depth_at] = v & ~mask;
                        g_stat.depth_patched++;
                    }
                }
                depth_at = -1;
                i += 5;
                continue;
            }
        }
        i++;
    }
}

/* ══ kprobe：只 nofault 拷贝 + 查表 + 写内存，全程不睡眠 ══ */
static int gc_pre(struct kprobe *p, struct pt_regs *regs)
{
    void __user *udata;
    struct kgsl_gpu_command cmd;
    struct kgsl_command_object objs[32];
    u32 num, n;
    size_t bytes;

    if (!READ_ONCE(g_cfg.enable)) return 0;
    if (READ_ONCE(g_cfg.tgid) && current->tgid != READ_ONCE(g_cfg.tgid)) return 0;

    udata = (void __user *)(uintptr_t)regs->regs[2];
    if (!udata) return 0;

    if (copy_from_user_nofault(&cmd, udata, sizeof(cmd))) return 0;
    if (!cmd.numcmds || !cmd.cmdlist) return 0;
    if (g_cfg.ctx_filter && cmd.context_id != g_cfg.ctx_filter) return 0;

    num = cmd.numcmds;
    if (num > 32) num = 32;
    bytes = (size_t)num * sizeof(*objs);
    if (copy_from_user_nofault(objs, (void __user *)(uintptr_t)cmd.cmdlist, bytes))
        return 0;

    g_stat.submits++;

    for (n = 0; n < num; n++) {
        u32 sz = 0, dwords;
        u32 *ib = cache_lookup(objs[n].gpuaddr, &sz);

        if (!ib) {
            u32 w = atomic_inc_return(&g_ring_w) - 1;
            g_ring[w & (RING_SZ - 1)] = objs[n].gpuaddr;
            g_stat.cache_miss++;
            continue;
        }
        g_stat.cache_hit++;

        dwords = (u32)(objs[n].size / 4);
        if (dwords > sz / 4) dwords = sz / 4;
        if (dwords < 4) continue;

        patch_ib(ib, dwords);
    }
    return 0;
}

/* ══ FLUSH：进程上下文，可睡眠、可调 kgsl 内部函数 ══ */
static int gc_flush(void)
{
    void *dev;
    u32 w, begin, i, resolved = 0;

    if (!g_find || !g_getdev) return -ENOENT;
    dev = g_getdev(0);
    if (!dev) return -ENODEV;

    w = (u32)atomic_read(&g_ring_w);
    begin = (w > RING_SZ) ? (w - RING_SZ) : 0;

    for (i = begin; i < w; i++) {
        u64 ga = g_ring[i & (RING_SZ - 1)];
        char *base;
        void *host = NULL;
        u32 size = 0;

        if (!ga) continue;
        base = (char *)g_find(dev, ga);
        if (!base) continue;

        if (g_off_host >= 0) {
            host = *(void **)(base + g_off_host);
            size = *(u32 *)(base + g_off_size);
            if (!host || !is_kernel_ptr((u64)(uintptr_t)host)) {
                host = NULL;
                g_off_host = g_off_size = -1;
            }
        }
        if (!host) {
            int off;
            for (off = 0; off <= 0x200; off += 8) {
                void *h = *(void **)(base + off);
                u32 cand[2], k;
                u8 probe[8] = {0};

                cand[0] = (off >= 8) ? *(u32 *)(base + off - 8) : 0;
                cand[1] = *(u32 *)(base + off + 8);

                if (!h || !is_kernel_ptr((u64)(uintptr_t)h)) continue;

                for (k = 0; k < 2; k++) {
                    if (cand[k] == 0 || cand[k] > (256u << 20)) continue;
                    if (copy_from_kernel_nofault(probe, h, sizeof(probe)) != 0) continue;
                    host = h;
                    size = cand[k];
                    g_off_host = off;
                    g_off_size = (k == 0) ? (off - 8) : (off + 8);
                    pr_info(GC_TAG ": memdesc 校准 host@0x%x size@0x%x\n",
                            g_off_host, g_off_size);
                    break;
                }
                if (host) break;
            }
        }
        if (!host || !size) continue;

        cache_store(ga, host, size);
        resolved++;
    }
    g_stat.cache_used = g_cache_used;

    if (g_draws_period >= 60) {
        u32 k;
        pr_info(GC_TAG ": draws=%u 寄存器Top8:", g_draws_period);
        for (k = 0; k < g_hist_n && k < 8; k++)
            pr_cont(" 0x%04x x%u", g_hist_reg[k], g_hist_cnt[k]);
        pr_cont("\n");
        g_hist_n = 0;
        g_draws_period = 0;
    }
    return (int)resolved;
}

/* ── 设备节点 ── */
static const struct file_operations gc_fops = {
    .owner          = THIS_MODULE,
    .unlocked_ioctl = NULL,
};

static long gc_ioctl(struct file *f, unsigned int c, unsigned long a)
{
    long r = 0;

    mutex_lock(&g_ioctl_lock);
    switch (c) {
    case GPUCHAMS_SET_CFG: {
        struct gpu_chams_cfg t;
        if (copy_from_user(&t, (void __user *)a, sizeof(t))) { r = -EFAULT; break; }
        memcpy(&g_cfg, &t, sizeof(g_cfg));
        pr_info(GC_TAG ": cfg enable=%u mode=%u depth=%u tmode=%d tgid=%d idx[%u,%u]\n",
                g_cfg.enable, g_cfg.mode, g_cfg.depth_off, (int)g_cfg.target_mode,
                g_cfg.tgid, g_cfg.min_idx, g_cfg.max_idx);
        break;
    }
    case GPUCHAMS_GET_STAT:
        if (copy_to_user((void __user *)a, &g_stat, sizeof(g_stat))) r = -EFAULT;
        break;
    case GPUCHAMS_FLUSH: {
        int res = gc_flush();
        g_stat.cache_used = g_cache_used;
        if (res < 0) r = res;
        if (copy_to_user((void __user *)a, &g_stat, sizeof(g_stat))) r = -EFAULT;
        break;
    }
    case GPUCHAMS_GET_HIST: {
        struct gpu_chams_hist h;
        u32 k;
        memset(&h, 0, sizeof(h));
        h.draws = g_draws_period;
        h.n = (g_hist_n > 8) ? 8 : g_hist_n;
        for (k = 0; k < h.n; k++) {
            h.reg[k] = g_hist_reg[k];
            h.cnt[k] = g_hist_cnt[k];
        }
        if (copy_to_user((void __user *)a, &h, sizeof(h))) r = -EFAULT;
        break;
    }
    case GPUCHAMS_RESET:
        memset(&g_stat, 0, sizeof(g_stat));
        g_hist_n = 0;
        g_draws_period = 0;
        break;
    default:
        r = -EINVAL;
        break;
    }
    mutex_unlock(&g_ioctl_lock);
    return r;
}

static const struct file_operations gcu_fops = {
    .owner          = THIS_MODULE,
    .unlocked_ioctl = gc_ioctl,
};

static struct miscdevice gc_misc = {
    .minor = MISC_DYNAMIC_MINOR,
    .name  = GPUCHAMS_DEV_NAME,
    .fops  = &gcu_fops,
    .mode  = 0666,
};

static struct kprobe g_kp;
static bool g_hooked;

static int __init gc_init(void)
{
    struct kprobe kp = { .symbol_name = "kallsyms_lookup_name" };

    if (register_kprobe(&kp) == 0) {
        g_ksym = (void *)kp.addr;
        unregister_kprobe(&kp);
    }
    if (!g_ksym) {
        pr_err(GC_TAG ": 找不到 kallsyms_lookup_name\n");
        return -ENOENT;
    }

    g_find   = (void *)g_ksym("kgsl_sharedmem_find");
    g_getdev = (void *)g_ksym("kgsl_get_device");
    if (!g_find || !g_getdev) {
        pr_err(GC_TAG ": kgsl 符号缺失 find=%px getdev=%px\n", g_find, g_getdev);
        return -ENOENT;
    }

    if (misc_register(&gc_misc)) {
        pr_err(GC_TAG ": /dev 节点注册失败\n");
        return -ENODEV;
    }

    g_kp.symbol_name = "kgsl_ioctl_gpu_command";
    g_kp.pre_handler = gc_pre;
    if (register_kprobe(&g_kp)) {
        g_kp.symbol_name = "adreno_ringbuffer_submitcmd";
        if (register_kprobe(&g_kp)) {
            pr_err(GC_TAG ": hook 装不上\n");
            misc_deregister(&gc_misc);
            return -ENOENT;
        }
    }
    g_hooked = true;

    pr_info(GC_TAG ": 就绪（默认 enable=0，不动手）hook=%s dev=/dev/%s\n",
            g_kp.symbol_name, gc_misc.name);
    return 0;
}

static void __exit gc_exit(void)
{
    WRITE_ONCE(g_cfg.enable, 0);
    if (g_hooked) unregister_kprobe(&g_kp);
    misc_deregister(&gc_misc);
    pr_info(GC_TAG ": 卸载\n");
}

module_init(gc_init);
module_exit(gc_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("gpu chams (safe IB patch)");