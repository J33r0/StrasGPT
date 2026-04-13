#define _GNU_SOURCE
#include "ws_scheduler.h"
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <sched.h>
#include <omp.h>

// Chase-Lev deque
static inline void deque_init(ws_deque_t *d) {
    atomic_store_explicit(&d->bottom, 0, memory_order_relaxed);
    atomic_store_explicit(&d->top,    0, memory_order_relaxed);
}
static inline void deque_push(ws_deque_t *d, ws_range_t r) {
    int64_t b = atomic_load_explicit(&d->bottom, memory_order_relaxed);
    d->buf[b & WS_DEQUE_MASK] = r;
    atomic_store_explicit(&d->bottom, b + 1, memory_order_release);
}
static inline bool deque_pop(ws_deque_t *d, ws_range_t *out) {
    int64_t b = atomic_load_explicit(&d->bottom, memory_order_relaxed) - 1;
    atomic_store_explicit(&d->bottom, b, memory_order_seq_cst);
    int64_t t = atomic_load_explicit(&d->top, memory_order_seq_cst);
    if (t > b) { atomic_store_explicit(&d->bottom, b+1, memory_order_relaxed); return false; }
    *out = d->buf[b & WS_DEQUE_MASK];
    if (t == b) {
        if (!atomic_compare_exchange_strong_explicit(&d->top, &t, t+1,
                memory_order_seq_cst, memory_order_relaxed)) {
            atomic_store_explicit(&d->bottom, b+1, memory_order_relaxed);
            return false;
        }
        atomic_store_explicit(&d->bottom, b+1, memory_order_relaxed);
    }
    return true;
}
static inline bool deque_steal(ws_deque_t *d, ws_range_t *out) {
    int64_t t = atomic_load_explicit(&d->top,    memory_order_acquire);
    int64_t b = atomic_load_explicit(&d->bottom, memory_order_acquire);
    if (t >= b) return false;
    *out = d->buf[t & WS_DEQUE_MASK];
    return atomic_compare_exchange_strong_explicit(&d->top, &t, t+1,
        memory_order_seq_cst, memory_order_relaxed);
}

// Core weight
static int get_core_weight(int cpu_id) {
    char path[128];
    snprintf(path, sizeof(path),
        "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpu_id);
    FILE *f = fopen(path, "r");
    if (!f) return 2;
    long freq = 0;
    if (fscanf(f, "%ld", &freq) != 1) freq = 0;
    fclose(f);
    if (freq >= 3000000) return 4;
    if (freq >= 2500000) return 2;
    return 1;
}

void ws_ctx_init(ws_ctx_t *ctx, int n_threads) {
    if (n_threads > WS_MAX_WORKERS) n_threads = WS_MAX_WORKERS;
    memset(ctx, 0, sizeof(*ctx));
    ctx->n_workers = n_threads;
    for (int i = 0; i < 8; i++) ctx->cpu_weight[i] = get_core_weight(i);
    ctx->total_weight = 0;
    for (int i = 0; i < n_threads; i++) {
        deque_init(&ctx->workers[i].deque);
        atomic_store(&ctx->workers[i].remaining, 0);
        ctx->workers[i].weight = ctx->cpu_weight[i < 8 ? i : 7];
        ctx->total_weight += ctx->workers[i].weight;
    }
    // seq=0 means "no call in progress"; setup_done starts at 0 too
    atomic_store(&ctx->seq,         0);
    atomic_store(&ctx->setup_done,  0);
    atomic_store(&ctx->threads_done,0);
}

void ws_ctx_destroy(ws_ctx_t *ctx) { (void)ctx; }

// Victim selection
static int find_victim(ws_ctx_t *ctx, int thief_id, bool cost_aware, uint64_t *rng) {
    int n = ctx->n_workers;
    if (n <= 1) return -1;
    uint64_t s = *rng; s ^= s<<13; s ^= s>>7; s ^= s<<17; *rng = s;
    if (!cost_aware) {
        int v = (int)(s % (uint64_t)(n-1));
        if (v >= thief_id) v++;
        return v;
    }
    int best = -1; int64_t best_work = 4;
    for (int i = 0; i < n; i++) {
        if (i == thief_id) continue;
        int64_t w = atomic_load_explicit(&ctx->workers[i].remaining, memory_order_relaxed);
        if (w > best_work) { best_work = w; best = i; }
    }
    return best;
}

#define WS_YIELD() do { \
    __asm__ volatile("" ::: "memory"); \
    sched_yield(); \
} while(0)

void ws_for_omp(ws_ctx_t *ctx, int64_t begin, int64_t end,
                ws_task_fn fn, void *arg, ws_cost_fn cost_fn)
{
    if (begin >= end) return;

    int     tid = omp_get_thread_num();
    int64_t N   = end - begin;
    int     W   = omp_get_num_threads();

    // Each thread reads the current sequence number FIRST, before doing
    // anything else. This is the epoch for this call.
    int my_seq = atomic_load_explicit(&ctx->seq, memory_order_acquire);

    // Update weight for this thread's actual CPU 
    int cpu = sched_getcpu();
    if (cpu < 0 || cpu >= 8) cpu = tid < 8 ? tid : 0;
    ctx->workers[tid].weight = ctx->cpu_weight[cpu];

    if (tid == 0) {
        // Assign actual number of active workers dynamically to support Android OpenMP thread caps
        ctx->n_workers = W;

        // Setup 
        ctx->total_weight = 0;
        for (int w = 0; w < W; w++) ctx->total_weight += ctx->workers[w].weight;
        if (ctx->total_weight <= 0) ctx->total_weight = W;

        ctx->prefix_sum = NULL;
        if (cost_fn) {
            ctx->prefix_sum = (int64_t *)malloc((N+1) * sizeof(int64_t));
            ctx->prefix_sum[0] = 0;
            for (int64_t i = 0; i < N; i++)
                ctx->prefix_sum[i+1] = ctx->prefix_sum[i] + cost_fn(begin+i, arg);
        }
        ctx->iter_base = begin;

        int64_t cursor = begin;
        for (int w = 0; w < W; w++) {
            ws_worker_t *wk = &ctx->workers[w];
            int64_t chunk;
            if (w == W - 1) {
                chunk = end - cursor;
            } else {
                chunk = (int64_t)((N * wk->weight) / ctx->total_weight);
                if (chunk < 1 && cursor < end) chunk = 1;
                if (cursor + chunk > end) chunk = end - cursor;
            }
            if (chunk < 0) chunk = 0;
            atomic_store_explicit(&wk->remaining, chunk, memory_order_relaxed);
            atomic_store_explicit(&wk->deque.bottom, 0, memory_order_relaxed);
            atomic_store_explicit(&wk->deque.top,    0, memory_order_relaxed);
            if (chunk > 0) deque_push(&wk->deque, (ws_range_t){cursor, cursor+chunk});
            cursor += chunk;
        }
        // Release: all setup writes are visible after this
        atomic_store_explicit(&ctx->setup_done, my_seq + 1, memory_order_release);
    } else {
        // Wait for thread 0 to finish setup for this epoch (my_seq+1)
        while (atomic_load_explicit(&ctx->setup_done, memory_order_acquire)
               != my_seq + 1)
            WS_YIELD();
    }

    // Work 
    ws_worker_t *self = &ctx->workers[tid];
    bool cost_aware   = (ctx->prefix_sum != NULL);
    uint64_t rng      = (uint64_t)(uintptr_t)self ^ ((uint64_t)tid * 2654435761ULL);

    ws_range_t r;
    while (deque_pop(&self->deque, &r)) {
        for (int64_t i = r.begin; i < r.end; i++) fn(i, arg);
        atomic_fetch_sub_explicit(&self->remaining, r.end - r.begin, memory_order_relaxed);
    }

    // Work stealing loop
    for (;;) {
        int vid = find_victim(ctx, tid, cost_aware, &rng);
        if (vid < 0) break;
        ws_worker_t *v = &ctx->workers[vid];
        if (!deque_steal(&v->deque, &r)) {
            bool any = false;
            for (int i = 0; i < W; i++)
                if (atomic_load_explicit(&ctx->workers[i].remaining, memory_order_relaxed) > 0)
                    { any = true; break; }
            if (!any) break;
            WS_YIELD();
            continue;
        }
        for (int64_t i = r.begin; i < r.end; i++) fn(i, arg);
        atomic_fetch_sub_explicit(&v->remaining, r.end - r.begin, memory_order_relaxed);
        while (deque_pop(&self->deque, &r)) {
            for (int64_t i = r.begin; i < r.end; i++) fn(i, arg);
            atomic_fetch_sub_explicit(&self->remaining, r.end - r.begin, memory_order_relaxed);
        }
        bool any = false;
        for (int i = 0; i < W; i++)
            if (atomic_load_explicit(&ctx->workers[i].remaining, memory_order_relaxed) > 0)
                { any = true; break; }
        if (!any) break;
    }

    // All threads signal done, then wait for thread 0 to advance seq
    int done = atomic_fetch_add_explicit(&ctx->threads_done, 1, memory_order_acq_rel) + 1;

    if (tid == 0) {
        // Wait for all W threads
        while (atomic_load_explicit(&ctx->threads_done, memory_order_acquire) < W)
            WS_YIELD();
        if (ctx->prefix_sum) { free(ctx->prefix_sum); ctx->prefix_sum = NULL; }
        atomic_store_explicit(&ctx->threads_done, 0, memory_order_relaxed);
        // Advance seq — this is the signal non-zero threads wait on
        atomic_store_explicit(&ctx->seq, my_seq + 1, memory_order_release);
    } else {
        (void)done;
        // Wait for thread 0 to finish cleanup and advance seq
        while (atomic_load_explicit(&ctx->seq, memory_order_acquire) != my_seq + 1)
            WS_YIELD();
    }
}
