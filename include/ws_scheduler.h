#ifndef WS_SCHEDULER_H
#define WS_SCHEDULER_H

#include <stdatomic.h>
#include <stdint.h>
#include <omp.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void    (*ws_task_fn)(int64_t i, void *arg);
typedef int64_t (*ws_cost_fn)(int64_t i, void *arg);

#define WS_MAX_WORKERS 16
#define WS_DEQUE_CAP   4096
#define WS_DEQUE_MASK  (WS_DEQUE_CAP - 1)

typedef struct { int64_t begin, end; } ws_range_t;

typedef struct {
    _Atomic int64_t bottom;
    _Atomic int64_t top;
    ws_range_t      buf[WS_DEQUE_CAP];
} ws_deque_t;

typedef struct {
    ws_deque_t      deque;
    _Atomic int64_t remaining;
    int             weight;
    char            _pad[64];
} ws_worker_t;

typedef struct {
    ws_worker_t  workers[WS_MAX_WORKERS];
    int          n_workers;
    int64_t      total_weight;
    int          cpu_weight[8];
    int64_t     *prefix_sum;
    int64_t      iter_base;
    // Epoch-based sync 
    _Atomic int  seq;           // current epoch, read by all threads at entry
    _Atomic int  setup_done;    // set to seq+1 by thread 0 after setup
    _Atomic int  threads_done;  // incremented by each thread when work done
} ws_ctx_t;

void ws_ctx_init(ws_ctx_t *ctx, int n_threads);
void ws_ctx_destroy(ws_ctx_t *ctx);
void ws_for_omp(ws_ctx_t *ctx, int64_t begin, int64_t end,
                ws_task_fn fn, void *arg, ws_cost_fn cost_fn);

#ifdef __cplusplus
}
#endif

#endif // WS_SCHEDULER_H
