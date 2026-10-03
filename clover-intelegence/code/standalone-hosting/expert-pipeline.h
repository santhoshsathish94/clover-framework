#ifndef CLOVER_EXPERT_PIPELINE_H
#define CLOVER_EXPERT_PIPELINE_H
#include <pthread.h>
#include <time.h>

typedef struct { unsigned char *data; size_t capacity,bytes; unsigned expert; Root *root; int ready,valid; } ExpertBuffer;
typedef struct {
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t event;
    ExpertBuffer buffers[2];
    int initialized,stop,pending,slot;
    uint64_t reads,bytes;
    double wait_seconds,read_seconds;
} ExpertPipeline;

static double pipeline_seconds(void)
{
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC,&time);
    return (double)time.tv_sec+time.tv_nsec*1e-9;
}

static void *expert_pipeline_worker(void *argument)
{
    ExpertPipeline *pipeline=argument;
    pthread_mutex_lock(&pipeline->mutex);
    for (;;) {
        while (!pipeline->pending && !pipeline->stop) pthread_cond_wait(&pipeline->event,&pipeline->mutex);
        if (pipeline->stop) break;
        ExpertBuffer *buffer=&pipeline->buffers[pipeline->slot];
        pthread_mutex_unlock(&pipeline->mutex);
        double started=pipeline_seconds();
        int valid=root_read_at(buffer->root->file,buffer->data,buffer->bytes,buffer->root->offsets[buffer->expert*152]);
        double elapsed=pipeline_seconds()-started;
        pthread_mutex_lock(&pipeline->mutex);
        buffer->valid=valid; buffer->ready=1;
        pipeline->reads++; pipeline->bytes+=buffer->bytes; pipeline->read_seconds+=elapsed;
        pipeline->pending=0;
        pthread_cond_broadcast(&pipeline->event);
    }
    pthread_mutex_unlock(&pipeline->mutex);
    return NULL;
}

static int expert_pipeline_open(ExpertPipeline *pipeline,size_t budget)
{
    memset(pipeline,0,sizeof *pipeline);
    if (!budget) return 1;
    size_t capacity=budget/2;
    for (unsigned slot=0;slot<2;slot++) {
        pipeline->buffers[slot].data=malloc(capacity);
        pipeline->buffers[slot].capacity=capacity;
        if (!pipeline->buffers[slot].data) {
            free(pipeline->buffers[0].data); free(pipeline->buffers[1].data);
            memset(pipeline,0,sizeof *pipeline); return 1;
        }
    }
    if (pthread_mutex_init(&pipeline->mutex,NULL)) goto fail;
    if (pthread_cond_init(&pipeline->event,NULL)) { pthread_mutex_destroy(&pipeline->mutex); goto fail; }
    pipeline->initialized=1;
    if (!pthread_create(&pipeline->thread,NULL,expert_pipeline_worker,pipeline)) return 1;
    pipeline->initialized=0;
    pthread_cond_destroy(&pipeline->event); pthread_mutex_destroy(&pipeline->mutex);
fail:
    free(pipeline->buffers[0].data); free(pipeline->buffers[1].data);
    memset(pipeline,0,sizeof *pipeline); return 0;
}

static int expert_pipeline_submit(ExpertPipeline *pipeline,Root *root,unsigned expert,unsigned slot)
{
    if (!pipeline->initialized || expert>=896 || slot>1) return 0;
    pthread_mutex_lock(&pipeline->mutex);
    while (pipeline->pending) pthread_cond_wait(&pipeline->event,&pipeline->mutex);
    ExpertBuffer *buffer=&pipeline->buffers[slot];
    size_t bytes=(size_t)(root->offsets[expert*152+152]-root->offsets[expert*152]);
    if (bytes>buffer->capacity) { pthread_mutex_unlock(&pipeline->mutex); return 0; }
    buffer->root=root; buffer->expert=expert; buffer->bytes=bytes; buffer->ready=0;
    pipeline->slot=(int)slot; pipeline->pending=1;
    pthread_cond_broadcast(&pipeline->event);
    pthread_mutex_unlock(&pipeline->mutex);
    return 1;
}

static int expert_pipeline_acquire(ExpertPipeline *pipeline,Root *root,unsigned expert,unsigned slot)
{
    double started=pipeline_seconds();
    pthread_mutex_lock(&pipeline->mutex);
    ExpertBuffer *buffer=&pipeline->buffers[slot];
    while (!buffer->ready) pthread_cond_wait(&pipeline->event,&pipeline->mutex);
    int valid=buffer->valid && buffer->root==root && buffer->expert==expert;
    if (valid) { root->prefetched=buffer->data; root->prefetched_bytes=buffer->bytes; root->prefetched_expert=expert; }
    pipeline->wait_seconds+=pipeline_seconds()-started;
    pthread_mutex_unlock(&pipeline->mutex);
    return valid;
}

static void expert_pipeline_release(Root *root)
{
    root->prefetched=NULL; root->prefetched_bytes=0;
}

static void expert_pipeline_close(ExpertPipeline *pipeline)
{
    if (!pipeline->initialized) return;
    pthread_mutex_lock(&pipeline->mutex);
    while (pipeline->pending) pthread_cond_wait(&pipeline->event,&pipeline->mutex);
    pipeline->stop=1; pthread_cond_broadcast(&pipeline->event);
    pthread_mutex_unlock(&pipeline->mutex);
    pthread_join(pipeline->thread,NULL);
    pthread_cond_destroy(&pipeline->event); pthread_mutex_destroy(&pipeline->mutex);
    free(pipeline->buffers[0].data); free(pipeline->buffers[1].data);
    memset(pipeline,0,sizeof *pipeline);
}
#endif