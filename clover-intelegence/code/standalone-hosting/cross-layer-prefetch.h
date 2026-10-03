#ifndef CLOVER_CROSS_LAYER_PREFETCH_H
#define CLOVER_CROSS_LAYER_PREFETCH_H

typedef struct {
    Root *root;
    unsigned expert;
    int armed;
    uint64_t submitted,hits,misses;
    double prediction_seconds;
} CrossLayerPrefetch;

static int cross_layer_submit(CrossLayerPrefetch *ahead,ExpertPipeline *pipeline,Root *root,unsigned expert)
{
    if (ahead->armed) return 0;
    if (!pipeline->initialized) return 1;
    if (!expert_pipeline_submit(pipeline,root,expert,0)) return 0;
    ahead->root=root; ahead->expert=expert; ahead->armed=1; ahead->submitted++;
    return 1;
}

static int cross_layer_match(CrossLayerPrefetch *ahead,ExpertPipeline *pipeline,Root *root,const int *selected,unsigned *missing,unsigned count)
{
    if (!ahead->armed) return 0;
    ahead->armed=0;
    if (ahead->root==root) {
        for (unsigned index=0;index<count;index++) {
            if ((unsigned)selected[missing[index]]==ahead->expert) {
                unsigned rank=missing[index];
                missing[index]=missing[0]; missing[0]=rank;
                ahead->root=NULL; ahead->hits++;
                return 1;
            }
        }
    }
    double started=pipeline_seconds();
    pthread_mutex_lock(&pipeline->mutex);
    while (pipeline->pending) pthread_cond_wait(&pipeline->event,&pipeline->mutex);
    pipeline->wait_seconds+=pipeline_seconds()-started;
    pthread_mutex_unlock(&pipeline->mutex);
    ahead->root=NULL; ahead->misses++;
    return 0;
}

#endif