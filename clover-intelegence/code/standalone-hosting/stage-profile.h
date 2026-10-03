#ifndef CLOVER_STAGE_PROFILE_H
#define CLOVER_STAGE_PROFILE_H
#ifdef CLOVER_PROFILE_STAGES
static void profile_boundary(int layer,const char *stage);
static void profile_duration(int layer,const char *stage,double seconds);
static void profile_projection(const unsigned char *weights,double seconds);
#define PROFILE_BOUNDARY(layer,stage) profile_boundary(layer,stage)
#define PROFILE_DURATION(layer,stage,seconds) profile_duration(layer,stage,seconds)
#define PROFILE_PROJECTION(weights,seconds) profile_projection(weights,seconds)
#define PROFILE_CLOCK(name) double name=now_s()
#define PROFILE_RESTART(name) name=now_s()
#else
#define PROFILE_BOUNDARY(layer,stage) ((void)0)
#define PROFILE_DURATION(layer,stage,seconds) ((void)0)
#define PROFILE_PROJECTION(weights,seconds) ((void)0)
#define PROFILE_CLOCK(name) ((void)0)
#define PROFILE_RESTART(name) ((void)0)
#endif
#endif