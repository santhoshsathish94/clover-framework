#ifndef K3_ROOT_FUNCTION_H
#define K3_ROOT_FUNCTION_H

typedef void (*RootObserver)(void *context, const char *stage,
                             float *const *values, int positions, int width);

void root(int layer, int expert, int positions, const float *const *input,
          float *const *output, RootObserver observer, void *context);

#endif