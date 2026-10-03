#define NORMALIZATION_NO_MAIN
#include "normalization.c"
#include <assert.h>

int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    Normalization *model = NULL, *independent = NULL;
    assert(normalization_open(argv[1],&model));
    assert(normalization_open(argv[1],&independent));
    float residual[7168] = {0}, snapshots[8*7168] = {0}, output[7168];
    assert(normalization_process(model,residual,snapshots,8,output));
    for (unsigned coordinate = 0; coordinate < 7168; coordinate++) assert(output[coordinate] == 0.0f);
    for (unsigned coordinate = 0; coordinate < 7168; coordinate++) output[coordinate] = 123.0f;
    assert(!normalization_process(model,residual,snapshots,7,output));
    residual[0] = NAN;
    assert(!normalization_process(model,residual,snapshots,8,output));
    residual[0] = 0;
    snapshots[7168] = INFINITY;
    assert(!normalization_process(model,residual,snapshots,8,output));
    for (unsigned coordinate = 0; coordinate < 7168; coordinate++) assert(output[coordinate] == 123.0f);
    snapshots[7168] = 0;
    assert(!normalization_process(NULL,residual,snapshots,8,output));
    normalization_close(model);
    assert(normalization_process(independent,residual,snapshots,8,output));
    normalization_close(independent);
    assert(!normalization_open(NULL,&model) && !model);
    puts("PASS: actual leaves, zero normalization, independent ownership, count/nonfinite/null rejection and unchanged failed outputs");
    return 0;
}