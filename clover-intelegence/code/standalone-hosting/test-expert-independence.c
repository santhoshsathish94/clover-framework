#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "live-root.h"
#include <assert.h>

enum { REC=51204, IN=3584, HID=3072 };

static unsigned char *slurp(const char *path,size_t *bytes)
{
    struct stat info;
    FILE *file=fopen(path,"rb");
    if (!file || stat(path,&info)) return NULL;
    unsigned char *data=malloc((size_t)info.st_size);
    if (!data || fread(data,1,(size_t)info.st_size,file)!=(size_t)info.st_size) return NULL;
    fclose(file);
    *bytes=(size_t)info.st_size;
    return data;
}

int main(int argc,char **argv)
{
    assert(argc==3);
    unsigned layer=(unsigned)atoi(argv[2]);
    char path[4096];
    snprintf(path,sizeof path,"%s/root-%u",argv[1],layer);
    Root *root=root_open(path,layer);
    RootScratch *scratch=calloc((size_t)omp_get_max_threads(),sizeof *scratch);
    assert(root && scratch);

    const char *sets[2]={"france","japan"};
    unsigned char *results[2],*inputs[2];
    size_t result_bytes[2],input_bytes[2];
    for (unsigned set=0;set<2;set++) {
        snprintf(path,sizeof path,"%s/root-%u/observations/%s/results.bin",argv[1],layer,sets[set]);
        results[set]=slurp(path,&result_bytes[set]);
        snprintf(path,sizeof path,"%s/root-%u/observations/%s/inputs.f32",argv[1],layer,sets[set]);
        inputs[set]=slurp(path,&input_bytes[set]);
        assert(results[set] && inputs[set]);
    }

    /* One expert that two different recorded inputs both selected. */
    unsigned found=0,chosen=0,set_a=0,rec_a=0,set_b=0,rec_b=0;
    for (unsigned sa=0;sa<2 && !found;sa++)
    for (unsigned ra=0;ra<result_bytes[sa]/REC && !found;ra++)
    for (unsigned sb=sa;sb<2 && !found;sb++)
    for (unsigned rb=(sb==sa?ra+1:0);rb<result_bytes[sb]/REC && !found;rb++) {
        int32_t ea,eb;
        memcpy(&ea,results[sa]+(size_t)ra*REC,4);
        memcpy(&eb,results[sb]+(size_t)rb*REC,4);
        if (ea!=eb) continue;
        if (!memcmp(inputs[sa]+(size_t)ra*IN*4,inputs[sb]+(size_t)rb*IN*4,IN*4)) continue;
        found=1; chosen=(unsigned)ea; set_a=sa; rec_a=ra; set_b=sb; rec_b=rb;
    }
    assert(found);

    const float *input_a=(const float *)(inputs[set_a]+(size_t)rec_a*IN*4);
    const float *input_b=(const float *)(inputs[set_b]+(size_t)rec_b*IN*4);
    const unsigned char *record_a=results[set_a]+(size_t)rec_a*REC;
    const unsigned char *record_b=results[set_b]+(size_t)rec_b*REC;
    const float *gate_a=(const float *)(record_a+4), *gate_b=(const float *)(record_b+4);
    const float *up_a=(const float *)(record_a+4+HID*4), *up_b=(const float *)(record_b+4+HID*4);

    float projected_gate_a[HID],projected_gate_b[HID],projected_up_a[HID],projected_up_b[HID];
    assert(root_project(root,scratch,chosen,0,input_a,projected_gate_a));
    assert(root_project(root,scratch,chosen,0,input_b,projected_gate_b));
    assert(root_project(root,scratch,chosen,1,input_a,projected_up_a));
    assert(root_project(root,scratch,chosen,1,input_b,projected_up_b));

    printf("layer %u expert %u, two different recorded inputs, one unchanged weight set\n",layer,chosen);
    printf("  recorded gate A == recorded gate B : %s\n",memcmp(gate_a,gate_b,HID*4)?"no":"yes");
    printf("  recorded up   A == recorded up   B : %s\n",memcmp(up_a,up_b,HID*4)?"no":"yes");
    printf("  weights x input A == recorded gate A : %s\n",memcmp(projected_gate_a,gate_a,HID*4)?"no":"yes");
    printf("  weights x input B == recorded gate B : %s\n",memcmp(projected_gate_b,gate_b,HID*4)?"no":"yes");
    printf("  weights x input A == recorded up   A : %s\n",memcmp(projected_up_a,up_a,HID*4)?"no":"yes");
    printf("  weights x input B == recorded up   B : %s\n",memcmp(projected_up_b,up_b,HID*4)?"no":"yes");
    printf("  weights x input A == weights x input B : %s\n",memcmp(projected_gate_a,projected_gate_b,HID*4)?"no":"yes");

    root_close(root); free(scratch);
    for (unsigned set=0;set<2;set++) { free(results[set]); free(inputs[set]); }
    return 0;
}
