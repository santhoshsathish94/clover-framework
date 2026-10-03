#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include "k3_tok.h"
#include "numeric-table.h"

static double input_clock(void)
{
    struct timespec value; clock_gettime(CLOCK_MONOTONIC,&value);
    return (double)value.tv_sec+value.tv_nsec*1e-9;
}

int main(int argc,char **argv)
{
    assert(argc==4 && strlen(argv[3])<1024);
    double tick=input_clock();
    Tok tokenizer; k3_tok_load(&tokenizer,argv[1]);
    printf("INPUT_STAGE tokenizer_load_ms=%.6f\n",(input_clock()-tick)*1000);
    int ids[256];
    tick=input_clock();
    int count=tok_encode(&tokenizer,argv[3],(int)strlen(argv[3]),ids,256);
    double first_encode=input_clock()-tick;
    assert(count>0 && count<256);
    char decoded[1024];
    int length=tok_decode(&tokenizer,ids,count,decoded,sizeof decoded);
    assert(length==(int)strlen(argv[3]) && !memcmp(decoded,argv[3],(size_t)length));
    tick=input_clock();
    for (unsigned repeat=0;repeat<1000;repeat++) {
        int actual[256];
        assert(tok_encode(&tokenizer,argv[3],(int)strlen(argv[3]),actual,256)==count);
        assert(!memcmp(ids,actual,(size_t)count*sizeof(int)));
    }
    printf("INPUT_STAGE text_to_ids_first_ms=%.6f repeated_mean_ms=%.6f tokens=%d\n",first_encode*1000,input_clock()-tick,count);
    char path[4096]; snprintf(path,sizeof path,"%s/inputs/seed.bin",argv[2]);
    tick=input_clock(); ClientInput *input=client_input_open(path); assert(input);
    printf("INPUT_STAGE seed_open_metadata_ms=%.6f\n",(input_clock()-tick)*1000);
    snprintf(path,sizeof path,"%s/model/model-00094-of-000096.safetensors",argv[2]);
    FILE *original=fopen(path,"rb"); assert(original);
    float vector[CLIENT_WIDTH]; unsigned char raw[CLIENT_WIDTH*2];
    for (int position=0;position<count;position++) {
        tick=input_clock(); assert(client_input(input,(uint32_t)ids[position],vector));
        double first=input_clock()-tick;
        assert(!fseeko(original,(off_t)(UINT64_C(2348810824)+(uint64_t)ids[position]*CLIENT_WIDTH*2),SEEK_SET));
        assert(fread(raw,1,sizeof raw,original)==sizeof raw);
        for (unsigned coordinate=0;coordinate<CLIENT_WIDTH;coordinate++) {
            float expected=table_value(raw+coordinate*2);
            assert(!memcmp(vector+coordinate,&expected,4));
        }
        tick=input_clock();
        for (unsigned repeat=0;repeat<1000;repeat++) assert(client_input(input,(uint32_t)ids[position],vector));
        printf("INPUT_STAGE position=%d id=%d first_row_ms=%.6f cached_row_mean_ms=%.6f exact_coordinates=7168\n",
            position,ids[position],first*1000,input_clock()-tick);
    }
    assert(!fclose(original)); client_input_close(input);
    puts("PASS: native text roundtrip and all embedding coordinates exact; no model layer ran");
    return 0;
}