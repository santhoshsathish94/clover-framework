#ifndef CLIENT_NUMERIC_TABLE_H
#define CLIENT_NUMERIC_TABLE_H
#include "decode.h"
#include "sha256.h"
#include <fcntl.h>
#include <math.h>
#include <float.h>
#include <fenv.h>
#include <limits.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

enum { CLIENT_WIDTH=7168, CLIENT_ROWS=163840, CLIENT_BLOCK_ROWS=16, CLIENT_BLOCKS=10240,
    CLIENT_RAW_BYTES=16*7168*2 };
typedef struct {
    FILE *file;
    unsigned char *metadata;
    const unsigned char *dictionary,*index;
    unsigned dictionary_count,bits,cached;
    unsigned char payload[CLIENT_RAW_BYTES],decoded[CLIENT_RAW_BYTES],raw[CLIENT_RAW_BYTES];
    uint32_t crc_table[256];
    const unsigned char *direct;
    size_t direct_bytes;
} NumericTable;

static uint32_t table_u32(const unsigned char *data)
{
    return (uint32_t)data[0]|(uint32_t)data[1]<<8|(uint32_t)data[2]<<16|(uint32_t)data[3]<<24;
}
static uint64_t table_u64(const unsigned char *data)
{
    return table_u32(data)|(uint64_t)table_u32(data+4)<<32;
}
static uint32_t table_crc(const NumericTable *table,const unsigned char *data,size_t bytes)
{
    uint32_t value=UINT32_MAX;
    for(size_t index=0;index<bytes;index++)value=table->crc_table[(value^data[index])&255]^(value>>8);
    return value^UINT32_MAX;
}
static void table_close(NumericTable *table)
{
    if(!table)return;
    if(table->file)fclose(table->file);
    if(table->direct)munmap((void *)table->direct,table->direct_bytes);
    free(table->metadata);free(table);
}
static NumericTable *table_open(const char *path,const char *expected_source)
{
    if(!path)return NULL;
    NumericTable *table=calloc(1,sizeof *table);
    if(!table)return NULL;
    table->cached=UINT32_MAX;
    table->file=fopen(path,"rb");
    if(!table->file||fseek(table->file,0,SEEK_END)){table_close(table);return NULL;}
    long file_bytes=ftell(table->file);
    if(file_bytes<128||fseek(table->file,0,SEEK_SET)){table_close(table);return NULL;}
    unsigned char header[128];
    if(fread(header,1,128,table->file)!=128||memcmp(header,"K3SEED1\0",8)||table_u32(header+8)!=1||
        table_u32(header+12)!=CLIENT_ROWS||table_u32(header+16)!=CLIENT_WIDTH||table_u32(header+20)!=CLIENT_BLOCK_ROWS||
        table_u32(header+32)!=CLIENT_BLOCKS||table_u32(header+36)){table_close(table);return NULL;}
    char source[65];
    for(unsigned index=0;index<32;index++)snprintf(source+index*2,3,"%02x",header[64+index]);
    if(strcmp(source,expected_source)){table_close(table);return NULL;}
    table->dictionary_count=table_u32(header+24);table->bits=table_u32(header+28);
    if(!table->dictionary_count||table->dictionary_count>65536){table_close(table);return NULL;}
    unsigned bits=0;
    for(unsigned value=table->dictionary_count-1;value;value>>=1)bits++;
    uint64_t index_offset=128+table->dictionary_count*2ULL,payload_offset=index_offset+CLIENT_BLOCKS*56ULL;
    if(table->bits!=bits||table_u64(header+40)!=128||table_u64(header+48)!=index_offset||
        table_u64(header+56)!=payload_offset||payload_offset>(uint64_t)file_bytes){table_close(table);return NULL;}
    size_t bytes=(size_t)payload_offset-128;
    table->metadata=malloc(bytes);
    if(!table->metadata||fread(table->metadata,1,bytes,table->file)!=bytes){table_close(table);return NULL;}
    ClientHash hash;unsigned char digest[32];client_hash_init(&hash);client_hash_update(&hash,header,96);
    client_hash_update(&hash,table->metadata,bytes);client_hash_final(&hash,digest);
    if(memcmp(digest,header+96,32)){table_close(table);return NULL;}
    table->dictionary=table->metadata;table->index=table->metadata+table->dictionary_count*2;
    unsigned char seen[65536]={0};
    for(unsigned index=0;index<table->dictionary_count;index++){
        unsigned word=table->dictionary[index*2]|(unsigned)table->dictionary[index*2+1]<<8;
        if(seen[word]){table_close(table);return NULL;}seen[word]=1;
    }
    uint64_t cursor=payload_offset;
    for(unsigned block=0;block<CLIENT_BLOCKS;block++){
        const unsigned char *entry=table->index+block*56;
        unsigned length=table_u32(entry+8);
        if(table_u64(entry)!=cursor||!length||length>CLIENT_RAW_BYTES||table_u32(entry+12)!=16||table_u32(entry+16)>5){table_close(table);return NULL;}
        cursor+=length;
        if(cursor>(uint64_t)file_bytes){table_close(table);return NULL;}
    }
    if(cursor!=(uint64_t)file_bytes){table_close(table);return NULL;}
    for(unsigned byte=0;byte<256;byte++){
        uint32_t crc=byte;
        for(unsigned bit=0;bit<8;bit++)crc=(crc>>1)^((crc&1)?UINT32_C(0xedb88320):0);
        table->crc_table[byte]=crc;
    }
    char direct_path[4096];
    int direct_length=snprintf(direct_path,sizeof direct_path,"%s.direct",path);
    if(direct_length>0&&(size_t)direct_length<sizeof direct_path){
        int handle=open(direct_path,O_RDONLY);
        if(handle>=0){
            struct stat direct_info;
            size_t bytes=(size_t)CLIENT_BLOCKS*CLIENT_RAW_BYTES;
            if(!fstat(handle,&direct_info)&&(size_t)direct_info.st_size==bytes){
                void *mapped=mmap(NULL,bytes,PROT_READ,MAP_SHARED,handle,0);
                if(mapped!=MAP_FAILED){table->direct=mapped;table->direct_bytes=bytes;}
            }
            close(handle);
        }
    }
    return table;
}
static unsigned table_bits(const unsigned char *data,size_t offset,unsigned count)
{
    unsigned value=0;
    for(unsigned bit=0;bit<count;bit++)value|=((data[(offset+bit)/8]>>((offset+bit)%8))&1U)<<bit;
    return value;
}
static int table_load(NumericTable *table,unsigned block)
{
    if(!table||block>=CLIENT_BLOCKS)return 0;
    if(table->cached==block)return 1;
    if(table->direct){
        memcpy(table->raw,table->direct+(size_t)block*CLIENT_RAW_BYTES,CLIENT_RAW_BYTES);
        table->cached=block;
        return 1;
    }
    table->cached=UINT32_MAX;
    const unsigned char *entry=table->index+block*56;
    unsigned length=table_u32(entry+8),codec=table_u32(entry+16);
    uint64_t offset=table_u64(entry);
    size_t count=CLIENT_BLOCK_ROWS*CLIENT_WIDTH,decoded_bytes=codec<=2?(count*table->bits+7)/8:CLIENT_RAW_BYTES;
    if(offset>LONG_MAX||fseek(table->file,(long)offset,SEEK_SET)||fread(table->payload,1,length,table->file)!=length||
        table_crc(table,table->payload,length)!=table_u32(entry+20))return 0;
    if(codec==0){if(length!=decoded_bytes)return 0;memcpy(table->decoded,table->payload,length);}
    else if(!decode_zlib(table->payload,length,table->decoded,decoded_bytes))return 0;
    if(codec<=2){
        if((count*table->bits)%8 && table->decoded[decoded_bytes-1]>>((count*table->bits)%8))return 0;
        for(size_t index=0;index<count;index++){
            unsigned id=0;
            if(codec==2)for(unsigned bit=0;bit<table->bits;bit++)id|=table_bits(table->decoded,bit*count+index,1)<<bit;
            else id=table_bits(table->decoded,index*table->bits,table->bits);
            if(id>=table->dictionary_count)return 0;
            memcpy(table->raw+index*2,table->dictionary+id*2,2);
        }
    }else if(codec==3)memcpy(table->raw,table->decoded,CLIENT_RAW_BYTES);
    else if(codec==4)for(size_t index=0;index<count;index++){
        table->raw[index*2]=table->decoded[index];table->raw[index*2+1]=table->decoded[count+index];
    }else for(size_t index=0;index<count;index++){
        unsigned word=0;
        for(unsigned bit=0;bit<16;bit++)word|=table_bits(table->decoded,bit*count+index,1)<<bit;
        table->raw[index*2]=(unsigned char)word;table->raw[index*2+1]=(unsigned char)(word>>8);
    }
    unsigned char digest[32];client_sha256(table->raw,CLIENT_RAW_BYTES,digest);
    if(memcmp(digest,entry+24,32))return 0;
    table->cached=block;
    return 1;
}
static float table_value(const unsigned char *data)
{
    uint32_t bits=((uint32_t)data[0]|(uint32_t)data[1]<<8)<<16;
    float value;memcpy(&value,&bits,4);return value;
}
static int client_numeric_environment(void)
{
    return sizeof(float)==4 && sizeof(double)==8 && FLT_RADIX==2 && FLT_MANT_DIG==24 &&
        DBL_MANT_DIG==53 && FLT_EVAL_METHOD==0 && fegetround()==FE_TONEAREST;
}

typedef struct { NumericTable *table; } ClientInput;
typedef struct { NumericTable *table; } ClientOutput;

ClientInput *client_input_open(const char *path)
{
    if(!client_numeric_environment())return NULL;
    ClientInput *input=calloc(1,sizeof *input);if(!input)return NULL;
    input->table=table_open(path,"4a79cdabdab6826b994aff69d90a72aa35dae32e90f8acaf1ce312c7bdc4a487");
    if(!input->table){free(input);return NULL;}
    return input;
}
void client_input_close(ClientInput *input)
{
    if(input){table_close(input->table);free(input);}
}
int client_input(ClientInput *input,uint32_t token,float vector[CLIENT_WIDTH])
{
    if(!input||!vector||token>=CLIENT_ROWS||!client_numeric_environment()||!table_load(input->table,token/16))return 0;
    const unsigned char *row=input->table->raw+(size_t)(token%16)*CLIENT_WIDTH*2;
    for(unsigned coordinate=0;coordinate<CLIENT_WIDTH;coordinate++)if(!isfinite(table_value(row+coordinate*2)))return 0;
    for(unsigned coordinate=0;coordinate<CLIENT_WIDTH;coordinate++)vector[coordinate]=table_value(row+coordinate*2);
    return 1;
}
ClientOutput *client_output_open(const char *path)
{
    if(!client_numeric_environment())return NULL;
    ClientOutput *output=calloc(1,sizeof *output);if(!output)return NULL;
    output->table=table_open(path,"11c1f1c09a8e0db55547b5e68ebfd1d8e3b503bee56c4e1312ef55ecd3e5580f");
    if(!output->table){free(output);return NULL;}return output;
}
void client_output_close(ClientOutput *output)
{
    if(output){table_close(output->table);free(output);}
}
static float client_head_row(const unsigned char *row,const float *vector)
{
    double lanes[16]={0};
    for(unsigned begin=0;begin<CLIENT_WIDTH;begin+=16)for(unsigned lane=0;lane<16;lane++)
        lanes[lane]+=(double)table_value(row+(begin+lane)*2)*(double)vector[begin+lane];
    double groups[4];
    for(unsigned group=0;group<4;group++)groups[group]=(lanes[group]+lanes[group+4])+(lanes[group+8]+lanes[group+12]);
    return (float)((groups[0]+groups[1])+(groups[2]+groups[3]));
}
int client_output_project(ClientOutput *output,const float vector[CLIENT_WIDTH],float *logits,uint32_t *token)
{
    if(!output||!vector||!token||!client_numeric_environment())return 0;
    for(unsigned coordinate=0;coordinate<CLIENT_WIDTH;coordinate++)if(!isfinite(vector[coordinate]))return 0;
    float maximum=0.0f;uint32_t selected=0;
    for(unsigned block=0;block<CLIENT_BLOCKS;block++){
        if(!table_load(output->table,block))return 0;
        for(unsigned local=0;local<16;local++){
            unsigned id=block*16+local;
            float score=client_head_row(output->table->raw+(size_t)local*CLIENT_WIDTH*2,vector);
            if(!isfinite(score))return 0;
            if(logits)logits[id]=score;
            if(!id||score>maximum){selected=id;maximum=score;}
        }
    }
    *token=selected;return 1;
}
#endif