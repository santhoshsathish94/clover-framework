#ifndef CLOVER_ROOT_METADATA_H
#define CLOVER_ROOT_METADATA_H
#include "generation.h"
#include <sys/stat.h>

static int root_metadata(const char *path, unsigned layer, size_t constant_bytes, unsigned counts[4])
{
    FILE *file=fopen(path,"rb");
    if (!file) return 0;
    if (fseek(file,0,SEEK_END)) { fclose(file); return 0; }
    long bytes=ftell(file);
    if (bytes<=0 || bytes>16*1024*1024 || fseek(file,0,SEEK_SET)) { fclose(file); return 0; }
    unsigned char *data=malloc((size_t)bytes);
    if (!data) { fclose(file); return 0; }
    int valid=fread(data,1,(size_t)bytes,file)==(size_t)bytes && !ferror(file);
    if (fclose(file)) valid=0;
    JsonReader reader={data,data+bytes,0};
    unsigned seen=0;
    if (valid) valid=json_take(&reader,'{');
    while (valid) {
        unsigned char *key=NULL; size_t length;
        if (!json_string(&reader,&key,&length)) { valid=0; break; }
        unsigned field=key_is(key,length,"layer")?1:key_is(key,length,"palette_count")?2:
            key_is(key,length,"unique_maps")?4:key_is(key,length,"unique_templates")?8:
            key_is(key,length,"template_refs")?16:0;
        free(key);
        if ((seen & field) || !json_take(&reader,':')) { valid=0; break; }
        if (field) {
            unsigned value;
            if (!generation_uint(&reader,&value)) { valid=0; break; }
            if (field==1) { if (value!=layer) valid=0; }
            else counts[field==2?0:field==4?1:field==8?2:3]=value;
            seen|=field;
        } else valid=json_skip(&reader);
        if (json_take(&reader,'}')) break;
        if (!json_take(&reader,',')) valid=0;
    }
    json_space(&reader);
    size_t other_bytes=(size_t)counts[1]*16 + ((size_t)counts[2]+1)*2 + (size_t)counts[3]*2;
    if (!(seen & 2) && constant_bytes>other_bytes && (constant_bytes-other_bytes)%2==0)
        counts[0]=(unsigned)((constant_bytes-other_bytes)/2);
    valid=valid && (seen & 28)==28 && reader.cursor==reader.end && counts[0]>0 && counts[0]<=255 &&
        counts[1]>0 && counts[1]<=65535 && counts[2]>0 && counts[2]<=2688 && counts[3]>0 && counts[3]<=65535;
    valid=valid && constant_bytes==other_bytes+(size_t)counts[0]*2;
    free(data);
    return valid;
}

static int root_read_at(FILE *file, unsigned char *output, size_t bytes, uint64_t offset)
{
    if (offset>INT64_MAX || bytes>(uint64_t)INT64_MAX-offset) return 0;
    size_t received=0;
    while (received<bytes) {
        ssize_t count=pread(fileno(file),output+received,bytes-received,(off_t)(offset+received));
        if (count<0 && errno==EINTR) continue;
        if (count<=0) return 0;
        received+=(size_t)count;
    }
    return 1;
}
#endif