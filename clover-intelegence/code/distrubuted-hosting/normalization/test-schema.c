#define NORMALIZATION_NO_MAIN
#include "normalization.c"
#include <assert.h>

static void test_replacement(const unsigned char *original, size_t bytes, const char *before, const char *after)
{
    assert(strlen(before) == strlen(after));
    unsigned char *copy = malloc(bytes + 1);
    Normalization *model = calloc(1,sizeof *model);
    assert(copy && model);
    memcpy(copy,original,bytes);
    copy[bytes] = 0;
    unsigned char *match = (unsigned char *)strstr((char *)copy,before);
    assert(match);
    memcpy(match,after,strlen(after));
    assert(!normalization_parse(copy,bytes,model));
    free(copy);
    normalization_close(model);
}

int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    FILE *file = fopen(argv[1],"rb");
    assert(file && !fseek(file,0,SEEK_END));
    long size = ftell(file);
    assert(size > 0 && size < 1024 * 1024 && !fseek(file,0,SEEK_SET));
    size_t bytes = (size_t)size;
    unsigned char *data = malloc(bytes + 2);
    Normalization *model = calloc(1,sizeof *model);
    assert(data && model && fread(data,1,bytes,file) == bytes && !fclose(file));
    data[bytes] = 0;
    assert(normalization_parse(data,bytes,model));
    assert(normalization_prepare_fold(model));
    test_replacement(data,bytes,"clover-bf16-tensors-v1","clover-bf16-tensors-v2");
    test_replacement(data,bytes,"\"little\"","\"bigger\"");
    test_replacement(data,bytes,"\"base64\"","\"base65\"");
    test_replacement(data,bytes,"\"tensor_count\": 3","\"tensor_count\": 2");
    test_replacement(data,bytes,"\"payload_bytes\": 43008","\"payload_bytes\": 43009");
    test_replacement(data,bytes,"\"BF16\"","\"FP16\"");
    test_replacement(data,bytes,"7168","7167");
    test_replacement(data,bytes,"\"nbytes\": 14336","\"nbytes\": 14335");
    test_replacement(data,bytes,"output_attn_res_norm.weight","output_attn_res_proj.weight");
    assert(!normalization_parse(data,bytes / 2,model));
    data[bytes] = '!';
    assert(!normalization_parse(data,bytes + 1,model));
    data[bytes] = 0;
    unsigned char *encoded = (unsigned char *)strstr((char *)data,"\"data_base64\": \"");
    assert(encoded);
    encoded += strlen("\"data_base64\": \"");
    unsigned char saved[4];
    memcpy(saved,encoded,4);
    encoded[0] = '!';
    assert(!normalization_parse(data,bytes,model));
    memcpy(encoded,"gH8A",4);
    assert(!normalization_parse(data,bytes,model));
    memcpy(encoded,saved,4);
    assert(normalization_parse(data,bytes,model));
    JsonReader integer = {(const unsigned char *)"18446744073709551616",(const unsigned char *)"18446744073709551616" + 20,0};
    uint64_t value;
    assert(!normalization_integer(&integer,&value));
    integer = (JsonReader){(const unsigned char *)"01",(const unsigned char *)"01" + 2,0};
    assert(!normalization_integer(&integer,&value));
    free(data);
    normalization_close(model);
    puts("PASS: actual leaves parse and 15 schema/shape/type/name/framing/Base64/nonfinite/integer rejection checks");
    return 0;
}