#ifndef SERVER_SAFETENSORS_H
#define SERVER_SAFETENSORS_H
/* Tensors of a safetensors file (8-byte little-endian header length, a JSON header, then the raw data), found by their own names: this is how the engine
   reads the open Kimi K3 weights (the embedding table, the output head and the final normalisation tensors) without a converted copy. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

typedef struct {
    uint64_t offset, bytes;      /* from the start of the file */
    int bf16;
    unsigned dimensions;
    uint64_t shape[4];
} SafetensorsTensor;

static const char *safetensors_after_key(const char *object, const char *key)
{
    char quoted[64];
    snprintf(quoted, sizeof quoted, "\"%s\"", key);
    const char *at = strstr(object, quoted);
    if (!at) return NULL;
    at += strlen(quoted);
    while (*at == ' ') at++;
    if (*at != ':') return NULL;
    at++;
    while (*at == ' ') at++;
    return at;
}

static int safetensors_numbers(const char *at, uint64_t *values, unsigned capacity, unsigned *count)
{
    if (*at != '[') return 0;
    at++;
    unsigned n = 0;
    for (;;) {
        while (*at == ' ') at++;
        if (*at == ']') break;
        if (n == capacity) return 0;
        char *stop;
        unsigned long long value = strtoull(at, &stop, 10);
        if (stop == at) return 0;
        values[n++] = value;
        at = stop;
        while (*at == ' ') at++;
        if (*at == ',') at++; else if (*at != ']') return 0;
    }
    *count = n;
    return 1;
}

/* 1 and the tensor's place in the file if `path` is a safetensors file that holds `name`; the file size too, when asked. */
static int safetensors_find(const char *path, const char *name, SafetensorsTensor *tensor, uint64_t *file_bytes)
{
    int handle = open(path, O_RDONLY);
    if (handle < 0) return 0;
    struct stat info;
    unsigned char prefix[8];
    char *header = NULL;
    int found = 0;
    if (!fstat(handle, &info) && pread(handle, prefix, 8, 0) == 8) {
        uint64_t header_bytes = 0;
        for (unsigned byte = 0; byte < 8; byte++) header_bytes |= (uint64_t)prefix[byte] << (8 * byte);
        if (header_bytes >= 2 && header_bytes <= (64ULL << 20) && 8 + header_bytes <= (uint64_t)info.st_size &&
            (header = malloc((size_t)header_bytes + 1)) && pread(handle, header, (size_t)header_bytes, 8) == (ssize_t)header_bytes) {
            header[header_bytes] = 0;
            char key[512];
            snprintf(key, sizeof key, "\"%s\"", name);
            char *at = strstr(header, key);
            if (at) {
                at += strlen(key);
                while (*at == ' ') at++;
                if (*at == ':') {
                    at++;
                    while (*at == ' ') at++;
                }
                char *object_end = *at == '{' ? strchr(at, '}') : NULL;
                if (object_end) {
                    *object_end = 0;
                    const char *dtype = safetensors_after_key(at, "dtype");
                    const char *shape = safetensors_after_key(at, "shape");
                    const char *offsets = safetensors_after_key(at, "data_offsets");
                    uint64_t pair[2];
                    unsigned pair_count = 0;
                    SafetensorsTensor t = {0};
                    if (dtype && shape && offsets && safetensors_numbers(shape, t.shape, 4, &t.dimensions) &&
                        safetensors_numbers(offsets, pair, 2, &pair_count) && pair_count == 2 && pair[1] >= pair[0] &&
                        8 + header_bytes + pair[1] <= (uint64_t)info.st_size) {
                        t.bf16 = !strncmp(dtype, "\"BF16\"", 6);
                        t.offset = 8 + header_bytes + pair[0];
                        t.bytes = pair[1] - pair[0];
                        *tensor = t;
                        if (file_bytes) *file_bytes = (uint64_t)info.st_size;
                        found = 1;
                    }
                }
            }
        }
    }
    free(header);
    close(handle);
    return found;
}
#endif
