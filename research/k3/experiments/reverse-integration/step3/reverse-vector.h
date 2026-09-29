#include <zlib.h>

static float reverse_vector_values[E];
static int reverse_vector_ready;

static void reverse_vector_write(FILE *trace, const void *data, size_t size)
{
    if (fwrite(data, 1, size, trace) != size) die("reverse vector report write failed");
}

static void reverse_vector_record(FILE *trace, uint32_t kind, const unsigned char *payload, uint32_t size)
{
    const uint32_t header[2] = {kind, size};
    reverse_vector_write(trace, header, sizeof(header));
    reverse_vector_write(trace, payload, size);
}

static void reverse_vector(float values[E])
{
    const char *path = getenv("K3_REVERSE_VECTOR_REPORT");
    if (!path) die("K3_REVERSE_VECTOR_REPORT required");
    FILE *trace = fopen(path, "wb");
    if (!trace) die("reverse vector report open failed");
    const uint32_t header[3] = {0x31564552, E, REVERSE_STAGE};
    unsigned char original[E * 4], arranged[E * 4], decoded[E * 4];
    memcpy(original, values, sizeof(original));
    reverse_vector_write(trace, header, sizeof(header));
    reverse_vector_write(trace, original, sizeof(original));
    const uLong capacity = compressBound(sizeof(original));
    unsigned char *payload = malloc(capacity);
    if (!payload) die("reverse vector compression allocation failed");
    for (unsigned layout = 0; layout < 2; layout++) {
        const unsigned char *bytes = (const unsigned char *)values;
        for (unsigned position = 0; position < E; position++)
            for (unsigned lane = 0; lane < 4; lane++)
                arranged[layout ? lane * E + position : 4 * position + lane] = bytes[4 * position + lane];
        uLongf size = capacity, decoded_size = sizeof(decoded);
        if (compress2(payload, &size, arranged, sizeof(arranged), 6) != Z_OK)
            die("reverse vector compression failed");
        memset(values, 0, sizeof(original));
        if (uncompress(decoded, &decoded_size, payload, size) != Z_OK || decoded_size != sizeof(decoded))
            die("reverse vector decompression failed");
        unsigned char *restored = (unsigned char *)values;
        for (unsigned position = 0; position < E; position++)
            for (unsigned lane = 0; lane < 4; lane++)
                restored[4 * position + lane] = decoded[layout ? lane * E + position : 4 * position + lane];
        if (memcmp(original, values, sizeof(original))) die("reverse vector consumed bytes differ");
        reverse_vector_record(trace, layout + 1, payload, (uint32_t)size);
    }
    free(payload);
    if (fclose(trace)) die("reverse vector report close failed");
}