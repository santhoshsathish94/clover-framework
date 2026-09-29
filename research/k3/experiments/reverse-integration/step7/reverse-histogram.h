#include "expert-histogram-codec.h"

static void reverse_histogram(unsigned counts[16], unsigned char payload[5])
{
    if (!coverage_groups) {
        histogram_selfcheck();
        if (histogram_decoder_checks != 105 || histogram_invalid_checks != 4)
            die("reverse histogram control coverage differs");
    }
    uint64_t rank;
    unsigned expected[16];
    memcpy(expected, counts, sizeof(expected));
    if (!histogram_encode(counts, 32, 16, &rank)) die("reverse histogram encode failed");
    for (unsigned byte = 0; byte < 5; byte++) payload[byte] = rank >> (8 * byte);
    memset(counts, 0, sizeof(expected));
    rank = 0;
    for (unsigned byte = 0; byte < 5; byte++) rank |= (uint64_t)payload[byte] << (8 * byte);
    if (!histogram_decode(rank, 32, 16, counts) || memcmp(counts, expected, sizeof(expected)))
        die("reverse histogram consumed counts differ");
}