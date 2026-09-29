#include "expert-mask-consumer.h"

static void reverse_mask(unsigned char codes[32], unsigned char payload[5], unsigned *bits)
{
    if (!coverage_groups) {
        expert_mask_selfcheck();
        if (expert_mask_roundtrip_checks != 5249 || expert_mask_invalid_checks != 34)
            die("reverse mask control coverage differs");
    }
    uint32_t original = 0;
    for (unsigned position = 0; position < 32; position++) original |= ((codes[position] >> 2) & 1U) << position;
    const ExpertMaskRank encoded = expert_mask_encode(original, 32);
    unsigned rank_bits = 0;
    for (uint64_t remaining = expert_mask_choose(32, encoded.count) - 1; remaining; remaining >>= 1) rank_bits++;
    *bits = 6 + rank_bits;
    const uint64_t packed = (encoded.rank << 6) | encoded.count;
    const unsigned size = (*bits + 7) / 8;
    if (size > 5) die("reverse mask payload exceeds format bound");
    for (unsigned byte = 0; byte < size; byte++) payload[byte] = packed >> (8 * byte);
    for (unsigned position = 0; position < 32; position++) codes[position] &= ~4U;
    uint64_t loaded = 0;
    for (unsigned byte = 0; byte < size; byte++) loaded |= (uint64_t)payload[byte] << (8 * byte);
    const ExpertMaskRank restored = {(unsigned)(loaded & 63), loaded >> 6};
    uint32_t decoded = 0;
    if (!expert_mask_decode(restored, 32, &decoded) || decoded != original)
        die("reverse consumed mask differs");
    for (unsigned position = 0; position < 32; position++) codes[position] |= ((decoded >> position) & 1U) << 2;
}