static void reverse_order(unsigned *scale, unsigned counts[16], GroupOrderInteger *rank,
                          unsigned char payload[29], unsigned *bits)
{
    const unsigned expected_scale = *scale;
    const GroupOrderInteger expected_rank = *rank;
    unsigned expected[16];
    memcpy(expected, counts, sizeof(expected));
    GroupOrderInteger ways;
    if (!group_order_ways(counts, 32, &ways) || *rank >= ways) die("reverse order domain differs");
    unsigned rank_bits = 0, offset = 0;
    for (GroupOrderInteger remaining = ways - 1; remaining; remaining >>= 1) rank_bits++;
    memset(payload, 0, 29);
    group_order_put(payload, &offset, *scale, 8);
    for (unsigned code = 0; code < 16; code++) group_order_put(payload, &offset, counts[code], 6);
    group_order_put(payload, &offset, *rank, rank_bits);
    *bits = offset;
    memset(counts, 0, sizeof(expected));
    *rank = 0;
    *scale = 0;
    offset = 0;
    *scale = group_order_get(payload, &offset, 8);
    for (unsigned code = 0; code < 16; code++) counts[code] = group_order_get(payload, &offset, 6);
    if (!group_order_ways(counts, 32, &ways)) die("reverse order decoded domain differs");
    rank_bits = 0;
    for (GroupOrderInteger remaining = ways - 1; remaining; remaining >>= 1) rank_bits++;
    *rank = group_order_get(payload, &offset, rank_bits);
    if (offset != *bits || *scale != expected_scale || *rank != expected_rank || memcmp(counts, expected, sizeof(expected)))
        die("reverse order consumed fields differ");
}