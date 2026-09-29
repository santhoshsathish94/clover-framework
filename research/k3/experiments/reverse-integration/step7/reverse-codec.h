#ifndef REVERSE_STAGE
#error REVERSE_STAGE must identify this cumulative build
#endif

static CoverageGroup reverse_packed_group(const unsigned char *packed, unsigned scale,
                                         int layer, int expert, int part, int row,
                                         int index, int width, int rows, int positions)
{
    CoverageGroup decoded = {0};
    decoded.index = index;
    unsigned char consumed[17];
    consumed[0] = scale;
    memcpy(consumed + 1, packed, 16);
    decoded.scale = consumed[0];
    for (unsigned coordinate = 0; coordinate < 32; coordinate++)
        decoded.codes[coordinate] = (consumed[1 + coordinate / 2] >> (4 * (coordinate % 2))) & 15;
    unsigned counts[16];
    GroupOrderInteger rank, ways;
    if (!group_order_encode(decoded.codes, 32, counts, &rank) || !group_order_ways(counts, 32, &ways))
        die("reverse audit encoding failed");
    unsigned rank_bits = 0, offset = 0;
    for (GroupOrderInteger remaining = ways - 1; remaining; remaining >>= 1) rank_bits++;
    unsigned char audit[29] = {0};
    group_order_put(audit, &offset, scale, 8);
    for (unsigned code = 0; code < 16; code++) group_order_put(audit, &offset, counts[code], 6);
    group_order_put(audit, &offset, rank, rank_bits);
    char rank_text[40];
    group_order_decimal(rank, rank_text);
    fprintf(coverage_log,
        "{\"type\":\"group\",\"layer\":%d,\"expert\":%d,\"part\":%d,"
        "\"row\":%d,\"group\":%d,\"width\":%d,\"rows\":%d,\"positions\":%d,"
        "\"scale\":%u,\"rank\":\"%s\",\"payload_bits\":%u,\"joint_mode\":0,"
        "\"consumer_format\":\"packed\",\"consumer_payload\":[",
        layer, expert, part, row, index, width, rows, positions, scale, rank_text, offset);
    for (unsigned byte = 0; byte < 17; byte++) fprintf(coverage_log, "%s%u", byte ? "," : "", consumed[byte]);
    fprintf(coverage_log, "],\"code_changes\":0,\"weight_changes\":0,\"packed\":[");
    for (unsigned byte = 0; byte < 16; byte++) fprintf(coverage_log, "%s%u", byte ? "," : "", packed[byte]);
    fprintf(coverage_log, "],\"payload\":[");
    for (unsigned byte = 0; byte < (offset + 7) / 8; byte++) fprintf(coverage_log, "%s%u", byte ? "," : "", audit[byte]);
    fprintf(coverage_log, "]}\n");
    coverage_groups++;
    coverage_weights += 32;
    return decoded;
}