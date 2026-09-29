static void reverse_four_mask(CoverageGroup *group, unsigned char payload[17])
{
    unsigned char packed[16] = {0}, expected[32];
    memcpy(expected, group->codes, sizeof(expected));
    const unsigned expected_scale = group->scale;
    for (unsigned position = 0; position < 32; position++)
        packed[position / 2] |= group->codes[position] << (4 * (position % 2));
    expert_group_masks(packed, group->masks);
    for (unsigned plane = 0; plane < 4; plane++)
        for (unsigned byte = 0; byte < 4; byte++) payload[4 * plane + byte] = group->masks[plane] >> (8 * byte);
    payload[16] = group->scale;
    memset(group->masks, 0, sizeof(group->masks));
    memset(group->codes, 0, sizeof(group->codes));
    for (unsigned plane = 0; plane < 4; plane++)
        for (unsigned byte = 0; byte < 4; byte++) group->masks[plane] |= (uint32_t)payload[4 * plane + byte] << (8 * byte);
    group->scale = payload[16];
    for (unsigned position = 0; position < 32; position++) {
        group->codes[position] = expert_group_code(group->masks, position);
        const float value = expert_group_weight(group->codes[position], group->scale);
        if (group->codes[position] != expected[position] ||
            memcmp(&value, &DQ[expected_scale][expected[position]], sizeof(value)))
            die("reverse four-mask decoded value differs");
    }
    if (group->scale != expected_scale) die("reverse four-mask scale differs");
}