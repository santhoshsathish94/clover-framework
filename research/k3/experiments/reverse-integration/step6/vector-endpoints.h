static int endpoint_mode(void)
{
    static int mode = -1;
    if (mode < 0) {
        const char *setting = getenv("K3_VECTOR_ENDPOINTS");
        mode = setting ? atoi(setting) : 0;
        if (mode < 0 || mode > 3) die("K3_VECTOR_ENDPOINTS must be 0, 1, 2 or 3");
    }
    return mode;
}

static const uint16_t *endpoint_embedding_table;
static const int *endpoint_ids;
static unsigned long long endpoint_embed_checked, endpoint_embed_changed;
static unsigned long long endpoint_entry_norm_checked, endpoint_entry_norm_changed;
static unsigned long long endpoint_tail_checked, endpoint_tail_changed;
static unsigned long long endpoint_tail_norm_checked, endpoint_tail_norm_changed;
static unsigned long long endpoint_logits_checked, endpoint_logits_changed;
static int endpoint_source_count;
static const float *endpoint_sources[16];
static float endpoint_coefficients[16];
static float endpoint_tail_weights[E];
static float endpoint_tail_inverse;
static int endpoint_capture_pending;
static int endpoint_coefficients_ready;

static float endpoint_bf16(uint16_t stored)
{
    const uint32_t word = (uint32_t)stored << 16;
    float value;
    memcpy(&value, &word, sizeof(value));
    return value;
}

static void endpoint_bind_embedding(const uint16_t *table, const int *ids)
{
    endpoint_embedding_table = table;
    endpoint_ids = ids;
    (void)endpoint_mode();
}

static void endpoint_entry(float *output, const float *weights,
                           const float *reference_embedding, int position)
{
    const uint16_t *row = endpoint_embedding_table + (size_t)endpoint_ids[position] * E;
    double square_sum = 0.0;
    for (int coordinate = 0; coordinate < E; coordinate++) {
        const float value = endpoint_bf16(row[coordinate]);
        endpoint_embed_changed += memcmp(&value, reference_embedding + coordinate, sizeof(float)) != 0;
        endpoint_embed_checked++;
        square_sum += (double)value * (double)value;
    }
    const float inverse = (float)(1.0 / sqrt(square_sum / (double)E + EPS5));
    for (int coordinate = 0; coordinate < E; coordinate++) {
        const float value = endpoint_bf16(row[coordinate]);
        const float weighted = weights[coordinate] * value;
        const float normalized = weighted * inverse;
        endpoint_entry_norm_changed += memcmp(&normalized, output + coordinate, sizeof(float)) != 0;
        endpoint_entry_norm_checked++;
        output[coordinate] = normalized;
    }
}

static void endpoint_begin_tail(float *const *sources, int count)
{
    if (!(endpoint_mode() & 2)) return;
    if (count < 1 || count > 16 || endpoint_source_count) die("endpoint tail source state");
    endpoint_source_count = count;
    for (int source = 0; source < count; source++) endpoint_sources[source] = sources[source];
    endpoint_capture_pending = 1;
}

static void endpoint_capture_coefficients(const float *coefficients, int count)
{
    if (!endpoint_capture_pending) return;
    if (count != endpoint_source_count) die("endpoint tail coefficient count");
    memcpy(endpoint_coefficients, coefficients, sizeof(float) * count);
    endpoint_capture_pending = 0;
    endpoint_coefficients_ready = 1;
}

static float endpoint_aggregate(int coordinate)
{
    float value = 0.0f;
    for (int source = 0; source < endpoint_source_count; source++) {
        const float contribution = endpoint_coefficients[source] * endpoint_sources[source][coordinate];
        value = value + contribution;
    }
    return value;
}

static float endpoint_normalized(int coordinate)
{
    const float aggregate = endpoint_aggregate(coordinate);
    const float weighted = endpoint_tail_weights[coordinate] * aggregate;
    return weighted * endpoint_tail_inverse;
}

static void endpoint_tail(float *logits, float *normalized, const float *aggregate,
                          const float *weights, const uint16_t *head)
{
    if (!endpoint_coefficients_ready) die("endpoint tail coefficients unavailable");
    memcpy(endpoint_tail_weights, weights, sizeof(endpoint_tail_weights));
    double square_sum = 0.0;
    for (int coordinate = 0; coordinate < E; coordinate++) {
        const float value = endpoint_aggregate(coordinate);
        endpoint_tail_changed += memcmp(&value, aggregate + coordinate, sizeof(float)) != 0;
        endpoint_tail_checked++;
        square_sum += (double)value * (double)value;
    }
    endpoint_tail_inverse = (float)(1.0 / sqrt(square_sum / (double)E + EPS5));
    for (int coordinate = 0; coordinate < E; coordinate++) {
        const float value = endpoint_normalized(coordinate);
        endpoint_tail_norm_changed += memcmp(&value, normalized + coordinate, sizeof(float)) != 0;
        endpoint_tail_norm_checked++;
        normalized[coordinate] = value;
    }
    float *generated = malloc(sizeof(float) * VOCAB);
    if (!generated) die("endpoint logits allocation");
#pragma omp parallel for schedule(static)
    for (int row = 0; row < VOCAB; row++) {
        const uint16_t *head_row = head + (size_t)row * E;
        double lanes[16] = {0};
        for (int block = 0; block < E; block += 16) {
            for (int lane = 0; lane < 16; lane++) {
                const int coordinate = block + lane;
                const float weight = endpoint_bf16(head_row[coordinate]);
                const float value = endpoint_normalized(coordinate);
                lanes[lane] += (double)weight * (double)value;
            }
        }
        double groups[4];
        for (int group = 0; group < 4; group++)
            groups[group] = (lanes[group] + lanes[group + 4]) +
                            (lanes[group + 8] + lanes[group + 12]);
        generated[row] = (float)((groups[0] + groups[1]) + (groups[2] + groups[3]));
    }
    for (int row = 0; row < VOCAB; row++) {
        endpoint_logits_changed += memcmp(generated + row, logits + row, sizeof(float)) != 0;
        endpoint_logits_checked++;
        logits[row] = generated[row];
    }
    free(generated);
}

static void endpoint_report(void)
{
    const int mode = endpoint_mode();
    const char *path = getenv("K3_ENDPOINT_REPORT");
    if (!path) die("K3_ENDPOINT_REPORT required for endpoint study");
    const int counts_valid =
        endpoint_embed_checked == ((mode & 1) ? (unsigned long long)(NPOS - TLO) * E : 0) &&
        endpoint_entry_norm_checked == endpoint_embed_checked &&
        endpoint_tail_checked == ((mode & 2) ? E : 0) &&
        endpoint_tail_norm_checked == endpoint_tail_checked &&
        endpoint_logits_checked == ((mode & 2) ? VOCAB : 0) &&
        (!(mode & 2) || endpoint_source_count == 9);
    const unsigned long long changed = endpoint_embed_changed + endpoint_entry_norm_changed +
        endpoint_tail_changed + endpoint_tail_norm_changed + endpoint_logits_changed;
    FILE *report = fopen(path, "wb");
    if (!report) die("endpoint report open");
    fprintf(report,
        "{\n  \"mode\": %d,\n  \"positions\": %d,\n  \"tail_sources\": %d,\n"
        "  \"embedding\": {\"checked\": %llu, \"changed\": %llu},\n"
        "  \"entry_norm\": {\"checked\": %llu, \"changed\": %llu},\n"
        "  \"tail_aggregate\": {\"checked\": %llu, \"changed\": %llu},\n"
        "  \"tail_norm\": {\"checked\": %llu, \"changed\": %llu},\n"
        "  \"head_logits\": {\"checked\": %llu, \"changed\": %llu},\n"
        "  \"counts_valid\": %s,\n  \"gate\": \"%s\"\n}\n",
        mode, NPOS, endpoint_source_count,
        endpoint_embed_checked, endpoint_embed_changed,
        endpoint_entry_norm_checked, endpoint_entry_norm_changed,
        endpoint_tail_checked, endpoint_tail_changed,
        endpoint_tail_norm_checked, endpoint_tail_norm_changed,
        endpoint_logits_checked, endpoint_logits_changed,
        counts_valid ? "true" : "false", counts_valid && changed == 0 ? "PASS" : "FAIL");
    if (fclose(report) != 0) die("endpoint report close");
    if (!counts_valid || changed) die("endpoint same-result check failed");
}