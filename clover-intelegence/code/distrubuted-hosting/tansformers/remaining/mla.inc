static int transformer_mla(Transformer *transformer, TransformerSequence *sequence)
{
    const size_t stride = 2 * TRANSFORMER_ROWS + 64;
    size_t count = sequence->positions + 1;
    if (count > SIZE_MAX / stride / sizeof(float) || count > SIZE_MAX / (2 * sizeof(float))) return 0;
    if (count > sequence->mla_capacity) {
        size_t capacity = sequence->mla_capacity ? sequence->mla_capacity : 8;
        while (capacity < count) {
            if (capacity > SIZE_MAX / 2) return 0;
            capacity *= 2;
        }
        if (capacity > SIZE_MAX / stride / sizeof(float)) return 0;
        float *cache = realloc(sequence->mla_cache, capacity * stride * sizeof(float));
        if (!cache) return 0;
        sequence->mla_cache = cache;
        sequence->mla_capacity = capacity;
    }
    float *scores = malloc(count * 2 * sizeof(float));
    if (!scores) return 0;
    float *exponentials = scores + count;
    transformer_project(transformer, 20, sequence->normalized, sequence->mla_query_latent);
    transformer_normalize(sequence->mla_query_latent, sequence->mla_query_latent,
        transformer->operator_records[21].data, 1536);
    transformer_project(transformer, 22, sequence->mla_query_latent, sequence->mla_query);
    transformer_project(transformer, 23, sequence->normalized, sequence->mla_kv_latent);
    transformer_normalize(sequence->mla_kv_latent, sequence->mla_kv_latent,
        transformer->operator_records[24].data, 512);
    transformer_project(transformer, 25, sequence->mla_kv_latent, sequence->mla_expanded);
    float *current = sequence->mla_cache + sequence->positions * stride;
    for (unsigned head = 0; head < TRANSFORMER_HEADS; head++) {
        memcpy(current + head * 128, sequence->mla_expanded + head * 256, 128 * sizeof(float));
        memcpy(current + TRANSFORMER_ROWS + head * 128, sequence->mla_expanded + head * 256 + 128, 128 * sizeof(float));
    }
    memcpy(current + 2 * TRANSFORMER_ROWS, sequence->mla_kv_latent + 512, 64 * sizeof(float));
    const float scale = 1.0f / sqrtf(192.0f);
    for (unsigned head = 0; head < TRANSFORMER_HEADS; head++) {
        const float *query = sequence->mla_query + head * 192;
        for (size_t position = 0; position < count; position++) {
            const float *cached = sequence->mla_cache + position * stride;
            double score = 0.0;
            for (unsigned coordinate = 0; coordinate < 128; coordinate++)
                score += (double)query[coordinate] * (double)cached[head * 128 + coordinate];
            for (unsigned coordinate = 0; coordinate < 64; coordinate++)
                score += (double)query[128 + coordinate] * (double)cached[2 * TRANSFORMER_ROWS + coordinate];
            scores[position] = (float)score * scale;
        }
        float maximum = scores[0];
        for (size_t position = 1; position < count; position++) if (scores[position] > maximum) maximum = scores[position];
        double total = 0.0;
        for (size_t position = 0; position < count; position++) {
            exponentials[position] = expf(scores[position] - maximum);
            total += (double)exponentials[position];
        }
        float *output = sequence->attention + head * 128;
        memset(output, 0, 128 * sizeof(float));
        for (size_t position = 0; position < count; position++) {
            float weight = (float)((double)exponentials[position] / total);
            const float *value = sequence->mla_cache + position * stride + TRANSFORMER_ROWS + head * 128;
            for (unsigned coordinate = 0; coordinate < 128; coordinate++) output[coordinate] = output[coordinate] + weight * value[coordinate];
        }
    }
    free(scores);
    transformer_project(transformer, 6, sequence->normalized, sequence->gate);
    for (unsigned coordinate = 0; coordinate < TRANSFORMER_ROWS; coordinate++)
        sequence->gate[coordinate] = sequence->attention[coordinate] * transformer_sigmoid(sequence->gate[coordinate]);
    transformer_project(transformer, 7, sequence->gate, sequence->residual);
    return 1;
}