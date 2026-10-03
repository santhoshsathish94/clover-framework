static int transformer_load_qkv(Transformer *transformer, const char *path)
{
    const size_t bytes = TRANSFORMER_MLA ? 56251936 :
        TRANSFORMER_QKV_PREFIX + (size_t)TRANSFORMER_ROWS * 3 * TRANSFORMER_QKV_COMPONENT;
    FILE *file = transformer_open_sized(path, bytes);
    if (!file) return 0;
    int loaded = transformer_read(file, 0, bytes, &transformer->qkv);
    if (fclose(file)) loaded = 0;
    if (!loaded) return 0;
    const unsigned char *data = transformer->qkv;
    const uint32_t fields[] = {1, TRANSFORMER_LAYER, TRANSFORMER_WIDTH,
        TRANSFORMER_MLA ? 18432 : TRANSFORMER_ROWS, TRANSFORMER_MLA ? 192 : 128,
        3, TRANSFORMER_MLA ? 0 : 4, TRANSFORMER_MLA ? 7 : 6,
        TRANSFORMER_MLA ? 32 : TRANSFORMER_QKV_COMPONENT, 0};
    if (memcmp(data, TRANSFORMER_MLA ? "K3MLA001" : "K3QKV001", 8)) return 0;
    for (unsigned field = 0; field < 10; field++) if (transformer_u32(data + 8 + field * 4) != fields[field]) return 0;
    if (transformer_f32(data + 48) != 1e-5f || transformer_f32(data + 52) != 1e-6f || transformer_u64(data + 56) != bytes) return 0;
    size_t palette_offset = TRANSFORMER_MLA ? 64 + 7 * 32 : 88 + TRANSFORMER_WIDTH * 4;
    for (unsigned code = 0; code < 256; code++) {
        float expected = (float)(code < 128 ? (int)code : (int)code - 256);
        float stored = transformer_f32(data + palette_offset + code * 4);
        if (memcmp(&stored, &expected, 4)) return 0;
        transformer->palette[code] = stored;
    }
    if (!TRANSFORMER_MLA) {
        for (unsigned stage = 0; stage < 6; stage++) if (transformer_u32(data + 64 + stage * 4) != stage + 1) return 0;
        transformer->input_gains = data + 88;
        for (unsigned coordinate = 0; coordinate < TRANSFORMER_WIDTH; coordinate++)
            if (!isfinite(transformer_f32(data + 88 + coordinate * 4))) return 0;
        for (unsigned row = 0; row < TRANSFORMER_ROWS; row++) for (unsigned component = 0; component < 3; component++) {
            const unsigned char *record = data + TRANSFORMER_QKV_PREFIX + ((size_t)row * 3 + component) * TRANSFORMER_QKV_COMPONENT;
            for (unsigned field = 0; field < 5; field++) if (!isfinite(transformer_f32(record + field * 4))) return 0;
        }
    } else {
        const unsigned schema[7][4] = {{4,2,7168,1},{20,1,1536,7168},{21,2,1536,1},
            {22,1,18432,1536},{23,1,576,7168},{24,2,512,1},{25,1,24576,512}};
        size_t next = 64 + 7 * 32 + 1024;
        for (unsigned index = 0; index < 7; index++) {
            const unsigned char *encoded = data + 64 + index * 32;
            for (unsigned field = 0; field < 4; field++) if (transformer_u32(encoded + field * 4) != schema[index][field]) return 0;
            unsigned slot = schema[index][0], kind = schema[index][1], rows = schema[index][2], columns = schema[index][3];
            size_t length = kind == 1 ? rows * 4ULL + (size_t)rows * columns : (size_t)rows * columns * 4;
            if (transformer_u64(encoded + 16) != next || transformer_u64(encoded + 24) != length || length > bytes - next) return 0;
            TransformerRecord *record = &transformer->operator_records[slot];
            *record = (TransformerRecord){slot, 1, kind, rows, columns, rows * columns, next, length, transformer->qkv + next};
            unsigned floats = kind == 1 ? rows : rows * columns;
            for (unsigned coordinate = 0; coordinate < floats; coordinate++)
                if (!isfinite(transformer_f32(record->data + (size_t)coordinate * 4))) return 0;
            next += length;
        }
        if (next != bytes) return 0;
        transformer->input_gains = transformer->operator_records[4].data;
    }
    return 1;
}