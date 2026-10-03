#define main standalone_program_main
#include "clover-one.c"
#undef main
#include <assert.h>

static void check_global_values(void)
{
    char path[4096];
    resident_path(path, sizeof(path), "model/model-00094-of-000096.safetensors");
    size_t bytes;
    const unsigned char *original = map_file(path, &bytes);
    assert(n_model == 5);
    for (unsigned record = 0; record < 5; record++) {
        assert(mrec[record].off >= 0 && mrec[record].nbytes > 0);
        assert((uint64_t)mrec[record].off + (uint64_t)mrec[record].nbytes <= bytes);
    }
    const unsigned tokens[] = {0, 15, 16, 1008, 11989, 163839};
    float embedding[E];
    for (unsigned position = 0; position < sizeof(tokens) / sizeof(tokens[0]); position++) {
        assert(client_input(resident_input, tokens[position], embedding));
        const unsigned char *row = original + mrec[0].off + (size_t)tokens[position] * E * 2;
        for (unsigned coordinate = 0; coordinate < E; coordinate++) {
            float expected = table_value(row + coordinate * 2);
            assert(!memcmp(embedding + coordinate, &expected, sizeof(expected)));
        }
    }
    for (unsigned tensor = 0; tensor < 3; tensor++) {
        const unsigned char *values = original + mrec[tensor + 1].off;
        assert(mrec[tensor + 1].nbytes == E * 2);
        for (unsigned coordinate = 0; coordinate < E; coordinate++) {
            float expected = table_value(values + coordinate * 2);
            assert(!memcmp(resident_leaves->gains[tensor] + coordinate, &expected, sizeof(expected)));
        }
    }
    for (unsigned coordinate = 0; coordinate < E; coordinate++) {
        float expected = resident_leaves->gains[0][coordinate] * resident_leaves->gains[1][coordinate];
        assert(!memcmp(resident_leaves->fold + coordinate, &expected, sizeof(expected)));
    }
    float input[E];
    for (unsigned coordinate = 0; coordinate < E; coordinate++)
        input[coordinate] = (float)((int)(coordinate % 37) - 18) / 37.0f;
    float *expected = malloc(VOCAB * sizeof(float));
    float *actual = malloc(VOCAB * sizeof(float));
    assert(expected && actual);
    Bf(expected, input, (const uint16_t *)(original + mrec[4].off), VOCAB, E);
    uint32_t selected;
    assert(client_output_project(resident_output, input, actual, &selected));
    assert(!memcmp(expected, actual, VOCAB * sizeof(float)));
    unsigned best = 0;
    for (unsigned token = 1; token < VOCAB; token++) if (expected[token] > expected[best]) best = token;
    assert(selected == best);
    for (unsigned block = 0; block < CLIENT_BLOCKS; block++) {
        assert(table_load(resident_output->table, block));
        assert(!memcmp(resident_output->table->raw,
            original + mrec[4].off + (size_t)block * CLIENT_RAW_BYTES, CLIENT_RAW_BYTES));
    }
    free(expected);
    free(actual);
    assert(!munmap((void *)original, bytes));
    puts("PASS: six seed rows, all 21504 leaf values, all fruit bytes and 163840 scores match original data/arithmetic");
}

static void check_retained_values(void)
{
    unsigned long long vectors = 0, coefficients = 0;
    for (unsigned pass = 0; pass < 2; pass++) {
        for (int position = 0; position < NLAY; position++) {
            int layer = pass ? NLAY - 1 - position : position;
            prepared_open(layer);
            operator_open(layer);
            assert(!memcmp(operator_palette, prepared_header.palette, 1024));
            for (unsigned slot = 0; slot < 37; slot++) {
                const PreparedRecord *record = prepared_slots[slot];
                if (!operator_vectors[slot] && !operator_pointers[slot]) continue;
                assert(record);
                const unsigned char *original = prepared_groups[record->group] + record->offset;
                if (operator_vectors[slot]) {
                    const float *values = prepared_vector(layer, slot, record->count);
                    assert(!memcmp(values, original, record->length));
                    vectors += record->count;
                } else {
                    OperatorMatrix matrix = operator_matrix_view(slot_ptr(layer, (int)slot), record->columns, record->rows);
                    for (unsigned row = 0; row < record->rows; row++) {
                        assert(!memcmp(matrix.scales + row * matrix.scale_stride, original + row * 4, 4));
                        assert(!memcmp(matrix.codes + row * matrix.row_stride,
                            original + record->rows * 4 + (size_t)row * record->columns, record->columns));
                    }
                    coefficients += record->count;
                }
            }
            operator_close();
            prepared_close();
        }
        assert(resident_maps == 372 && resident_unmaps == 0 && resident_tap_layers == 69);
    }
    assert(coefficients == 39154876416ULL && vectors == 21780480ULL);
    printf("PASS: 186 retained bindings, %llu coefficient IDs, %llu vectors/taps; no remapping\n", coefficients, vectors);
}

static void check_observation_bindings(void)
{
    const char *sets[] = {"france", "japan"};
    unsigned records = 0;
    for (unsigned set = 0; set < 2; set++) {
        assert(!setenv("K3_RESULT_SET", sets[set], 1));
        for (int layer = 1; layer < NLAY; layer++) {
            recorded_open(layer);
            for (unsigned record = 0; record < 80; record++) {
                const unsigned char *values = recorded_roots + (size_t)record * 51204;
                int32_t expert;
                memcpy(&expert, values, sizeof(expert));
                assert(expert >= 0 && expert < NEXP);
                const float *input = (const float *)(recorded_inputs + (size_t)record * LAT * 4);
                const float *output = recorded_down(layer, expert, input);
                assert(!memcmp(output, values + 4 + 3 * I_ * 4, LAT * sizeof(float)));
                records++;
            }
        }
    }
    assert(records == 14720);
    puts("PASS: 14720 existing expert observation records resolve and match; no expert projections");
}

int main(int argc, char **argv)
{
    assert(argc == 2);
    resident_configure(argv[1]);
    result_options();
    load_index(getenv("K3_INDEX"));
    prepared_init();
    resident_startup(0);
    for (int file = 0; file < n_files; file++) assert(!fmap[file]);
    check_global_values();
    check_retained_values();
    check_observation_bindings();
    assert(resident_maps == 372 && resident_unmaps == 0);
    resident_shutdown();
    assert(resident_unmaps == 372 && !resident_input && !resident_output && !resident_leaves);
    puts("PASS: current dataset bindings and shutdown; no model request, no persisted vectors");
    return 0;
}