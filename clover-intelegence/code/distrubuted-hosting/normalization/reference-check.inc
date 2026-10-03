static const char *test_leaves;

static void test_finish_layer(int layer)
{
    if (!layer) return;
    for (unsigned matrix = 0; matrix < 3; matrix++) assert(oracle_hits[matrix] == 80);
    if (layer != 92) return;
    assert(nsnap == 8);
    Normalization *model = NULL;
    assert(normalization_open(test_leaves,&model));
    float gains[3][7168], fold[7168];
    for (unsigned tensor = 0; tensor < 3; tensor++) {
        const MRec *record = &mrec[tensor + 1];
        assert(record->nbytes == 14336);
        const unsigned char *raw = file_ptr(record->file_id) + record->off;
        for (unsigned coordinate = 0; coordinate < 7168; coordinate++) {
            uint32_t bits = ((uint32_t)raw[coordinate * 2] | ((uint32_t)raw[coordinate * 2 + 1] << 8)) << 16;
            memcpy(gains[tensor] + coordinate,&bits,4);
        }
        assert(!memcmp(gains[tensor],model->gains[tensor],sizeof gains[tensor]));
    }
    for (unsigned coordinate = 0; coordinate < 7168; coordinate++) fold[coordinate] = gains[0][coordinate] * gains[1][coordinate];
    assert(!memcmp(fold,model->fold,sizeof fold));
    float aggregate[7168], expected[7168], actual[7168], snapshots[8*7168];
    float *sources[9];
    for (unsigned position = 0; position < 5; position++) {
        for (unsigned snapshot = 0; snapshot < 8; snapshot++) {
            sources[snapshot] = snap[position][snapshot];
            memcpy(snapshots + snapshot * 7168,sources[snapshot],7168*4);
        }
        sources[8] = resid[position];
        AR(aggregate,sources,9,fold);
        rmsnorm(expected,aggregate,gains[2],7168,EPS5);
        assert(normalization_process(model,resid[position],snapshots,8,actual));
        size_t differences = 0;
        for (unsigned coordinate = 0; coordinate < 7168; coordinate++)
            if (memcmp(actual + coordinate,expected + coordinate,4)) differences++;
        if (differences) fprintf(stderr,"normalization differs at position %u: %zu values\n",position,differences);
        assert(!differences);
    }
    normalization_close(model);
    printf("NORMALIZATION_PASS\t%s\t5\t35840\t21504\n",prompt_name);
    fflush(stdout);
}

int main(int argc, char **argv)
{
    if (argc != 4) return 2;
    dataset_root = argv[1];
    prompt_name = argv[2];
    test_leaves = argv[3];
    char *args[] = {"original-tail-reference","92",NULL};
    return original_main(2,args);
}