static void transformer_activation(float *gate, const float *up, unsigned width)
{
    for (unsigned coordinate = 0; coordinate < width; coordinate++) {
        float activated = (4.0f * tanhf(gate[coordinate] / 4.0f)) * transformer_sigmoid(gate[coordinate]);
        float capped = 25.0f * tanhf(up[coordinate] / 25.0f);
        gate[coordinate] = activated * capped;
    }
}

static int transformer_route(const Transformer *transformer, TransformerSequence *sequence)
{
    float scores[896], best[16];
    unsigned count = 0;
    for (unsigned expert = 0; expert < 896; expert++) {
        const unsigned char *row = transformer->records[29].data + (size_t)expert * TRANSFORMER_WIDTH * 4;
        double score = 0.0;
        for (unsigned coordinate = 0; coordinate < TRANSFORMER_WIDTH; coordinate++)
            score += (double)transformer_f32(row + coordinate * 4) * (double)sequence->postnorm[coordinate];
        scores[expert] = transformer_sigmoid((float)score);
        float choice = scores[expert] + transformer_f32(transformer->records[30].data + expert * 4);
        if (!isfinite(choice)) return 0;
        if (count == 16 && !(choice > best[15])) continue;
        unsigned position = count < 16 ? count : 15;
        while (position && choice > best[position - 1]) {
            best[position] = best[position - 1];
            sequence->selected[position] = sequence->selected[position - 1];
            position--;
        }
        best[position] = choice;
        sequence->selected[position] = expert;
        if (count < 16) count++;
    }
    double total = 0.0;
    for (unsigned rank = 0; rank < 16; rank++) total += (double)scores[sequence->selected[rank]];
    float inverse = (float)(1.0 / (total + 1e-20));
    for (unsigned rank = 0; rank < 16; rank++) sequence->weights[rank] = scores[sequence->selected[rank]] * inverse;
    return 1;
}

static int transformer_expert(Transformer *transformer, TransformerSequence *sequence, unsigned expert)
{
    if (!root_project(transformer->root, &sequence->root_scratch, expert, 0, sequence->latent, sequence->expert_gate) ||
        !root_project(transformer->root, &sequence->root_scratch, expert, 1, sequence->latent, sequence->expert_up)) return 0;
    transformer_activation(sequence->expert_gate, sequence->expert_up, TRANSFORMER_EXPERT);
    return root_project(transformer->root, &sequence->root_scratch, expert, 2, sequence->expert_gate, sequence->expert_down);
}

static int transformer_mix_experts(Transformer *transformer, TransformerSequence *sequence)
{
    memset(sequence->mixture, 0, sizeof sequence->mixture);
    for (unsigned rank = 0; rank < 16; rank++) {
        if (!transformer_expert(transformer, sequence, sequence->selected[rank])) return 0;
        for (unsigned coordinate = 0; coordinate < TRANSFORMER_LATENT; coordinate++)
            sequence->mixture[coordinate] = sequence->mixture[coordinate] + sequence->weights[rank] * sequence->expert_down[coordinate];
    }
    transformer_normalize(sequence->latent_norm, sequence->mixture, transformer->records[33].data, TRANSFORMER_LATENT);
    transformer_project(transformer, 32, sequence->latent_norm, sequence->routed);
    return 1;
}

static void transformer_shared(const Transformer *transformer, TransformerSequence *sequence)
{
    transformer_project(transformer, 34, sequence->postnorm, sequence->shared_gate);
    transformer_project(transformer, 35, sequence->postnorm, sequence->shared_up);
    transformer_activation(sequence->shared_gate, sequence->shared_up, TRANSFORMER_SHARED);
    transformer_project(transformer, 36, sequence->shared_gate, sequence->shared_output);
}

static int transformer_moe(Transformer *transformer, TransformerSequence *sequence)
{
    transformer_normalize(sequence->postnorm, sequence->aggregate, transformer->records[5].data, TRANSFORMER_WIDTH);
    if (!transformer_finite(sequence->postnorm, TRANSFORMER_WIDTH) || !transformer_route(transformer, sequence)) return 0;
    transformer_project(transformer, 31, sequence->postnorm, sequence->latent);
    if (!transformer_mix_experts(transformer, sequence)) return 0;
    transformer_shared(transformer, sequence);
    for (unsigned coordinate = 0; coordinate < TRANSFORMER_WIDTH; coordinate++) {
        float mixed = sequence->routed[coordinate] + sequence->shared_output[coordinate];
        sequence->residual[coordinate] = sequence->residual[coordinate] + mixed;
    }
    return 1;
}