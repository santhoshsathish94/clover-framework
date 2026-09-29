static unsigned long long histogram_decoder_checks, histogram_invalid_checks;

static int histogram_enabled(void)
{
    static int enabled = -1;
    if (enabled < 0) {
        const char *setting = getenv("K3_HISTOGRAM_RANK");
        enabled = setting ? atoi(setting) : 0;
        if (enabled < 0 || enabled > 1) die("K3_HISTOGRAM_RANK must be 0 or 1");
    }
    return enabled;
}

static uint64_t histogram_choose(unsigned total, unsigned count)
{
    if (count > total) return 0;
    if (count > total - count) count = total - count;
    uint64_t value = 1;
    for (unsigned factor = 1; factor <= count; factor++)
        value = value * (total - count + factor) / factor;
    return value;
}

static int histogram_encode(const unsigned *counts, unsigned total, unsigned alphabet, uint64_t *rank)
{
    if (total > 32 || alphabet < 2 || alphabet > 16) return 0;
    unsigned sum = 0;
    for (unsigned code = 0; code < alphabet; code++) {
        if (counts[code] > total) return 0;
        sum += counts[code];
        if (sum > total) return 0;
    }
    if (sum != total) return 0;
    unsigned prefix = 0;
    *rank = 0;
    for (unsigned ordinal = 1; ordinal < alphabet; ordinal++) {
        prefix += counts[ordinal - 1];
        *rank += histogram_choose(prefix + ordinal - 1, ordinal);
    }
    return 1;
}

static int histogram_decode(uint64_t rank, unsigned total, unsigned alphabet, unsigned *counts)
{
    if (total > 32 || alphabet < 2 || alphabet > 16) return 0;
    const unsigned width = total + alphabet - 1;
    if (rank >= histogram_choose(width, alphabet - 1)) return 0;
    unsigned bars[15];
    int upper = (int)width - 1;
    for (unsigned ordinal = alphabet - 1; ordinal > 0; ordinal--) {
        int position = upper;
        while (position >= 0 && histogram_choose((unsigned)position, ordinal) > rank) position--;
        if (position < 0) return 0;
        bars[ordinal - 1] = (unsigned)position;
        rank -= histogram_choose((unsigned)position, ordinal);
        upper = position - 1;
    }
    if (rank) return 0;
    counts[0] = bars[0];
    for (unsigned code = 1; code + 1 < alphabet; code++)
        counts[code] = bars[code] - bars[code - 1] - 1;
    counts[alphabet - 1] = width - 1 - bars[alphabet - 2];
    return 1;
}

static void histogram_roundtrip(const unsigned *counts, unsigned total, unsigned alphabet)
{
    uint64_t rank;
    unsigned decoded[16];
    if (!histogram_encode(counts, total, alphabet, &rank) ||
        !histogram_decode(rank, total, alphabet, decoded) ||
        memcmp(counts, decoded, alphabet * sizeof(unsigned))) die("histogram roundtrip failed");
    histogram_decoder_checks++;
}

static void histogram_selfcheck(void)
{
    for (unsigned first = 0; first < 7; first++)
        for (unsigned second = first + 1; second < 8; second++)
            for (unsigned third = second + 1; third < 9; third++) {
                const unsigned counts[4] = {first, second - first - 1, third - second - 1, 8 - third};
                histogram_roundtrip(counts, 6, 4);
            }
    for (unsigned code = 0; code < 16; code++) {
        unsigned counts[16] = {0};
        counts[code] = 32;
        histogram_roundtrip(counts, 32, 16);
    }
    const uint64_t limit = histogram_choose(47, 15);
    const uint64_t ranks[5] = {0, 1, limit / 2, limit - 2, limit - 1};
    for (unsigned sample = 0; sample < 5; sample++) {
        unsigned counts[16];
        uint64_t encoded;
        if (!histogram_decode(ranks[sample], 32, 16, counts) ||
            !histogram_encode(counts, 32, 16, &encoded) || encoded != ranks[sample])
            die("histogram boundary check failed");
        histogram_decoder_checks++;
    }
    unsigned counts[16] = {0};
    uint64_t ignored;
    if (histogram_decode(limit, 32, 16, counts)) die("invalid histogram rank accepted");
    histogram_invalid_checks++;
    if (histogram_encode(counts, 32, 16, &ignored)) die("short histogram accepted");
    histogram_invalid_checks++;
    counts[0] = 33;
    if (histogram_encode(counts, 32, 16, &ignored)) die("excess histogram accepted");
    histogram_invalid_checks++;
    if (histogram_decode(0, 32, 1, counts)) die("invalid histogram alphabet accepted");
    histogram_invalid_checks++;
}