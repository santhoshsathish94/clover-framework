#include "reverse-vector.h"

static int basis_enabled(void)
{
    static int enabled = -1;
    if (enabled < 0) {
        const char *setting = getenv("K3_BASIS_ROUTER");
        enabled = setting ? atoi(setting) : 0;
        if (enabled < 0 || enabled > 2) die("K3_BASIS_ROUTER must be 0, 1 or 2");
    }
    return enabled;
}

static int basis_count;
static const float *basis_sources[16];
static float basis_coefficients[16];
static float basis_norm_weights[E];
static long double basis_inverse;
static float basis_rounded_inverse;
static int basis_prepared;
static float basis_rows[NEXP][4];
static int basis_row_seen[NEXP];

static void basis_capture(float *const *sources, const float *coefficients, int count)
{
    if (!basis_enabled() || basis_count || cur_L != 84 || count != 9) return;
    basis_count = count;
    for (int source = 0; source < count; source++) {
        basis_sources[source] = sources[source];
        basis_coefficients[source] = coefficients[source];
    }
}

static float basis_component(int coordinate)
{
    if (reverse_vector_ready) return reverse_vector_values[coordinate];
    float total = 0.0f;
    for (int source = 0; source < basis_count; source++) {
        const float contribution = basis_coefficients[source] * basis_sources[source][coordinate];
        total = total + contribution;
    }
    return total;
}

static void basis_prepare(const float *weights)
{
    if (basis_count != 9 || basis_prepared) die("basis preparation state");
    memcpy(basis_norm_weights, weights, sizeof(basis_norm_weights));
    if (basis_enabled() == 2) {
        for (int coordinate = 0; coordinate < E; coordinate++) reverse_vector_values[coordinate] = basis_component(coordinate);
        reverse_vector(reverse_vector_values);
        reverse_vector_ready = 1;
        double square_sum = 0.0;
        for (int coordinate = 0; coordinate < E; coordinate++) {
            const float value = basis_component(coordinate);
            square_sum += (double)value * (double)value;
        }
        basis_rounded_inverse = (float)(1.0 / sqrt(square_sum / (double)E + EPS5));
        basis_prepared = 1;
        return;
    }
    long double norm_squared = 0.0L;
    for (int first = 0; first < basis_count; first++) {
        for (int second = first; second < basis_count; second++) {
            long double gram_entry = 0.0L;
            for (int coordinate = 0; coordinate < E; coordinate++)
                gram_entry += (long double)basis_sources[first][coordinate] *
                              (long double)basis_sources[second][coordinate];
            long double term = (long double)basis_coefficients[first] *
                               (long double)basis_coefficients[second] * gram_entry;
            norm_squared += first == second ? term : 2.0L * term;
        }
    }
    if (!isfinite(norm_squared) || norm_squared < 0.0L) die("basis Gram norm invalid");
    basis_inverse = 1.0L / sqrtl(norm_squared / (long double)E + (long double)EPS5);
    basis_prepared = 1;
}

static float basis_project(const float *weights)
{
    if (!basis_prepared) die("basis projection before preparation");
    if (basis_enabled() == 2) {
        double projection = 0.0;
        for (int coordinate = 0; coordinate < E; coordinate++) {
            const float aggregate = basis_component(coordinate);
            const float weighted = basis_norm_weights[coordinate] * aggregate;
            const float normalized = weighted * basis_rounded_inverse;
            projection += (double)weights[coordinate] * (double)normalized;
        }
        return (float)projection;
    }
    long double projection = 0.0L;
    for (int source = 0; source < basis_count; source++) {
        long double transformed_source = 0.0L;
        for (int coordinate = 0; coordinate < E; coordinate++)
            transformed_source += (long double)weights[coordinate] *
                                  (long double)basis_norm_weights[coordinate] *
                                  (long double)basis_sources[source][coordinate];
        projection += (long double)basis_coefficients[source] * transformed_source;
    }
    return (float)(projection * basis_inverse);
}

static void basis_report(void)
{
    const char *path = getenv("K3_BASIS_REPORT");
    if (!path) die("K3_BASIS_REPORT required for candidate");
    FILE *report = fopen(path, "wb");
    if (!report) die("basis report open");
    const uint32_t header[4] = {0x31505242, NEXP, 84, 0};
    if (fwrite(header, sizeof(uint32_t), 4, report) != 4) die("basis report header");
    for (int expert = 0; expert < NEXP; expert++) {
        if (!basis_row_seen[expert]) die("basis projection row missing");
        if (fwrite(basis_rows[expert], sizeof(float), 4, report) != 4)
            die("basis report row");
    }
    if (fclose(report) != 0) die("basis report close");
        printf("basis replacement: layer 84 position 0, %d sources, %d router rows, %s\n",
            basis_count, NEXP, basis_enabled() == 2 ? "rounded-expression" : "Gram form");
}