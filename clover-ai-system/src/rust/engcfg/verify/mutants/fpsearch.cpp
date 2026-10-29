// fpsearch: finds inputs of print_summary on which the one-ulp rewrites of its floating-point expressions print differently from the original expressions.
//   fpsearch pct | gb | ms
// pct: (hit, need) integers: original `100.0 * h / need` against `h / need * 100.0` and `100.0 * (h / need)`, printed with %.1f
// gb : (from_drive, expert_raw): original `d * raw / 1e9` against `d / 1e9 * raw` and `d * raw * 1e-9`, printed with %.1f
// ms : (n_calls, t_calls): original `1000.0 * t / n` against `t / n * 1000.0`, `1000.0 * (t / n)` and `t / (n / 1000.0)`, printed with %.1f
// Only exact decimal ties can print differently (the true value is then exactly on a rounding boundary and the two roundings of the double can fall on different sides).
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
static std::string f1(double v) { char b[400]; snprintf(b, sizeof b, "%.1f", v); return b; }
static int shown = 0;
int main(int argc, char **argv)
{
    if (argc < 2) return 2;
    const std::string m = argv[1];
    uint64_t tried = 0, diff_a = 0, diff_b = 0, diff_c = 0;
    if (m == "pct") {
        for (uint64_t need = 1; need <= 200000; need++) {
            uint64_t g = need, h = 2000; while (h) { uint64_t t = g % h; g = h; h = t; }   // gcd(need, 2000)
            const uint64_t step = need / g;
            for (uint64_t hit = step; hit <= 3 * need; hit += step) {
                const double H = (double)hit, N = (double)need;
                const std::string o = f1(100.0 * H / N), a = f1(H / N * 100.0), b = f1(100.0 * (H / N));
                tried++; diff_a += o != a; diff_b += o != b;
                if ((o != a || o != b) && shown < 6) { shown++; printf("pct hit=%llu need=%llu original=%s h/n*100=%s 100*(h/n)=%s\n", (unsigned long long)hit, (unsigned long long)need, o.c_str(), a.c_str(), b.c_str()); }
            }
        }
        printf("pct: %llu exact-tie-capable pairs, %llu differ for h/n*100, %llu for 100*(h/n)\n", (unsigned long long)tried, (unsigned long long)diff_a, (unsigned long long)diff_b);
    } else if (m == "gb") {
        const long raws[] = { 17547264, 1000, 4096, 1 << 20 };
        for (long raw : raws) for (uint64_t d = 0; d <= 3000000; d++) {
            const double D = (double)d, R = (double)raw;
            const std::string o = f1(D * R / 1e9), a = f1(D / 1e9 * R), b = f1(D * R * 1e-9);
            tried++; diff_a += o != a; diff_b += o != b;
            if ((o != a || o != b) && shown < 6) { shown++; printf("gb from_drive=%llu raw=%ld original=%s d/1e9*raw=%s d*raw*1e-9=%s\n", (unsigned long long)d, raw, o.c_str(), a.c_str(), b.c_str()); }
        }
        printf("gb: %llu inputs, %llu differ for d/1e9*raw, %llu for d*raw*1e-9\n", (unsigned long long)tried, (unsigned long long)diff_a, (unsigned long long)diff_b);
    } else if (m == "ms") {
        for (int n = 1; n <= 3000; n++) for (int j = 0; j <= 4000; j++) {
            const double N = (double)n, T = ((double)j * 10 + 5) / 100.0 * N / 1000.0;   // t chosen so that 1000 * t / n sits on a decimal tie k + 0.05
            const std::string o = f1(1000.0 * T / N), a = f1(T / N * 1000.0), b = f1(1000.0 * (T / N)), c = f1(T / (N / 1000.0));
            tried++; diff_a += o != a; diff_b += o != b; diff_c += o != c;
            if (o != a && shown < 8) { shown++; printf("ms n_calls=%d t_calls=%.17g original=%s t/n*1000=%s 1000*(t/n)=%s t/(n/1000)=%s\n", n, T, o.c_str(), a.c_str(), b.c_str(), c.c_str()); }
        }
        printf("ms: %llu inputs, %llu differ for t/n*1000, %llu for 1000*(t/n), %llu for t/(n/1000)\n", (unsigned long long)tried, (unsigned long long)diff_a, (unsigned long long)diff_b, (unsigned long long)diff_c);
    } else return 2;
    return 0;
}
