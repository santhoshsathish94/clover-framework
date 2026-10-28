---
{
  "schema_version": "1.0",
  "id": "MEM-OBS-FLIPT-METRICS-EXPORTERS-CYCLE1-001",
  "title": "Observation: Flipt OTLP & Prometheus Metrics Exporters Cycle 1 Compiler & Schema Verification",
  "entry_type": "observation",
  "epistemic_status": "observed",
  "status": "active",
  "concepts": [
    "flipt-metrics-exporters",
    "otlp-metrics-exporter",
    "prometheus-metrics-exporter",
    "viper-decode-hooks",
    "url-parse-host-port-scheme"
  ],
  "tags": [
    "flipt",
    "golang",
    "swebench-pro-v2",
    "2ca5dfb3",
    "cycle-1",
    "observation",
    "internal/config/metrics.go",
    "internal/config/config.go",
    "internal/metrics/metrics.go"
  ],
  "created_at": "2026-10-28T19:00:00Z",
  "updated_at": "2026-10-28T19:00:05Z",
  "provenance": {
    "source_type": "experiment",
    "source_ref": "swebench-pro-v2:flipt-2ca5dfb3:cycle1",
    "actor": "human_and_assistant",
    "recorded_at": "2026-10-28T19:00:00Z"
  },
  "relationships": [],
  "revisions": [
    {
      "timestamp": "2026-10-28T19:00:00Z",
      "actor": "human_and_assistant",
      "action": "staged",
      "reason": "Staged Cycle 1 compiler & test failure observation on flipt-2ca5dfb3"
    },
    {
      "timestamp": "2026-10-28T19:00:05Z",
      "actor": "clover_cycle_verifier",
      "action": "promoted",
      "reason": "Promoted Cycle 1 observation for flipt-2ca5dfb3"
    }
  ]
}
---

## Summary
In Cycle 1 on `flipt-io/flipt` (`instance_flipt-io__flipt-2ca5dfb3513e4e786d2b037075617cccc286d5c3`), Clover read `internal/metrics/metrics.go` (`5,636` chars) and attempted to read `internal/config/metrics.go` (`Error: File /app/internal/config/metrics.go does not exist`). Running the locked SWE-Bench Pro V2 verifier (`/tests/test.sh`) exposed the exact compiler and unit-test contracts across `internal/config` and `internal/metrics`:

## Content
1. `internal/config/metrics.go` is a new file defining `MetricsConfig` (`Enabled bool`, `Exporter MetricsExporter`, `OTLP OTLPMetricsConfig`), `MetricsExporter` (`MetricsPrometheus`, `MetricsOTLP`), `stringToMetricsExporter`, `OTLPMetricsConfig` (`Endpoint string`, `Headers map[string]string`), `setDefaults` (`enabled: true`, `exporter: MetricsPrometheus`), `validate()` (`errFieldRequired("metrics.otlp.endpoint")` when `Exporter == MetricsOTLP` and `OTLP.Endpoint == ""`), and `IsZero() bool { return !c.Enabled }`.
2. `internal/config/config.go` must wire `Metrics MetricsConfig` on `Config`, register `stringToEnumHookFunc(stringToMetricsExporter)` in `DecodeHooks`, and set `Metrics: MetricsConfig{Enabled: true, Exporter: MetricsPrometheus}` in `Default()`.
3. `internal/metrics/metrics.go` must expose `metricExpOnce sync.Once`, `metricExp sdkmetric.Reader`, `metricExpFunc func(context.Context) error`, `metricExpErr error`, and `GetExporter(ctx context.Context, cfg *config.MetricsConfig) (sdkmetric.Reader, func(context.Context) error, error)`. In `getExporter`, when `u, err := url.Parse(cfg.OTLP.Endpoint)` parses a bare `host:port` like `localhost:4317`, `u.Scheme` is `"localhost"`, so `default:` must fall back to `otlpmetricgrpc.New(ctx, otlpmetricgrpc.WithEndpoint(cfg.OTLP.Endpoint), ...)` just like `internal/tracing/tracing.go`.
