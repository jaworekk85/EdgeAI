# Public run storage — planned

Completed public runs will use immutable directories:

```text
results/<experiment>/<UTC timestamp>_<short-hash>/
  config.yaml
  device.json
  metrics.json
  traces/
  plots/
```

The Phase 1 store must create each directory exclusively and refuse reuse.
Generated runs are ignored by Git by default; this does not replace the immutability
rule. Deliberate publication/export workflows may be added later. Private-derived
results belong under `private/results/` and must never be routed here.
