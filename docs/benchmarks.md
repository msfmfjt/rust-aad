# P0 benchmark plan (no results yet)

Baseline on `msfmfjt/rust-pricing-library` commit `f82a8b2c3fb2231959ae4104ec30778cdba683a1`. Measure *both* original and independent versions with identical inputs and workloads.

| Workload | Cases |
|---|---|
| Vanilla/Asian payoff | 10k, 100k, 1M paths; 1, 12, 120 observations |
| Branch-heavy barrier/autocall | rare, 50/50 and frequent branch selection; invalid inactive branches |
| Bergomi path | 1F/2F, 120 steps, price+all first-order risks |
| LSV calibration | finite particle calibration VJP, external to generic tape |
| Hull-White hybrid | path+calibration+market mapping, application-only |

Collect graph compilation time, initial workspace allocation, warm forward-only, warm forward + all-gradient VJP, per-call allocations, peak RSS, confidence bands and CPU features, with at least ten warmed iterations. Separate model-specific costs from generic graph cost. Require exact replay agreement where the original contracts require it; otherwise document numeric tolerances. Vendor AADC/XAD numbers are third-party data until both run on the same host with the same workload and equivalent differentiation semantics.

## P0 gate

No performance claim until actual measured baseline records exist. Automated benchmarks are a next deliverable; none have been run for this bootstrap.
