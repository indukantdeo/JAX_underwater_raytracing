# Validation Report: dickins

- Frequency: `230.00 Hz`
- Source depth: `18.00 m`
- Requested beams: `21`
- Actual beams used: `21`
- Bellhop-recommended beams: `482`
- Auto beam count: `False`
- Source amplitude range: `[1.000000, 1.000000]`
- Step size: `5.00 m`
- Solver runtime: `10.871493 s`
- Launch fan time: `0.392744 s`
- Ray rollout time: `8.922737 s`
- Accumulation time: `1.110271 s`
- Accumulation backend: `windowed`
- Precision: `float64`
- Beam chunk size requested: `8`
- Beam chunk size used: `8`
- Status: `validated_against_reference`

## Field Metrics

- rmse: `76.516932`
- mae: `65.559550`
- max_abs_error: `231.931190`
- p95_abs_error: `159.999985`
- bias: `65.420167`
- reference_mean: `106.968602`
- candidate_mean: `172.388769`
- correlation: `0.422760`

## Slice Metrics

### 0.0m

- rmse: `73.659143`
- mae: `63.359316`
- max_abs_error: `211.608551`
- p95_abs_error: `138.422306`
- bias: `63.359316`
- reference_mean: `107.867846`
- candidate_mean: `171.227161`
- correlation: `0.627713`

### 200.0m

- rmse: `84.972585`
- mae: `76.706455`
- max_abs_error: `227.832581`
- p95_abs_error: `142.180020`
- bias: `76.706455`
- reference_mean: `94.770987`
- candidate_mean: `171.477442`
- correlation: `0.630118`

### 1000.0m

- rmse: `77.449286`
- mae: `68.030012`
- max_abs_error: `208.801041`
- p95_abs_error: `140.055645`
- bias: `68.030012`
- reference_mean: `102.015077`
- candidate_mean: `170.045088`
- correlation: `0.559248`
