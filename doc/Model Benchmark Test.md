# Model Benchmark Test

## Benchmark Methodology

Increase the request rate while monitoring latency. Identify the maximum sustainable request rate and output-token throughput that satisfies both TTFT and TPOT SLA.

## Test Overview

**Test Category:** Inference Benchmark

### GLM5.2 — W8A8

| Item | Configuration |
| --- | --- |
| Model | GLM5.2 |
| NPUs | 32 |
| Deployment | Prefill/decode disaggregation |
| Serving configuration | W8A8 |

#### Test Cases

| Input Length | Output Length | Prefix Cache Hit Rate |
| ---: | ---: | ---: |
| 128K | 1K | 90% |
| 16K | 1K | 0% |
| 3.5K | 1.5K | 0% |

#### Stress Test Configuration

- **Latency SLA:** TTFT <= 20 s (128K), TTFT <= 10 s (16K), TTFT <= 4 s (3.5K); TPOT <= 20 ms.
- **Primary KPI:** Maximum output-token throughput per NPU under SLA.

### GLM5.2 — W4A4

| Item | Configuration |
| --- | --- |
| Model | GLM5.2 |
| NPUs | 32 |
| Deployment | Prefill/decode disaggregation |
| Serving configuration | W4A4 |

#### Test Cases

| Input Length | Output Length | Prefix Cache Hit Rate |
| ---: | ---: | ---: |
| 128K | 1K | 90% |
| 16K | 1K | 0% |
| 3.5K | 1.5K | 0% |

#### Stress Test Configuration

- **Latency SLA:** TTFT <= 20 s (128K), TTFT <= 10 s (16K), TTFT <= 4 s (3.5K); TPOT <= 20 ms.
- **Primary KPI:** Maximum output-token throughput per NPU under SLA.

## POC Test Results

The following result metrics are required:

- Total output throughput
- Single-NPU output throughput

| Serving Configuration | Input / Output | Prefix Cache Hit Rate | Average TTFT | Average TPOT | Total Output Throughput | Single-NPU Output Throughput | SLA Met |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| W8A8 | 128K / 1K | 90% | TBD | TBD | TBD | TBD | TBD |
| W8A8 | 16K / 1K | 0% | TBD | TBD | TBD | TBD | TBD |
| W8A8 | 3.5K / 1.5K | 0% | TBD | TBD | TBD | TBD | TBD |
| W4A4 | 128K / 1K | 90% | TBD | TBD | TBD | TBD | TBD |
| W4A4 | 16K / 1K | 0% | TBD | TBD | TBD | TBD | TBD |
| W4A4 | 3.5K / 1.5K | 0% | TBD | TBD | TBD | TBD | TBD |
