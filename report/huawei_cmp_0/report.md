| Dataset | Best MAX_RUNNING_REQUESTS | AISBench concurrency | Peak throughput/GPU |
| --- | ---: | ---: | ---: |
| dataset_3_5k_p0 | 64 | 36 | 115.76 tok/s/GPU |
| dataset_16k_p0 | - | - | - |
| dataset_128k_p90 | - | - | - |

## dataset_3_5k_p0

Cell: TTFT ms / TPOT ms / tok/s/GPU / SLA

| AISBench concurrency | MR=64 |
| ---: | --- |
| 4 | 3517.6 / 14.5 / 29.83 / PASS |
| 8 | 2458.6 / 17.6 / 52.18 / PASS |
| 12 | 3333.6 / 19.8 / 68.17 / PASS |
| 16 | 4028.2 / 22.2 / 80.61 / FAIL_BOTH |
| 20 | 4792.2 / 26.1 / 85.53 / FAIL_BOTH |
| 24 | 5531.2 / 26.9 / 98.26 / FAIL_BOTH |
| 28 | 6279.2 / 29.7 / 103.63 / FAIL_BOTH |
| 32 | 6999.3 / 30.3 / 114.74 / FAIL_BOTH |
| 36 | 7750.5 / 33.8 / 115.76 / FAIL_BOTH |
| 40 | - |
| 44 | - |
| 48 | - |
| 52 | - |
| 56 | - |
| 60 | - |
| 64 | - |
| **Peak** | 115.76 @ c36 |

## dataset_16k_p0

Cell: TTFT ms / TPOT ms / tok/s/GPU / SLA

| AISBench concurrency | MR=64 |
| ---: | --- |
| 4 | - |
| 8 | - |
| 12 | - |
| 16 | - |
| 20 | - |
| 24 | - |
| 28 | - |
| 32 | - |
| 36 | - |
| 40 | - |
| 44 | - |
| 48 | - |
| 52 | - |
| 56 | - |
| 60 | - |
| 64 | - |
| **Peak** | - |

## dataset_128k_p90

Cell: TTFT ms / TPOT ms / tok/s/GPU / SLA

| AISBench concurrency | MR=64 |
| ---: | --- |
| 4 | - |
| 8 | - |
| 12 | - |
| 16 | - |
| 20 | - |
| 24 | - |
| 28 | - |
| 32 | - |
| 36 | - |
| 40 | - |
| 44 | - |
| 48 | - |
| 52 | - |
| 56 | - |
| 60 | - |
| 64 | - |
| **Peak** | - |
