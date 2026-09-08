| Dataset | Best MAX_RUNNING_REQUESTS | AISBench concurrency | Peak throughput/GPU |
| --- | ---: | ---: | ---: |
| dataset_3_5k_p0 | 16 | 40 | 73.88 tok/s/GPU |

## dataset_3_5k_p0

Cell: TTFT ms / TPOT ms / tok/s/GPU / SLA

| AISBench concurrency | MR=16 | MR=32 | MR=64 |
| ---: | --- | --- | --- |
| 4 | 1464.2 / 14.5 / 32.40 / PASS | - | - |
| 8 | 2432.2 / 17.5 / 52.49 / PASS | - | - |
| 12 | 3263.7 / 19.8 / 68.55 / PASS | - | - |
| 16 | 6078.6 / 22.0 / 69.33 / FAIL_BOTH | - | - |
| 20 | 15660.4 / 22.2 / 69.85 / FAIL_BOTH | - | - |
| 24 | 24036.6 / 22.1 / 71.35 / FAIL_BOTH | - | - |
| 28 | 32698.3 / 22.1 / 72.38 / FAIL_BOTH | - | - |
| 32 | 41633.2 / 22.1 / 73.32 / FAIL_BOTH | - | - |
| 36 | 50788.7 / 22.3 / 73.27 / FAIL_BOTH | - | - |
| 40 | 59878.3 / 22.3 / 73.88 / FAIL_BOTH | - | - |
| 44 | - | - | - |
| 48 | - | - | - |
| 52 | - | - | - |
| 56 | - | - | - |
| 60 | - | - | - |
| 64 | - | - | - |
| 170 | - | - | - |
| **Peak** | 73.88 @ c40 | - | - |
