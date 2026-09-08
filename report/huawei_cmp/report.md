| Dataset | Best MAX_RUNNING_REQUESTS | AISBench concurrency | Peak throughput/GPU |
| --- | ---: | ---: | ---: |
| dataset_3_5k_p0 | 64 | 64 | 170.62 tok/s/GPU |
| dataset_16k_p0 | 64 | 32 | 49.34 tok/s/GPU |
| dataset_128k_p90 | - | - | - |

## dataset_3_5k_p0

Cell: TTFT ms / TPOT ms / tok/s/GPU / SLA

| AISBench concurrency | MR=64 |
| ---: | --- |
| 4 | 1269.6 / 17.3 / 27.58 / PASS |
| 8 | 1874.2 / 19.5 / 48.06 / PASS |
| 12 | 3069.4 / 23.2 / 59.46 / FAIL_TPOT |
| 16 | 3251.3 / 23.3 / 78.14 / FAIL_TPOT |
| 20 | 4095.4 / 26.9 / 84.13 / FAIL_BOTH |
| 24 | 4441.0 / 26.7 / 100.72 / FAIL_BOTH |
| 28 | 5130.1 / 29.9 / 104.63 / FAIL_BOTH |
| 32 | 5566.3 / 29.8 / 118.87 / FAIL_BOTH |
| 36 | 5616.7 / 33.1 / 121.20 / FAIL_BOTH |
| 40 | 6270.7 / 32.9 / 133.91 / FAIL_BOTH |
| 44 | 6241.1 / 35.4 / 137.97 / FAIL_BOTH |
| 48 | 6044.2 / 35.6 / 149.55 / FAIL_BOTH |
| 52 | 6754.7 / 38.7 / 148.87 / FAIL_BOTH |
| 56 | 7238.3 / 38.7 / 159.52 / FAIL_BOTH |
| 60 | 7848.1 / 41.1 / 160.30 / FAIL_BOTH |
| 64 | 8463.3 / 40.8 / 170.62 / FAIL_BOTH |
| **Peak** | 170.62 @ c64 |

## dataset_16k_p0

Cell: TTFT ms / TPOT ms / tok/s/GPU / SLA

| AISBench concurrency | MR=64 |
| ---: | --- |
| 4 | 4339.3 / 19.8 / 20.74 / PASS |
| 8 | 8334.6 / 24.5 / 30.56 / FAIL_TPOT |
| 12 | 10677.2 / 32.6 / 34.79 / FAIL_BOTH |
| 16 | 12342.5 / 37.1 / 40.51 / FAIL_BOTH |
| 20 | 13784.8 / 45.5 / 42.25 / FAIL_BOTH |
| 24 | 14816.5 / 50.6 / 45.85 / FAIL_BOTH |
| 28 | 15860.2 / 59.0 / 46.69 / FAIL_BOTH |
| 32 | 16980.0 / 64.1 / 49.34 / FAIL_BOTH |
| 36 | 17578.8 / 74.4 / 47.49 / FAIL_BOTH |
| 40 | 24373.5 / 78.3 / 46.98 / FAIL_BOTH |
| 44 | 33267.8 / 78.6 / 48.13 / FAIL_BOTH |
| 48 | 43323.6 / 79.6 / 46.79 / FAIL_BOTH |
| 52 | - |
| 56 | - |
| 60 | - |
| 64 | - |
| **Peak** | 49.34 @ c32 |

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
