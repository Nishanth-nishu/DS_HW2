# Q6 Benchmark Analysis - Connected Components

Source: `benchmark/results.csv`  |  process counts: 1 2 4 8  |  statistic: median over trials

`t_algo` = `t_compute` + `t_comm` (excludes file reading and distribution).

### Runtime - median `t_algo` (seconds)

| Input size | P=1 | P=2 | P=4 | P=8 |
|:--|:-:|:-:|:-:|:-:|
| small | 0.0014 | 0.0018 | 0.0013 | 0.0115 |
| medium | 0.0071 | 0.0054 | 0.0061 | 0.0176 |
| large | 0.0135 | 0.0104 | 0.0123 | 0.0238 |
| verylarge | 0.0143 | 0.0083 | 0.0104 | 0.0202 |
| path | 0.0017 | 0.0033 | 0.0055 | 0.0260 |
| shuffled | 0.0040 | 0.0154 | 0.0198 | 0.0902 |
| chain | 0.0040 | 0.0049 | 0.0058 | 0.0249 |

### Speed-up  S(P) = T1 / TP

| Input size | P=1 | P=2 | P=4 | P=8 |
|:--|:-:|:-:|:-:|:-:|
| small | 1.00 | 0.74 | 1.04 | 0.12 |
| medium | 1.00 | 1.32 | 1.16 | 0.41 |
| large | 1.00 | 1.30 | 1.09 | 0.57 |
| verylarge | 1.00 | 1.73 | 1.37 | 0.70 |
| path | 1.00 | 0.52 | 0.32 | 0.07 |
| shuffled | 1.00 | 0.26 | 0.20 | 0.04 |
| chain | 1.00 | 0.82 | 0.69 | 0.16 |

### Efficiency  E(P) = S(P) / P

| Input size | P=1 | P=2 | P=4 | P=8 |
|:--|:-:|:-:|:-:|:-:|
| small | 100.0% | 37.0% | 25.9% | 1.5% |
| medium | 100.0% | 66.0% | 28.9% | 5.1% |
| large | 100.0% | 65.0% | 27.4% | 7.1% |
| verylarge | 100.0% | 86.3% | 34.3% | 8.8% |
| path | 100.0% | 26.0% | 7.9% | 0.8% |
| shuffled | 100.0% | 13.0% | 5.1% | 0.6% |
| chain | 100.0% | 40.8% | 17.3% | 2.0% |

### Communication share - `t_comm` as % of `t_algo`

The Allreduce moves V ints per round regardless of P, while the per-rank edge work falls as 1/P.

| Input size | P=1 | P=2 | P=4 | P=8 |
|:--|:-:|:-:|:-:|:-:|
| small | 0.1% | 8.0% | 22.3% | 94.1% |
| medium | 0.1% | 19.0% | 17.9% | 81.3% |
| large | 0.0% | 16.7% | 17.3% | 71.9% |
| verylarge | 0.0% | 10.7% | 14.3% | 74.1% |
| path | 0.2% | 40.5% | 40.1% | 89.9% |
| shuffled | 0.1% | 18.3% | 22.7% | 85.6% |
| chain | 0.1% | 26.9% | 31.5% | 87.8% |

### Convergence rounds

Deterministic and machine-independent: these depend on the graph and the partition, not on timing.

| Input size | P=1 | P=2 | P=4 | P=8 |
|:--|:-:|:-:|:-:|:-:|
| small | 2 | 2 | 2 | 2 |
| medium | 2 | 2 | 2 | 2 |
| large | 2 | 2 | 2 | 2 |
| verylarge | 2 | 2 | 2 | 2 |
| path | 2 | 3 | 3 | 3 |
| shuffled | 2 | 9 | 9 | 10 |
| chain | 2 | 3 | 3 | 3 |

### Plots written

- `benchmark/plots/time.svg`
- `benchmark/plots/speedup.svg`
- `benchmark/plots/efficiency.svg`
- `benchmark/plots/comm_fraction.svg`
