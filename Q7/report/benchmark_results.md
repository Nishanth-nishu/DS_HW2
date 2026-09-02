# Q7 Benchmark Analysis - Server Log Analytics

Source: `benchmark/results.csv`  |  process counts: 1 2 4 8  |  statistic: median over trials

`t_algo` = `t_compute` + `t_comm`, where `t_comm` = `t_dist` (Bcast header + Scatterv records) + `t_reduce` (Reduce scalars + Gatherv per-key maps).

### Input sizes

| Label | parameters |
|:--|:-:|
| small | N=100000 K=10 S=32 |
| medium | N=1000000 K=10 S=64 |
| keys_few | N=1000000 K=10 S=4 |
| keys_many | N=1000000 K=10 S=5000 |
| topk_large | N=1000000 K=100 S=128 |

### Runtime - median `t_algo` (seconds)

| Input size | P=1 | P=2 | P=4 | P=8 |
|:--|:-:|:-:|:-:|:-:|
| small | 0.0059 | 0.0058 | 0.0091 | 0.0462 |
| medium | 0.0492 | 0.0569 | 0.0517 | 0.2189 |
| keys_few | 0.0494 | 0.0418 | 0.0514 | 0.2141 |
| keys_many | 0.0617 | 0.0514 | 0.0654 | 0.2598 |
| topk_large | 0.0505 | 0.0571 | 0.0523 | 0.2154 |

### Speed-up  S(P) = T1 / TP

| Input size | P=1 | P=2 | P=4 | P=8 |
|:--|:-:|:-:|:-:|:-:|
| small | 1.00 | 1.01 | 0.64 | 0.13 |
| medium | 1.00 | 0.87 | 0.95 | 0.22 |
| keys_few | 1.00 | 1.18 | 0.96 | 0.23 |
| keys_many | 1.00 | 1.20 | 0.94 | 0.24 |
| topk_large | 1.00 | 0.89 | 0.97 | 0.23 |

### Efficiency  E(P) = S(P) / P

| Input size | P=1 | P=2 | P=4 | P=8 |
|:--|:-:|:-:|:-:|:-:|
| small | 100.0% | 50.7% | 16.1% | 1.6% |
| medium | 100.0% | 43.3% | 23.8% | 2.8% |
| keys_few | 100.0% | 59.0% | 24.0% | 2.9% |
| keys_many | 100.0% | 60.0% | 23.6% | 3.0% |
| topk_large | 100.0% | 44.3% | 24.1% | 2.9% |

### Communication share - `t_comm` as % of `t_algo`

Rising with P is expected: the Bcast of B moves n*p elements to every process regardless of P, while the computation per process falls as 1/P.

| Input size | P=1 | P=2 | P=4 | P=8 |
|:--|:-:|:-:|:-:|:-:|
| small | 72.2% | 84.6% | 89.9% | 98.9% |
| medium | 68.7% | 75.2% | 86.1% | 98.3% |
| keys_few | 69.0% | 81.6% | 86.6% | 98.3% |
| keys_many | 57.0% | 71.1% | 74.4% | 96.3% |
| topk_large | 66.9% | 74.6% | 85.6% | 98.2% |

### Phase breakdown (seconds, median)

| Size | P | t_dist | t_compute | t_reduce | t_algo |
|:--|:-:|:-:|:-:|:-:|:-:|
| small | 1 | 0.0041 | 0.0016 | 0.0001 | 0.0059 |
| small | 2 | 0.0041 | 0.0009 | 0.0008 | 0.0058 |
| small | 4 | 0.0074 | 0.0009 | 0.0008 | 0.0091 |
| small | 8 | 0.0442 | 0.0005 | 0.0015 | 0.0462 |
| medium | 1 | 0.0337 | 0.0154 | 0.0001 | 0.0492 |
| medium | 2 | 0.0418 | 0.0141 | 0.0010 | 0.0569 |
| medium | 4 | 0.0438 | 0.0072 | 0.0008 | 0.0517 |
| medium | 8 | 0.2135 | 0.0037 | 0.0016 | 0.2189 |
| keys_few | 1 | 0.0339 | 0.0153 | 0.0001 | 0.0494 |
| keys_few | 2 | 0.0331 | 0.0077 | 0.0010 | 0.0418 |
| keys_few | 4 | 0.0438 | 0.0069 | 0.0008 | 0.0514 |
| keys_few | 8 | 0.2088 | 0.0037 | 0.0016 | 0.2141 |
| keys_many | 1 | 0.0334 | 0.0265 | 0.0018 | 0.0617 |
| keys_many | 2 | 0.0329 | 0.0148 | 0.0037 | 0.0514 |
| keys_many | 4 | 0.0437 | 0.0168 | 0.0049 | 0.0654 |
| keys_many | 8 | 0.2226 | 0.0097 | 0.0275 | 0.2598 |
| topk_large | 1 | 0.0337 | 0.0167 | 0.0002 | 0.0505 |
| topk_large | 2 | 0.0415 | 0.0145 | 0.0011 | 0.0571 |
| topk_large | 4 | 0.0438 | 0.0075 | 0.0010 | 0.0523 |
| topk_large | 8 | 0.2095 | 0.0040 | 0.0019 | 0.2154 |

### Plots written

- `benchmark/plots/time.svg`
- `benchmark/plots/speedup.svg`
- `benchmark/plots/efficiency.svg`
- `benchmark/plots/comm_fraction.svg`
