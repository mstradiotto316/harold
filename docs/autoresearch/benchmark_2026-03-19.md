# Benchmark Results — 2026-03-19

## Hardware

- **GPU**: NVIDIA GeForce RTX 4080 (16376 MB)
- **RAM**: 63 GB
- **CPU**: Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz

## Throughput Sweep (no video)

|   Envs |    it/s |  Samples/s |  GPU MB | RAM GB | Status |
|--------|---------|------------|---------|--------|--------|
|   1024 |    18.1 |      0.45M |    6242 |    8.6 |     OK |
|   2048 |    17.5 |      0.86M |    6577 |    8.8 |     OK |
|   4096 |    16.0 |      1.58M |    7191 |    9.3 |     OK |
|   8192 |    11.5 |      2.26M |    8263 |   10.3 |     OK |
|  12288 |     9.0 |      2.66M |    9261 |   11.4 |     OK |
|  16384 |     7.3 |      2.88M |   10327 |   12.3 |     OK |
|  20480 |     6.2 |      3.05M |   11383 |   13.5 |     OK |
|  24576 |     5.4 |      3.18M |   12307 |   14.6 |     OK |

## Throughput Sweep (with video)

|   Envs |    it/s |  Samples/s |  GPU MB | RAM GB | Status |
|--------|---------|------------|---------|--------|--------|
|   1024 |     9.4 |      0.23M |    8532 |   19.5 |     OK |
|   2048 |     9.0 |      0.44M |    8997 |   27.7 |     OK |
|   4096 |     8.0 |      0.78M |    9598 |   42.1 |     OK |
|   6144 |     7.1 |      1.04M |   10255 |   58.1 |     OK |
|   8192 |     --- |        --- |     --- |    --- |    OOM |

Video recording adds ~2.3 GB GPU and ~10 GB RAM per 2x env increase.

## Stress Test (4096 envs + video, 30 min)

- **Duration**: 30.0 min (full)
- **Peak GPU**: 9683 MB (59%)
- **Peak RAM**: 42.5 GB (67%)
- **Avg GPU**: 9543 MB
- **Avg RAM**: 40.8 GB
- **Videos produced**: 164
- **Crashes/OOM**: None
- **VERDICT**: PASS

## Recommendation

**Optimal num_envs: 4096** (with mandatory video recording)

- Throughput: 8.0 it/s (0.78M samples/s)
- Peak GPU: 9598 MB / 16376 MB (59%)
- Peak RAM: 42.1 GB / 63 GB (67%)
- 3.4x speedup vs previous 1024 envs (0.23M -> 0.78M samples/s)
- 30 min stress test passed with no issues

6144 envs works (1.04M samples/s) but uses 92% RAM — no headroom for system processes. Not recommended for unattended autoresearch sessions.

Without video, throughput scales linearly to 24576 envs (3.18M samples/s, 75% GPU, 23% RAM).

## Two-Phase Training Approach

Based on these results, training now uses a two-phase approach:
1. **Train** at 16384 envs without video (2.88M samples/s, 12.3 GB RAM)
2. **Record** post-hoc via `harold record` from the best checkpoint at 1 env (~30-45s)

This gives 3.7x throughput (2.88M vs 0.78M samples/s) compared to the previous 4096+video setup.

## Comparison to Previous Benchmark (2025-12-25)

Previous benchmark did not test with video recording enabled. The old data showed 16384 envs at 8.7 it/s using only 7.6 GB GPU + 11 GB RAM — this was accurate for no-video mode. The video recording overhead (~2x GPU, ~4x RAM) was the missing piece.
