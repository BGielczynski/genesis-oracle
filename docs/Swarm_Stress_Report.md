# Swarm Stress Test & Profiling Report

## Overview
This report documents the automated analysis of the JAX Business Revenue Model, performed by Subagent-Alpha (Stress-Tester) and Subagent-Beta (Profiler).

## Part 1: Stress Test Analysis (Subagent-Alpha)
The stress test identifies parameter combinations where the 95% Value-at-Risk (VaR) becomes negative.

### Breaking Points Summary
- **Total Scenarios Tested**: 432
- **Breaking Points (VaR 95% < 0)**: 419

### Top Critical Scenarios
| Demand Mean | Demand Std | Cost Mu | Cost Sigma | Expected Rev | VaR 95% |
|-------------|------------|---------|------------|--------------|---------|
| 40 | 30 | 4.5 | 1.2 | -156.74 | -638.39 |
| 40 | 40 | 4.5 | 1.2 | -148.29 | -632.20 |
| 60 | 40 | 4.5 | 1.2 | -147.57 | -631.28 |

**Analysis**: Scenarios with low demand (mean < 60) and high cost volatility (sigma > 1.0) are the primary drivers of failure.

## Part 2: Performance Profiling (Subagent-Beta)
The profiler measures the impact of JIT compilation and scaling across different sample sizes.

### JAX Execution Metrics
| Sample Size | Cold Start (s) | Warm Start (s) | Speedup |
|-------------|----------------|----------------|---------|
| 10,000      | 0.3086         | 0.0016         | 189.5x  |
| 100,000     | 0.2422         | 0.0027         | 89.9x   |
| 1,000,000   | 0.2232         | 0.0199         | 11.2x   |

**Analysis**: JIT compilation (Warm Start) provides immense speedups. For 10k samples, the warm execution is nearly 190 times faster than the first execution (which includes compilation overhead).
