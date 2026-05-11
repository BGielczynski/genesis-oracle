# Ascension Report

## Performance Comparison
- **Legacy Time (NumPy/Python):** 3.2987s
- **JAX Time (JIT/XLA):** 0.010228s
- **Speedup Factor:** 322.51x

## The Tracing Phenomenon
Der erste Lauf einer JIT-kompilierten Funktion in JAX ist signifikant langsamer, da JAX die Funktion "traced". Dabei wird ein abstrakter Berechnungsgraph erstellt, den der XLA-Compiler für die spezifische Hardware (GPU/TPU) optimiert und in Maschinencode übersetzt. Beim zweiten Lauf wird direkt der bereits kompilierte, hochoptimierte Code ausgeführt.
