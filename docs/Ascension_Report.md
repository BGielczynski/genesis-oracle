# Ascension Report

## Performance Comparison
- **Legacy Time (NumPy/Python):** TBD
- **JAX Time (JIT/XLA):** TBD
- **Speedup Factor:** TBD

## The Tracing Phenomenon
Der erste Lauf einer JIT-kompilierten Funktion in JAX ist immer langsamer als der zweite, da beim ersten Aufruf das "Tracing" stattfindet. Hierbei analysiert JAX den Python-Code, erstellt einen abstrakten Berechnungsgraphen und lässt den XLA-Compiler diesen für die spezifische Hardware optimieren und in Maschinencode übersetzen. Beim zweiten Aufruf entfallen diese Schritte und es wird direkt der hochoptimierte Binärcode ausgeführt.
