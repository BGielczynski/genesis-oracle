# Ascension Report

## Performance Comparison
- **Legacy Time (NumPy/Python):** 3.2987s
- **JAX Time (JIT/XLA):** 0.010228s
- **Speedup Factor:** 322.51x

## The Tracing Phenomenon
Der erste Lauf einer JIT-kompilierten Funktion in JAX ist signifikant langsamer, da JAX die Funktion "traced". Dabei wird ein abstrakter Berechnungsgraph erstellt, den der XLA-Compiler für die spezifische Hardware (GPU/TPU) optimiert und in Maschinencode übersetzt. Beim zweiten Lauf wird direkt der bereits kompilierte, hochoptimierte Code ausgeführt.

## Exercise 3: Automatic Differentiation
- **Optimierte Startgeschwindigkeit:** 30.0000 m/s
- **Ziel:** 150.0 m in 5.0 s

### Autodiff vs. Finite Differences
`jax.grad` nutzt **Automatic Differentiation (AD)**. Im Gegensatz zu finiten Differenzen ($ rac{f(x+h) - f(x)}{h} $), die den Gradienten nur approximieren und anfällig für numerische Instabilitäten (Rundungs- und Abbruchfehler) sind, berechnet JAX den exakten analytischen Gradienten durch Anwendung der Kettenregel auf den Berechnungsgraphen. AD ist effizienter, da es keine manuelle Wahl eines Schrittmaßes $h$ erfordert und exakte Werte liefert, selbst bei hochkomplexen Simulationen.
