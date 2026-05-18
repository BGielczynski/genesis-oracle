# Fabric Report: 1D Heat Equation via Physics-Informed Neural Networks

![Interactive 3D Projection of the PINN Solution](pinn_3d_fabric.png)

*To explore the continuous mathematical manifold interactively in 3D, please download or view the standalone HTML file here:*  
**[Download / View Interactive 3D Fabric Simulation (HTML)](pinn_3d_fabric.html)**

---

## The Operator Horizon (Fourier Neural Operators)

While our PINN forms a beautiful manifold, it is mathematically constrained to a single initial condition. Here is how a **Fourier Neural Operator (FNO)** solves this scalability problem:

* **Mapping Functional Spaces (Operator Learning):** While a standard PINN learns a mapping from specific coordinates $(x, t)$ to a scalar output for a *single* instance of the PDE, a Fourier Neural Operator learns a mapping between infinite-dimensional *functional spaces*. It takes an entire initial condition function as input and directly predicts the future state function, making it an operator rather than a simple coordinate-based approximator.
* **Global Convolutions in the Frequency Domain:** FNOs bypass local, grid-based convolutions by transforming the input data into the frequency domain using the Fast Fourier Transform (FFT). By applying linear transformations in the frequency space and then an inverse FFT, the network captures global spatial dependencies instantaneously and continuously.
* **Zero-Shot Generalization:** Because the FNO learns the underlying physics operator rather than memorizing a single solution trajectory, it achieves "Zero-Shot" predictions. Once trained on a diverse dataset of initial conditions, it can predict the evolution of an entirely new, unseen initial condition (like a square wave instead of a sine wave) in a single forward pass without requiring any retraining.
