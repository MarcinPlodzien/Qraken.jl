<p align="center">
  <img src="assets/Qraken.jl_logo.png" alt="Qraken.jl Logo" width="300">
</p>

<p align="center">
  <h1 align="center">Qraken.jl</h1>
  <p align="center"><strong>GPU-Accelerated Matrix-Free Quantum Simulator</strong></p>
</p>

---

**Qraken.jl** is a Julia package for simulating **pure states and mixed states** on **GPU** without constructing explicit operator matrices. Built on **CUDA.jl**, it uses GPU parallelism for efficient quantum simulation via bitwise operations on state amplitudes.

> **The GPU-Accelerated Companion to [SmoQ.jl](https://github.com/MarcinPlodzien/SmoQ.jl)**  
> While SmoQ.jl provides an educational, CPU-based platform for learning quantum simulation, **Qraken.jl** applies the same matrix-free approach on GPU for improved performance.

---

## Philosophy: Matrix-Free Meets GPU Parallelism

Traditional quantum simulators construct large matrices (gates, Hamiltonians, observables) and perform matrix-vector multiplications. **Qraken.jl** avoids this by operating directly on state amplitudes:

| Traditional Approach | Qraken.jl Approach |
|---------------------|-------------------|
| Construct 2ᴺ × 2ᴺ gate matrices | **No matrices** — gates act via bitwise index manipulation |
| Store O(4ᴺ) elements for density matrices | **MCWF trajectories** at O(2ᴺ) memory per trajectory |
| Sequential matrix-vector products | **GPU-parallel** amplitude updates across millions of threads |

```
Gate on qubit k:    Flip bit k in index → mix amplitude pairs in parallel
Observable ⟨Z⟩:     GPU reduction over |ψᵢ|² weighted by parity(bit_k)
Correlator ⟨XX⟩:    Process amplitude quartets via XOR patterns
```

---

## Why GPU? Why MCWF?

### Embarrassingly Parallel Quantum Simulation

Quantum state evolution is inherently parallelizable — each amplitude update is independent. Qraken.jl maps these operations to GPU threads:

- **2ᴺ amplitudes → 2ᴺ GPU threads**: Each thread handles its amplitude
- **Bitwise operations**: Partner amplitudes identified via XOR
- **Warp-level efficiency**: Contiguous memory access patterns for coalesced reads

### Monte Carlo Wave Function (MCWF) Scaling

For open quantum systems with noise, Qraken.jl uses **MCWF trajectories** instead of full density matrices:

| Approach | Memory | GPU Parallelism |
|----------|--------|-----------------|
| Density Matrix | O(4ᴺ) ≈ 16 TB for N=26 | Limited — matrix too large |
| **MCWF Trajectory** | O(2ᴺ) ≈ 1 GB for N=26 | Pure state fits in GPU memory |
| **Ensemble of Trajectories** | O(K × 2ᴺ) | Naturally parallel — K trajectories on K GPUs or batched |

MCWF enables simulation of **dissipative dynamics at pure-state memory cost**, making it well-suited for GPU acceleration.

---

## Core Capabilities (Planned)

| Feature | Description |
|---------|-------------|
| **GPU State Vectors** | CuArrays for 2ᴺ complex amplitudes with automatic device management |
| **Bitwise Gate Kernels** | Custom CUDA kernels for Rx, Ry, Rz, H, CNOT, CZ — no matrices |
| **GPU Observables** | Parallel reduction for ⟨X⟩, ⟨Y⟩, ⟨Z⟩, ⟨XX⟩, ⟨ZZ⟩, etc. |
| **MCWF on GPU** | Stochastic quantum jumps with per-trajectory random states |
| **Noise Channels** | Depolarizing, dephasing, amplitude damping — implemented as MCWF |
| **Trotter Evolution** | GPU-accelerated Hamiltonian time evolution |
| **QFI & Metrology** | Quantum Fisher Information with GPU-accelerated generators |

---

## Target Use Cases

1. **Large-N Open System Dynamics**  
   Simulate N = 26–30 qubits with noise using MCWF on a single GPU

2. **Ensemble Statistics**  
   Run thousands of MCWF trajectories in parallel for error bar estimation

3. **Variational Quantum Circuits**  
   Fast noisy circuit evaluation for VQE, QAOA, and quantum machine learning

4. **Quantum Metrology**  
   High-precision QFI calculation for metrological protocols

5. **Scrambling & Thermalization**  
   Study information spreading in many-body systems at scale

---

## Built on CUDA.jl

Qraken.jl is powered by [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl), Julia's native interface to NVIDIA GPUs:

```julia
using Qraken
using CUDA

# State lives on GPU
ψ = cu(zeros(ComplexF32, 2^N))
ψ[1] = 1.0f0  # |00...0⟩

# Gates execute as GPU kernels
apply_hadamard_gpu!(ψ, 1, N)
apply_cnot_gpu!(ψ, 1, 2, N)

# Observables via GPU reduction
⟨Z₁⟩ = expect_z_gpu(ψ, 1, N)

# MCWF trajectory with quantum jumps
apply_mcwf_step_gpu!(ψ, jump_operators, dt, rng)
```

---

> **Work in Progress** — This library is under active development.

<p align="center">
  <img src="assets/Qraken.jl_logo_desktop.png" alt="Qraken.jl" width="600">
</p>

---

## Related Projects

- **[SmoQ.jl](https://github.com/MarcinPlodzien/SmoQ.jl)** — The educational, CPU-based parent library. Learn how quantum simulators work from scratch!

---

## License

MIT License. See [LICENSE](LICENSE) for details.


 