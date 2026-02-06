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

### Open Quantum Systems: Two Equivalent Descriptions

Simulations of open quantum systems — where a quantum system interacts with an environment — can be performed in two equivalent ways:

**1. Density Matrix + Lindblad Master Equation**

The system state is described by a density matrix ρ evolving under the Lindblad equation:

$$\frac{d\rho}{dt} = -i[H,\rho] + \sum_k \gamma_k \left( L_k \rho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k, \rho\} \right)$$

This is exact but requires O(4ᴺ) memory to store ρ, limiting simulations to ~13 qubits on typical hardware.

**2. Monte Carlo Wave Function (MCWF)**

An equivalent description uses stochastic pure-state trajectories with quantum jumps. Instead of evolving the full density matrix, we simulate individual "quantum trajectories" that sample the possible measurement outcomes of the environment.

**Effective non-Hermitian Hamiltonian**

The key idea is to incorporate dissipation via an imaginary term in the Hamiltonian:

$$H_{\text{eff}} = H - \frac{i}{2} \sum_k \gamma_k L_k^\dagger L_k$$

where:
- H is the system Hamiltonian (coherent dynamics)
- Lₖ are the Lindblad (jump) operators describing coupling to the environment
- γₖ are the corresponding decay rates
- The imaginary term −(i/2)∑ₖ γₖ Lₖ†Lₖ causes **non-unitary evolution** — the state norm decreases over time

**Physical interpretation:** The norm decay encodes the probability that a quantum jump (e.g., spontaneous emission, decoherence event) has occurred. A state with lower norm is "more likely" to have experienced a jump.

**Stochastic evolution algorithm:**

At each time step dt:

1. **Evolve under H_eff** (no normalization):

$$|\tilde{\psi}(t+dt)\rangle = e^{-i H_{\text{eff}} \, dt} |\psi(t)\rangle$$

2. **Compute no-jump probability** from the squared norm:

$$p_{\text{no-jump}} = \langle\tilde{\psi}(t+dt)|\tilde{\psi}(t+dt)\rangle = 1 - dt \sum_k \gamma_k \langle\psi|L_k^\dagger L_k|\psi\rangle + O(dt^2)$$

3. **Monte Carlo decision** — draw random r ∈ [0,1]:
   - **No jump** (r < p_no-jump): Normalize the state and continue:
   
$$|\psi(t+dt)\rangle = \frac{|\tilde{\psi}(t+dt)\rangle}{\||\tilde{\psi}(t+dt)\rangle\|}$$
   
   - **Jump occurs** (r ≥ p_no-jump): Apply jump operator Lₖ with probability ∝ γₖ⟨ψ|Lₖ†Lₖ|ψ⟩, then normalize:
   
$$|\psi(t+dt)\rangle = \frac{L_k|\psi(t)\rangle}{\|L_k|\psi(t)\rangle\|}$$

4. **Repeat** for each time step in the trajectory

**Example:** For spontaneous emission of a two-level atom, L = σ⁻ = |g⟩⟨e| is the lowering operator. A quantum jump corresponds to detecting an emitted photon, collapsing the atom to the ground state.

**Ensemble average:** The density matrix is recovered by averaging over many independent trajectories:
$$\rho(t) = \lim_{K\to\infty} \frac{1}{K} \sum_{j=1}^{K} |\psi_j(t)\rangle\langle\psi_j(t)|$$

### GPU Parallelism via Independent Trajectories

The key insight: **individual MCWF trajectories are independent**. This enables natural GPU parallelization:

| Approach | Memory | Parallelism |
|----------|--------|-------------|
| Density Matrix | O(4ᴺ) ≈ 16 TB for N=26 | Limited — matrix too large |
| Single MCWF Trajectory | O(2ᴺ) ≈ 1 GB for N=26 | Pure state fits in GPU memory |
| **K Trajectories** | O(K × 2ᴺ) | **Embarrassingly parallel** — run K trajectories independently |

Qraken.jl exploits this by running many MCWF trajectories in parallel on GPU, enabling simulation of **dissipative dynamics at pure-state memory cost per trajectory**.

---

## Core Capabilities (Planned)

| Feature | Description |
|---------|-------------|
| **GPU State Vectors** | State vectors (2ᴺ complex amplitudes) stored and evolved on GPU |
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


 