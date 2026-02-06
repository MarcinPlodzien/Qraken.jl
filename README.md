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

### Bitwise Tricks for Gates

In the computational basis, each basis state $|i\rangle$ is labeled by an integer $i \in \{0, 1, \ldots, 2^N-1\}$. The binary representation of $i$ encodes which qubits are in state $|0\rangle$ or $|1\rangle$:

$$|i\rangle = |b_{N-1} \ldots b_1 b_0\rangle \quad \text{where } i = \sum_{k=0}^{N-1} b_k \cdot 2^k$$

**Single-qubit gate on qubit $k$:** A gate acting on qubit $k$ couples amplitude pairs whose indices differ only in bit $k$. For each index $i$:

1. Compute partner index: $j = i \oplus 2^k$ (XOR flips bit $k$)
2. Apply 2×2 gate matrix to the pair $(\psi_i, \psi_j)$:

$$\psi'_i = u_{00} \psi_i + u_{01} \psi_j$$

$$\psi'_j = u_{10} \psi_i + u_{11} \psi_j$$

**Concrete single-qubit gate examples:**

| Gate | Matrix $(u_{00}, u_{01}, u_{10}, u_{11})$ | Action on $(\psi_i, \psi_j)$ |
|------|------------------------------------------|------------------------------|
| $R_x(\theta)$ | $(\cos\frac{\theta}{2}, -i\sin\frac{\theta}{2}, -i\sin\frac{\theta}{2}, \cos\frac{\theta}{2})$ | Mix with phase |
| $R_y(\theta)$ | $(\cos\frac{\theta}{2}, -\sin\frac{\theta}{2}, \sin\frac{\theta}{2}, \cos\frac{\theta}{2})$ | Real rotation |
| $R_z(\theta)$ | $(e^{-i\theta/2}, 0, 0, e^{i\theta/2})$ | Phase only (no mixing) |
| $H$ | $\frac{1}{\sqrt{2}}(1, 1, 1, -1)$ | Superposition |
| $X$ | $(0, 1, 1, 0)$ | Swap $\psi_i \leftrightarrow \psi_j$ |

No $2^N \times 2^N$ matrix is ever constructed — just $2^{N-1}$ independent 2×2 operations.

**Two-qubit gate on qubits $k, l$:** Indices are grouped into quartets $(i_{00}, i_{01}, i_{10}, i_{11})$ differing in bits $k$ and $l$:

$$i_{ab} = i_{00} \oplus a \cdot 2^k \oplus b \cdot 2^l$$

Each quartet undergoes a 4×4 transformation.

**Concrete two-qubit gate examples:**

| Gate | Action |
|------|--------|
| **CZ** | Diagonal: $\psi_{i_{11}} \to -\psi_{i_{11}}$, others unchanged |
| **CNOT** (control $k$, target $l$) | If bit $k$ is 1, swap $\psi_{i_{10}} \leftrightarrow \psi_{i_{11}}$ |

**Explicit two-qubit rotation gates** (in basis $|00\rangle, |01\rangle, |10\rangle, |11\rangle$):

**$R_{zz}(\theta)$: Diagonal** — phases based on bit parity (no amplitude mixing):

$$\psi'_{00} = e^{-i\theta/2} \psi_{00}, \quad \psi'_{01} = e^{+i\theta/2} \psi_{01}, \quad \psi'_{10} = e^{+i\theta/2} \psi_{10}, \quad \psi'_{11} = e^{-i\theta/2} \psi_{11}$$

**$R_{xx}(\theta)$: Couples** $|00\rangle \leftrightarrow |11\rangle$ and $|01\rangle \leftrightarrow |10\rangle$:

$$\psi'_{00} = c\,\psi_{00} - is\,\psi_{11}, \quad \psi'_{11} = -is\,\psi_{00} + c\,\psi_{11}$$

$$\psi'_{01} = c\,\psi_{01} - is\,\psi_{10}, \quad \psi'_{10} = -is\,\psi_{01} + c\,\psi_{10}$$

**$R_{yy}(\theta)$: Similar** with sign difference on $(00,11)$ coupling:

$$\psi'_{00} = c\,\psi_{00} + is\,\psi_{11}, \quad \psi'_{11} = is\,\psi_{00} + c\,\psi_{11}$$

$$\psi'_{01} = c\,\psi_{01} - is\,\psi_{10}, \quad \psi'_{10} = -is\,\psi_{01} + c\,\psi_{10}$$

where $c = \cos(\theta/2)$, $s = \sin(\theta/2)$.

### Bitwise Tricks for Observables

**Single-qubit $\langle Z_k \rangle$:** The Pauli-Z operator is diagonal: $Z|b\rangle = (-1)^b|b\rangle$. The expectation value becomes:

$$\langle Z_k \rangle = \sum_{i=0}^{2^N-1} (-1)^{b_k(i)} |\psi_i|^2$$

where $b_k(i) = (i \gg k) \land 1$ extracts bit $k$ from index $i$. This is a weighted sum over probabilities — no matrix needed.

**Two-qubit $\langle Z_k Z_l \rangle$:** The parity of two bits determines the sign:

$$\langle Z_k Z_l \rangle = \sum_{i=0}^{2^N-1} (-1)^{b_k(i) \oplus b_l(i)} |\psi_i|^2$$

**Non-diagonal observables $\langle X_k \rangle$, $\langle Y_k \rangle$:** These couple amplitude pairs (like gates). For $\langle X_k \rangle$:

$$\langle X_k \rangle = \sum_{i: b_k(i)=0} 2 \cdot \text{Re}(\psi_i^* \psi_{i \oplus 2^k})$$

Each term involves a pair of amplitudes related by XOR — again, no matrix construction.

**Arbitrary Pauli string $P = \sigma_1 \otimes \sigma_2 \otimes \cdots \otimes \sigma_N$:** For a general Pauli string (e.g., $XZIY$), partition qubits into sets:
- $S_Z$: qubits with $Z$ (diagonal, contribute sign via bit parity)
- $S_X$: qubits with $X$ (flip bits, couple amplitude pairs)
- $S_Y$: qubits with $Y$ (flip bits + phase factor $i$)

Define the flip mask $m = \sum_{k \in S_X \cup S_Y} 2^k$. For Z-only strings ($S_X = S_Y = \emptyset$):

$$\langle P \rangle = \sum_i (-1)^{\bigoplus_{k \in S_Z} b_k(i)} |\psi_i|^2$$

For strings with X or Y, amplitudes are coupled in pairs/groups determined by the flip mask $m$, with phases from Y operators.

### Time Evolution via Trotterization

For Hamiltonian evolution $|\psi(t)\rangle = e^{-iHt}|\psi(0)\rangle$, constructing the full $2^N \times 2^N$ matrix $e^{-iHt}$ is impractical. Instead, we use **Trotter decomposition**.

For a Hamiltonian with local terms $H = \sum_j H_j$ (e.g., nearest-neighbor interactions):

$$e^{-iHt} \approx \left( \prod_j e^{-i H_j \cdot dt} \right)^{t/dt} + O(dt)$$

Each $e^{-i H_j \cdot dt}$ acts on only 1–2 qubits, so it can be applied using the bitwise gate tricks above.

**Trotter step algorithm (matrix-free):**

1. **Precompute local unitaries:** For each local term $H_j$, compute $U_j = e^{-i H_j \cdot dt}$ as a small 2×2 or 4×4 matrix. Store as `(qubit_indices, U_matrix)` pairs.

2. **Apply gates sequentially:** For each gate in the list:
   - **1-qubit gate on qubit $k$:** Loop over indices with bit $k = 0$, apply 2×2 to amplitude pairs
   - **2-qubit gate on qubits $k, l$:** Loop over indices with bits $k = l = 0$, apply 4×4 to quartets

**Pseudocode for one Trotter step:**
```
for gate in gates:
    if gate is 1-qubit on k:
        step = 2^k
        for i in 0..2^N-1 where bit_k(i) == 0:
            j = i + step
            (ψ[i], ψ[j]) ← U × (ψ[i], ψ[j])
    
    if gate is 2-qubit on (k, l):
        step_k, step_l = 2^k, 2^l
        for i in 0..2^N-1 where bit_k(i) == bit_l(i) == 0:
            i00, i01, i10, i11 = i, i+step_k, i+step_l, i+step_k+step_l
            (ψ[i00], ψ[i01], ψ[i10], ψ[i11]) ← U × (ψ[i00], ψ[i01], ψ[i10], ψ[i11])
```

**Example: Heisenberg model** $H = \sum_{\langle i,j \rangle} (J_x X_i X_j + J_y Y_i Y_j + J_z Z_i Z_j)$

Each bond $\langle i,j \rangle$ has local Hamiltonian $H_{ij} = J_x X_i X_j + J_y Y_i Y_j + J_z Z_i Z_j$. Note that XX, YY, ZZ **do not commute**, but this is handled by computing the **exact 4×4 matrix exponential**:

$$U_{ij} = e^{-i H_{ij} \cdot dt} = \exp\left(-i(J_x XX + J_y YY + J_z ZZ) \cdot dt\right)$$

This 4×4 exponential is computed exactly (cheap for a small matrix) and then applied bitwisely. The **Trotter error** arises from non-commutativity between **different bonds** (e.g., $[H_{12}, H_{23}] \neq 0$), not from within a single bond.

**Complexity comparison:**

| Method | Operations per step |
|--------|---------------------|
| Matrix exponentiation | $O(2^{3N})$ to compute $e^{-iHdt}$ |
| Matrix-vector multiply | $O(4^N)$ per step |
| **Bitwise Trotter** | $O(G \cdot 2^N)$ where $G = O(N)$ gates |

For $N = 26$ with $G \sim 50$ gates: Trotter is $\sim 10^7 \times$ cheaper than matrix methods.

### Bitwise Partial Trace

To compute the reduced density matrix $\rho_A = \text{Tr}_B(|\psi\rangle\langle\psi|)$ for subsystem $A$ (tracing out $B$), we use index bit masking.

Let $A$ contain $n_A$ qubits and $B$ contain $n_B = N - n_A$ qubits. For each pair of indices $(i_A, j_A)$ in the reduced space:

$$(\rho_A)_{i_A, j_A} = \sum_{i_B} \psi^*_{\text{idx}(i_A, i_B)} \cdot \psi_{\text{idx}(j_A, i_B)}$$

where $\text{idx}(i_A, i_B)$ reconstructs the full index from subsystem indices using bit interleaving based on the qubit partition.

**Key operations:**
- Extract subsystem bits via masks: `i_A = (i >> shift_A) & mask_A`
- Reconstruct full index: bit interleaving of $i_A$ and $i_B$
- Sum over traced-out indices $i_B$: $2^{n_B}$ terms per matrix element

This avoids constructing the full $\rho = |\psi\rangle\langle\psi|$ and directly computes the $2^{n_A} \times 2^{n_A}$ reduced matrix.

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
- $H$ is the system Hamiltonian (coherent dynamics)
- $L_k$ are the Lindblad (jump) operators describing coupling to the environment
- $\gamma_k$ are the corresponding decay rates
- The imaginary term $-\frac{i}{2}\sum_k \gamma_k L_k^\dagger L_k$ causes **non-unitary evolution** — the state norm decreases over time

**Physical interpretation:** The norm decay encodes the probability that a quantum jump (e.g., spontaneous emission, decoherence event) has occurred. A state with lower norm is "more likely" to have experienced a jump.

**Stochastic evolution algorithm:**

At each time step $dt$:

**Step 1: Evolve under $H_{\text{eff}}$** (no normalization)

$$|\tilde{\psi}(t+dt)\rangle = e^{-i H_{\text{eff}} \, dt} |\psi(t)\rangle$$

**Step 2: Compute no-jump probability** from the squared norm

$$p_{\text{no-jump}} = \langle\tilde{\psi}(t+dt)|\tilde{\psi}(t+dt)\rangle = 1 - dt \sum_k \gamma_k \langle\psi|L_k^\dagger L_k|\psi\rangle + O(dt^2)$$

**Step 3: Monte Carlo decision** — draw random $r \in [0,1]$

- If $r < p_{\text{no-jump}}$ (**no jump**): Normalize and continue

$$|\psi(t+dt)\rangle = \frac{|\tilde{\psi}(t+dt)\rangle}{\||\tilde{\psi}(t+dt)\rangle\|}$$

- If $r \geq p_{\text{no-jump}}$ (**jump occurs**): Apply $L_k$ with probability $\propto \gamma_k\langle\psi|L_k^\dagger L_k|\psi\rangle$

$$|\psi(t+dt)\rangle = \frac{L_k|\psi(t)\rangle}{\|L_k|\psi(t)\rangle\|}$$

**Step 4: Repeat** for each time step in the trajectory

**Example:** For spontaneous emission of a two-level atom, $L = \sigma^- = |g\rangle\langle e|$ is the lowering operator. A quantum jump corresponds to detecting an emitted photon, collapsing the atom to the ground state.

**Ensemble average:** The density matrix is recovered by averaging over many independent trajectories:

$$\rho(t) = \lim_{K\to\infty} \frac{1}{K} \sum_{j=1}^{K} |\psi_j(t)\rangle\langle\psi_j(t)|$$

**Observable estimation:** For any observable $O$, we estimate its expectation value by averaging over trajectories:

$$\langle O \rangle = \frac{1}{K} \sum_{j=1}^{K} \langle\psi_j|O|\psi_j\rangle$$

**Statistical error:** The standard error of the estimate scales as:

$$\sigma_{\langle O \rangle} = \frac{\sigma}{\sqrt{K}}$$

where $\sigma$ is the standard deviation of single-trajectory measurements. This $1/\sqrt{K}$ scaling means that doubling precision requires 4× more trajectories — a cost that GPU parallelism can efficiently handle.

### GPU Parallelism via Independent Trajectories

The key insight: **individual MCWF trajectories are independent**. This enables natural GPU parallelization:

| Approach | Memory scaling | 1 GB budget |
|----------|----------------|-------------|
| Density Matrix | O(4ᴺ) | N ≈ 13 qubits |
| MCWF Trajectory | O(2ᴺ) | **N ≈ 26 qubits** |
| K Trajectories | O(K × 2ᴺ) | Embarrassingly parallel |

Qraken.jl exploits this by running many MCWF trajectories in parallel on GPU, enabling simulation of **dissipative dynamics at pure-state memory cost per trajectory**.

This approach allows simulation of much larger systems compared to full density matrix evolution. While density matrices are limited to ~13 qubits on typical hardware, MCWF enables exact simulation of noisy quantum circuits for **N = 24–26 qubits**.

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

## References

The Monte Carlo Wave Function method implemented in Qraken.jl is based on foundational work in quantum optics:

1. **Dalibard, Castin, Mølmer (1992)** — *Wave-function approach to dissipative processes in quantum optics*  
   Physical Review Letters 68, 580. [DOI: 10.1103/PhysRevLett.68.580](https://doi.org/10.1103/PhysRevLett.68.580)

2. **Mølmer, Castin, Dalibard (1993)** — *Monte Carlo wave-function method in quantum optics*  
   Journal of the Optical Society of America B 10, 524. [DOI: 10.1364/JOSAB.10.000524](https://doi.org/10.1364/JOSAB.10.000524)

3. **Carmichael (1993)** — *An Open Systems Approach to Quantum Optics*  
   Springer-Verlag, Lecture Notes in Physics. [DOI: 10.1007/978-3-540-47620-7](https://doi.org/10.1007/978-3-540-47620-7)

---

## Related Projects

- **[SmoQ.jl](https://github.com/MarcinPlodzien/SmoQ.jl)** — The educational, CPU-based parent library. Learn how quantum simulators work from scratch!

---

## License

MIT License. See [LICENSE](LICENSE) for details.


 