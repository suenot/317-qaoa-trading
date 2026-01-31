# Chapter 187: QAOA Trading - Quantum Approximate Optimization for Portfolio Selection

## 1. Introduction

The Quantum Approximate Optimization Algorithm (QAOA) is one of the most promising near-term quantum algorithms for solving combinatorial optimization problems. Originally proposed by Farhi, Goldstone, and Gutmann in 2014, QAOA belongs to the family of variational quantum algorithms that use a hybrid quantum-classical loop: a parameterized quantum circuit prepares candidate solutions, while a classical optimizer tunes the parameters to minimize a cost function.

In the context of trading, many core problems are inherently combinatorial. Portfolio selection with cardinality constraints (choose exactly k assets out of n), trade execution scheduling across multiple venues, and discrete asset allocation are all NP-hard problems that classical solvers struggle with as the problem size grows. QAOA provides a structured approach to exploring the exponentially large solution space by leveraging quantum superposition and interference.

This chapter introduces the QAOA framework, maps portfolio optimization problems onto it, and provides a complete Rust implementation that simulates QAOA circuits classically. We integrate live market data from the Bybit exchange to demonstrate the algorithm on real cryptocurrency assets. While current quantum hardware is noisy and limited in qubit count, classical simulation of QAOA for moderate problem sizes (up to ~20 qubits) is practical and serves as a valuable tool for understanding and benchmarking these quantum-inspired approaches.

The key insight behind QAOA is that it transforms a discrete optimization problem into a continuous optimization over circuit parameters. This allows gradient-free (or gradient-based) classical optimizers to navigate the energy landscape, finding quantum states that encode high-quality approximate solutions to the original combinatorial problem.

## 2. Mathematical Foundation

### 2.1 Problem Encoding: QUBO and Ising Models

Most combinatorial optimization problems relevant to trading can be encoded as Quadratic Unconstrained Binary Optimization (QUBO) problems:

```
minimize: x^T Q x + c^T x
subject to: x_i in {0, 1}
```

where Q is an n x n matrix encoding pairwise interactions, c is a linear cost vector, and x is a binary decision vector. For portfolio selection, x_i = 1 means asset i is included in the portfolio.

The QUBO formulation maps directly to an Ising Hamiltonian via the substitution x_i = (1 - z_i) / 2, where z_i are Pauli-Z operators with eigenvalues +/-1:

```
H_C = sum_{i<j} J_{ij} Z_i Z_j + sum_i h_i Z_i + const
```

This is the cost Hamiltonian whose ground state encodes the optimal solution.

### 2.2 QAOA Circuit Structure

QAOA constructs a p-layer parameterized quantum circuit. Starting from a uniform superposition |+>^n (all qubits in the |+> state), the circuit alternates between two types of operations:

**Cost unitary (phase separation):**
```
U_C(gamma) = exp(-i * gamma * H_C)
```

This operator applies phases proportional to the cost function value. Solutions with lower cost accumulate different phases than solutions with higher cost, creating interference patterns that favor low-cost solutions.

**Mixer unitary (mixing):**
```
U_M(beta) = exp(-i * beta * H_M)
```

where H_M = sum_i X_i is the mixer Hamiltonian (sum of Pauli-X operators). This operator "mixes" amplitude between different computational basis states, allowing the algorithm to explore the solution space.

The full QAOA state with p layers is:

```
|gamma, beta> = U_M(beta_p) U_C(gamma_p) ... U_M(beta_1) U_C(gamma_1) |+>^n
```

The algorithm seeks parameters (gamma_1, ..., gamma_p, beta_1, ..., beta_p) that minimize:

```
F(gamma, beta) = <gamma, beta| H_C |gamma, beta>
```

### 2.3 MaxCut Formulation

The canonical example for QAOA is the MaxCut problem on a graph G = (V, E). The cost Hamiltonian is:

```
H_C = sum_{(i,j) in E} (1/2)(1 - Z_i Z_j)
```

Each term contributes +1 when qubits i and j are in different states (the edge is "cut") and 0 when they agree. Portfolio optimization problems can be mapped to weighted MaxCut or more general QUBO forms, where the graph structure encodes asset correlations and the weights encode expected returns and risk.

### 2.4 The p-Layer Ansatz and Approximation Ratio

The parameter p controls the circuit depth and the expressiveness of the ansatz. For p = 1, QAOA can achieve approximation ratios of at least 0.6924 for MaxCut on 3-regular graphs (Farhi et al., 2014). As p increases, QAOA can in principle achieve arbitrarily good approximation ratios, converging to the exact solution as p approaches infinity.

In practice, increasing p introduces a more complex optimization landscape with more local minima, requiring careful initialization strategies. A common approach is to use the optimal parameters from depth p as initial guesses for depth p+1 (parameter transfer or "warm-starting").

## 3. Trading Application

### 3.1 Portfolio Selection as Combinatorial Optimization

Consider n candidate assets with expected returns mu_i, pairwise covariance Sigma_{ij}, and a cardinality constraint to select exactly k assets. The optimization problem is:

```
minimize: -lambda * sum_i mu_i x_i + (1-lambda) * sum_{i,j} Sigma_{ij} x_i x_j
subject to: sum_i x_i = k, x_i in {0, 1}
```

where lambda is a risk-return tradeoff parameter. The cardinality constraint sum_i x_i = k is enforced via a penalty term:

```
H_penalty = A * (sum_i x_i - k)^2
```

where A is a large penalty coefficient. Expanding this penalty yields additional quadratic and linear terms that fold into the QUBO matrix.

The full QUBO objective becomes:

```
Q_{ij} = (1-lambda) * Sigma_{ij} + A (for i != j)
c_i = -lambda * mu_i + A * (1 - 2k)
constant = A * k^2
```

### 3.2 Trade Execution Scheduling

QAOA can also optimize the scheduling of trade executions across time slots. Given n orders and T time slots, the binary variable x_{i,t} = 1 indicates that order i is executed in slot t. The cost function includes market impact (correlated trades in the same slot increase impact) and timing risk (delayed execution increases exposure to price movement).

### 3.3 Asset Allocation with Cardinality Constraints

Beyond binary selection, multi-level asset allocation can be encoded using multiple binary variables per asset. For example, to represent allocation levels {0%, 25%, 50%, 75%, 100%}, each asset gets 2 binary variables (encoding 4 levels plus zero). The QUBO grows quadratically in the number of binary variables but remains tractable for QAOA simulation with moderate numbers of assets.

## 4. QAOA vs Classical Optimizers

### 4.1 When Does Quantum Advantage Matter?

For small problem sizes (n < 20 assets), classical brute-force or branch-and-bound algorithms solve portfolio selection efficiently. QAOA's value proposition emerges in several scenarios:

- **Large-scale combinatorial problems** (n > 50 assets with complex constraints) where classical exact solvers face exponential scaling
- **Real-time optimization** where QAOA on quantum hardware could provide fast approximate solutions
- **Non-convex landscapes** where QAOA's quantum tunneling effect can escape local minima that trap classical optimizers
- **Multi-objective optimization** where the quantum superposition naturally encodes diverse solution portfolios

### 4.2 Current Limitations

- **Noise on quantum hardware**: Current NISQ devices have limited coherence times and gate fidelities, restricting practical QAOA to p <= 5-10 layers on ~50-100 qubits
- **Classical simulation cost**: Simulating n qubits requires 2^n complex amplitudes, limiting classical QAOA simulation to ~25-30 qubits
- **Parameter optimization**: The classical outer loop can itself become expensive, especially for large p, due to the non-convex landscape
- **Barren plateaus**: For deep circuits, the gradient of the cost function can become exponentially small, making optimization difficult

### 4.3 Classical Alternatives

For practical portfolio optimization today, these classical approaches remain competitive:

- **Simulated annealing**: Explores the solution space through thermal fluctuations
- **Genetic algorithms**: Evolutionary search with crossover and mutation
- **Branch and bound**: Exact solver with pruning (works well for moderate n)
- **Semidefinite programming (SDP) relaxation**: Provides bounds and approximate solutions

QAOA is best viewed as a research tool today, with potential for quantum advantage as hardware matures.

## 5. Implementation Walkthrough

Our Rust implementation provides a complete QAOA simulator with the following components:

### 5.1 State Vector Representation

We represent the quantum state as a complex vector of dimension 2^n. Each basis state |x> corresponds to a binary string x = x_1 x_2 ... x_n. The state vector is stored as two parallel arrays (real and imaginary parts) using the `ndarray` crate for efficient numerical operations.

```rust
pub struct QAOASimulator {
    n_qubits: usize,
    state_real: Array1<f64>,
    state_imag: Array1<f64>,
}
```

### 5.2 Cost Hamiltonian Application

The cost Hamiltonian is diagonal in the computational basis. For each basis state |x>, the phase separation unitary applies:

```rust
// For each basis state, compute cost and apply phase
for idx in 0..dim {
    let cost = compute_cost(idx, &qubo_matrix, &linear_terms);
    let phase = -gamma * cost;
    // Rotate: (re + i*im) * exp(-i*phase) = (re*cos + im*sin, im*cos - re*sin)
    let re = state_real[idx];
    let im = state_imag[idx];
    state_real[idx] = re * phase.cos() + im * phase.sin();
    state_imag[idx] = im * phase.cos() - re * phase.sin();
}
```

### 5.3 Mixer Hamiltonian Application

The mixer Hamiltonian H_M = sum_i X_i applies X rotations to each qubit independently. For a single qubit i, the rotation exp(-i * beta * X_i) mixes pairs of states that differ only in bit i:

```rust
for qubit in 0..n_qubits {
    let mask = 1 << qubit;
    for idx in 0..dim {
        if idx & mask == 0 {
            let pair = idx | mask;
            // Apply Rx(2*beta) rotation to the pair
            let cos_b = beta.cos();
            let sin_b = beta.sin();
            // Mix real and imaginary parts between |...0...> and |...1...>
        }
    }
}
```

### 5.4 Classical Optimizer

We use a Nelder-Mead simplex method for parameter optimization. This derivative-free method is well-suited for QAOA because:
- The cost landscape is non-convex with many local minima
- Gradient computation would require multiple circuit evaluations
- The parameter space is low-dimensional (2p parameters)

The optimizer iterates through reflection, expansion, contraction, and shrinkage steps to navigate the 2p-dimensional parameter space.

### 5.5 Portfolio Problem Encoding

The `PortfolioProblem` struct encapsulates the financial data:

```rust
pub struct PortfolioProblem {
    pub n_assets: usize,
    pub expected_returns: Vec<f64>,
    pub covariance_matrix: Vec<Vec<f64>>,
    pub risk_aversion: f64,
    pub cardinality: usize,
    pub penalty_strength: f64,
}
```

The `to_qubo()` method converts this into Q and c matrices suitable for QAOA.

## 6. Bybit Data Integration

The implementation includes a Bybit API client that fetches recent kline (candlestick) data for multiple trading pairs:

```rust
pub async fn fetch_bybit_klines(symbol: &str, interval: &str, limit: usize)
    -> Result<Vec<KlineData>>
```

We fetch daily candles for BTCUSDT, ETHUSDT, SOLUSDT, and XRPUSDT, then compute:

1. **Log returns**: r_t = ln(close_t / close_{t-1})
2. **Expected returns**: mu_i = mean(r_i) (annualized)
3. **Covariance matrix**: Sigma_{ij} = cov(r_i, r_j) (annualized)

These financial statistics feed directly into the portfolio optimization QUBO formulation. The Bybit REST API provides reliable, free access to historical market data without authentication for public endpoints:

```
GET https://api.bybit.com/v5/market/kline?category=spot&symbol=BTCUSDT&interval=D&limit=30
```

## 7. Key Takeaways

1. **QAOA is a variational quantum algorithm** that maps combinatorial optimization problems onto parameterized quantum circuits. The hybrid quantum-classical loop alternates between quantum state preparation and classical parameter optimization.

2. **Portfolio selection maps naturally to QUBO** formulations that QAOA can solve. Cardinality constraints, risk-return tradeoffs, and correlation structures all encode as quadratic binary objectives.

3. **The circuit depth p controls the quality-complexity tradeoff**. Shallow circuits (p=1,2) give fast but approximate solutions; deeper circuits improve quality but face harder optimization landscapes.

4. **Classical QAOA simulation is practical for small problems** (up to ~20-25 qubits). This makes it a valuable research and prototyping tool even without quantum hardware.

5. **Quantum advantage for portfolio optimization is not yet proven** but is theoretically plausible for large-scale instances with complex constraints. Current research focuses on identifying problem structures where QAOA outperforms classical heuristics.

6. **Nelder-Mead optimization works well** for tuning QAOA parameters due to the low-dimensional, non-convex nature of the parameter landscape.

7. **Real market data integration** (via Bybit API) grounds the algorithm in practical scenarios, demonstrating that the encoding framework handles real-world financial statistics including non-trivial correlation structures.

8. **Warm-starting strategies** (transferring optimal parameters from depth p to depth p+1) significantly improve convergence and solution quality in practice.

9. **The QUBO framework is universal** for binary optimization, meaning any combinatorial trading problem (scheduling, routing, allocation) can be encoded and solved with the same QAOA machinery.

10. **Rust provides excellent performance** for classical QAOA simulation due to its zero-cost abstractions, memory safety, and efficient numerical computation, making it suitable for simulating circuits with up to 20+ qubits.
