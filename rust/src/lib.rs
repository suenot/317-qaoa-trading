//! QAOA Trading - Quantum Approximate Optimization Algorithm for Portfolio Selection
//!
//! This crate provides a classical simulation of the QAOA algorithm applied to
//! portfolio optimization problems. It includes:
//! - QAOA circuit simulator (state vector simulation)
//! - Cost and mixer Hamiltonian evolution
//! - QUBO encoding of portfolio problems
//! - Nelder-Mead classical parameter optimizer
//! - Bybit API integration for market data

use anyhow::{anyhow, Result};
use ndarray::Array1;
use rand::Rng;
use serde::Deserialize;

// ─────────────────────────────────────────────────────────────────────────────
// QAOA Simulator
// ─────────────────────────────────────────────────────────────────────────────

/// Represents a QUBO problem: minimize x^T Q x + c^T x for binary x.
#[derive(Debug, Clone)]
pub struct QuboProblem {
    /// Quadratic coefficients (n x n, stored row-major).
    pub q_matrix: Vec<Vec<f64>>,
    /// Linear coefficients.
    pub linear: Vec<f64>,
    /// Number of binary variables (qubits).
    pub n: usize,
}

impl QuboProblem {
    /// Evaluate the QUBO cost for a given bitstring (encoded as usize).
    pub fn evaluate(&self, bitstring: usize) -> f64 {
        let mut cost = 0.0;
        for i in 0..self.n {
            let xi = ((bitstring >> i) & 1) as f64;
            cost += self.linear[i] * xi;
            for j in (i + 1)..self.n {
                let xj = ((bitstring >> j) & 1) as f64;
                cost += self.q_matrix[i][j] * xi * xj;
            }
        }
        cost
    }
}

/// Classical simulator for QAOA circuits.
pub struct QAOASimulator {
    pub n_qubits: usize,
    dim: usize,
    state_real: Array1<f64>,
    state_imag: Array1<f64>,
}

impl QAOASimulator {
    /// Create a new simulator for n qubits, initialized to |+>^n.
    pub fn new(n_qubits: usize) -> Self {
        let dim = 1 << n_qubits;
        let amp = 1.0 / (dim as f64).sqrt();
        let state_real = Array1::from_elem(dim, amp);
        let state_imag = Array1::zeros(dim);
        QAOASimulator {
            n_qubits,
            dim,
            state_real,
            state_imag,
        }
    }

    /// Reset state to uniform superposition |+>^n.
    pub fn reset(&mut self) {
        let amp = 1.0 / (self.dim as f64).sqrt();
        self.state_real.fill(amp);
        self.state_imag.fill(0.0);
    }

    /// Apply the cost unitary U_C(gamma) = exp(-i * gamma * H_C).
    /// H_C is diagonal in the computational basis with eigenvalues given by QUBO cost.
    pub fn apply_cost_unitary(&mut self, gamma: f64, problem: &QuboProblem) {
        for idx in 0..self.dim {
            let cost = problem.evaluate(idx);
            let phase = -gamma * cost;
            let (sin_p, cos_p) = phase.sin_cos();
            let re = self.state_real[idx];
            let im = self.state_imag[idx];
            self.state_real[idx] = re * cos_p + im * sin_p;
            self.state_imag[idx] = im * cos_p - re * sin_p;
        }
    }

    /// Apply the mixer unitary U_M(beta) = exp(-i * beta * sum_k X_k).
    /// This applies independent X-rotations to each qubit.
    pub fn apply_mixer_unitary(&mut self, beta: f64) {
        let (sin_b, cos_b) = beta.sin_cos();
        for qubit in 0..self.n_qubits {
            let mask = 1usize << qubit;
            for idx in 0..self.dim {
                if idx & mask == 0 {
                    let pair = idx | mask;
                    // Apply exp(-i * beta * X) to the 2x2 subspace
                    // Matrix: [[cos(beta), -i*sin(beta)], [-i*sin(beta), cos(beta)]]
                    let re0 = self.state_real[idx];
                    let im0 = self.state_imag[idx];
                    let re1 = self.state_real[pair];
                    let im1 = self.state_imag[pair];

                    self.state_real[idx] = re0 * cos_b + im1 * sin_b;
                    self.state_imag[idx] = im0 * cos_b - re1 * sin_b;
                    self.state_real[pair] = re1 * cos_b + im0 * sin_b;
                    self.state_imag[pair] = im1 * cos_b - re0 * sin_b;
                }
            }
        }
    }

    /// Run a full QAOA circuit with p layers.
    /// `params` should have length 2*p: [gamma_1, ..., gamma_p, beta_1, ..., beta_p].
    pub fn run_qaoa(&mut self, params: &[f64], problem: &QuboProblem) {
        let p = params.len() / 2;
        self.reset();
        for layer in 0..p {
            let gamma = params[layer];
            let beta = params[p + layer];
            self.apply_cost_unitary(gamma, problem);
            self.apply_mixer_unitary(beta);
        }
    }

    /// Compute the expectation value <psi| H_C |psi> of the cost Hamiltonian.
    pub fn expectation_value(&self, problem: &QuboProblem) -> f64 {
        let mut exp_val = 0.0;
        for idx in 0..self.dim {
            let prob = self.state_real[idx].powi(2) + self.state_imag[idx].powi(2);
            let cost = problem.evaluate(idx);
            exp_val += prob * cost;
        }
        exp_val
    }

    /// Get the probability distribution over all computational basis states.
    pub fn get_probabilities(&self) -> Vec<f64> {
        (0..self.dim)
            .map(|idx| self.state_real[idx].powi(2) + self.state_imag[idx].powi(2))
            .collect()
    }

    /// Sample the most probable bitstrings, returning (bitstring, probability) pairs
    /// sorted by descending probability.
    pub fn top_bitstrings(&self, top_k: usize) -> Vec<(usize, f64)> {
        let probs = self.get_probabilities();
        let mut indexed: Vec<(usize, f64)> = probs.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.truncate(top_k);
        indexed
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Nelder-Mead Optimizer
// ─────────────────────────────────────────────────────────────────────────────

/// Minimizes a function f: R^n -> R using the Nelder-Mead simplex algorithm.
pub fn nelder_mead<F>(
    f: F,
    initial: &[f64],
    max_iter: usize,
    tol: f64,
) -> (Vec<f64>, f64)
where
    F: Fn(&[f64]) -> f64,
{
    let n = initial.len();
    let alpha = 1.0; // reflection
    let gamma_expand = 2.0; // expansion
    let rho = 0.5; // contraction
    let sigma = 0.5; // shrink

    // Initialize simplex: n+1 vertices
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(initial.to_vec());
    for i in 0..n {
        let mut vertex = initial.to_vec();
        vertex[i] += if vertex[i].abs() < 1e-10 { 0.1 } else { vertex[i] * 0.1 };
        simplex.push(vertex);
    }

    let mut values: Vec<f64> = simplex.iter().map(|v| f(v)).collect();

    for _iter in 0..max_iter {
        // Sort by function value
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap());

        let sorted_simplex: Vec<Vec<f64>> = order.iter().map(|&i| simplex[i].clone()).collect();
        let sorted_values: Vec<f64> = order.iter().map(|&i| values[i]).collect();
        simplex = sorted_simplex;
        values = sorted_values;

        // Check convergence
        let range = values[n] - values[0];
        if range < tol {
            break;
        }

        // Centroid of all points except the worst
        let mut centroid = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                centroid[j] += simplex[i][j];
            }
        }
        for j in 0..n {
            centroid[j] /= n as f64;
        }

        // Reflection
        let reflected: Vec<f64> = (0..n)
            .map(|j| centroid[j] + alpha * (centroid[j] - simplex[n][j]))
            .collect();
        let f_reflected = f(&reflected);

        if f_reflected < values[0] {
            // Expansion
            let expanded: Vec<f64> = (0..n)
                .map(|j| centroid[j] + gamma_expand * (reflected[j] - centroid[j]))
                .collect();
            let f_expanded = f(&expanded);
            if f_expanded < f_reflected {
                simplex[n] = expanded;
                values[n] = f_expanded;
            } else {
                simplex[n] = reflected;
                values[n] = f_reflected;
            }
        } else if f_reflected < values[n - 1] {
            simplex[n] = reflected;
            values[n] = f_reflected;
        } else {
            // Contraction
            let contracted: Vec<f64> = (0..n)
                .map(|j| centroid[j] + rho * (simplex[n][j] - centroid[j]))
                .collect();
            let f_contracted = f(&contracted);
            if f_contracted < values[n] {
                simplex[n] = contracted;
                values[n] = f_contracted;
            } else {
                // Shrink
                for i in 1..=n {
                    for j in 0..n {
                        simplex[i][j] = simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]);
                    }
                    values[i] = f(&simplex[i]);
                }
            }
        }
    }

    // Return best
    let mut best_idx = 0;
    for i in 1..=n {
        if values[i] < values[best_idx] {
            best_idx = i;
        }
    }
    (simplex[best_idx].clone(), values[best_idx])
}

// ─────────────────────────────────────────────────────────────────────────────
// Portfolio Problem
// ─────────────────────────────────────────────────────────────────────────────

/// Encapsulates a portfolio selection problem.
#[derive(Debug, Clone)]
pub struct PortfolioProblem {
    /// Number of candidate assets.
    pub n_assets: usize,
    /// Expected returns for each asset.
    pub expected_returns: Vec<f64>,
    /// Covariance matrix (n_assets x n_assets).
    pub covariance_matrix: Vec<Vec<f64>>,
    /// Risk aversion parameter lambda in [0, 1].
    /// Higher = more return-seeking, lower = more risk-averse.
    pub risk_aversion: f64,
    /// Number of assets to select (cardinality constraint).
    pub cardinality: usize,
    /// Penalty strength for cardinality constraint.
    pub penalty_strength: f64,
}

impl PortfolioProblem {
    /// Convert to a QUBO problem for QAOA.
    pub fn to_qubo(&self) -> QuboProblem {
        let n = self.n_assets;
        let lambda = self.risk_aversion;
        let k = self.cardinality;
        let a = self.penalty_strength;

        let mut q_matrix = vec![vec![0.0; n]; n];
        let mut linear = vec![0.0; n];

        // Risk term: (1 - lambda) * Sigma_{ij} * x_i * x_j
        for i in 0..n {
            for j in (i + 1)..n {
                q_matrix[i][j] += (1.0 - lambda) * self.covariance_matrix[i][j];
            }
        }

        // Return term: -lambda * mu_i * x_i
        for i in 0..n {
            linear[i] += -lambda * self.expected_returns[i];
        }

        // Cardinality penalty: A * (sum_i x_i - k)^2
        // Expanding: A * (sum_i x_i^2 + 2*sum_{i<j} x_i*x_j - 2k*sum_i x_i + k^2)
        // Since x_i^2 = x_i for binary: A*(sum_i x_i + 2*sum_{i<j} x_i*x_j - 2k*sum_i x_i + k^2)
        for i in 0..n {
            linear[i] += a * (1.0 - 2.0 * k as f64);
        }
        for i in 0..n {
            for j in (i + 1)..n {
                q_matrix[i][j] += 2.0 * a;
            }
        }

        QuboProblem {
            q_matrix,
            linear,
            n,
        }
    }

    /// Evaluate a portfolio selection (bitstring) directly: returns (return, risk) tuple.
    pub fn evaluate_portfolio(&self, bitstring: usize) -> (f64, f64) {
        let mut total_return = 0.0;
        let mut total_risk = 0.0;
        let mut selected = Vec::new();

        for i in 0..self.n_assets {
            if (bitstring >> i) & 1 == 1 {
                selected.push(i);
                total_return += self.expected_returns[i];
            }
        }

        for &i in &selected {
            for &j in &selected {
                total_risk += self.covariance_matrix[i][j];
            }
        }

        (total_return, total_risk)
    }

    /// Get the list of selected asset indices from a bitstring.
    pub fn selected_assets(&self, bitstring: usize) -> Vec<usize> {
        (0..self.n_assets)
            .filter(|&i| (bitstring >> i) & 1 == 1)
            .collect()
    }
}

/// Run QAOA optimization for a portfolio problem.
/// Returns (best_params, best_cost, simulator_after_optimization).
pub fn optimize_qaoa(
    problem: &PortfolioProblem,
    p: usize,
    max_iter: usize,
    n_restarts: usize,
) -> (Vec<f64>, f64, QAOASimulator) {
    let qubo = problem.to_qubo();
    let n_params = 2 * p;
    let mut rng = rand::thread_rng();

    let mut global_best_params = vec![0.0; n_params];
    let mut global_best_cost = f64::MAX;

    for _ in 0..n_restarts {
        // Random initial parameters
        let initial: Vec<f64> = (0..n_params)
            .map(|i| {
                if i < p {
                    rng.gen_range(0.0..std::f64::consts::PI) // gamma
                } else {
                    rng.gen_range(0.0..std::f64::consts::PI / 2.0) // beta
                }
            })
            .collect();

        let qubo_clone = qubo.clone();
        let n_qubits = problem.n_assets;

        let cost_fn = |params: &[f64]| -> f64 {
            let mut sim = QAOASimulator::new(n_qubits);
            sim.run_qaoa(params, &qubo_clone);
            sim.expectation_value(&qubo_clone)
        };

        let (best_params, best_cost) = nelder_mead(cost_fn, &initial, max_iter, 1e-8);

        if best_cost < global_best_cost {
            global_best_cost = best_cost;
            global_best_params = best_params;
        }
    }

    // Run the simulator with optimal parameters for analysis
    let mut sim = QAOASimulator::new(problem.n_assets);
    sim.run_qaoa(&global_best_params, &qubo);

    (global_best_params, global_best_cost, sim)
}

// ─────────────────────────────────────────────────────────────────────────────
// Bybit API Integration
// ─────────────────────────────────────────────────────────────────────────────

/// Raw kline data from Bybit API.
#[derive(Debug, Clone)]
pub struct KlineData {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

#[derive(Debug, Deserialize)]
struct BybitResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: BybitResult,
}

#[derive(Debug, Deserialize)]
struct BybitResult {
    list: Vec<Vec<String>>,
}

/// Fetch kline data from Bybit REST API (blocking).
pub fn fetch_bybit_klines_blocking(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<KlineData>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::blocking::Client::new();
    let resp: BybitResponse = client
        .get(&url)
        .header("User-Agent", "qaoa-trading-bot/0.1")
        .send()?
        .json()?;

    if resp.ret_code != 0 {
        return Err(anyhow!("Bybit API error: {}", resp.ret_msg));
    }

    let mut klines: Vec<KlineData> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() >= 6 {
                Some(KlineData {
                    timestamp: row[0].parse().ok()?,
                    open: row[1].parse().ok()?,
                    high: row[2].parse().ok()?,
                    low: row[3].parse().ok()?,
                    close: row[4].parse().ok()?,
                    volume: row[5].parse().ok()?,
                })
            } else {
                None
            }
        })
        .collect();

    // Bybit returns newest first; reverse to chronological order
    klines.reverse();
    Ok(klines)
}

/// Fetch kline data from Bybit REST API (async).
pub async fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<KlineData>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::Client::new();
    let resp: BybitResponse = client
        .get(&url)
        .header("User-Agent", "qaoa-trading-bot/0.1")
        .send()
        .await?
        .json()
        .await?;

    if resp.ret_code != 0 {
        return Err(anyhow!("Bybit API error: {}", resp.ret_msg));
    }

    let mut klines: Vec<KlineData> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() >= 6 {
                Some(KlineData {
                    timestamp: row[0].parse().ok()?,
                    open: row[1].parse().ok()?,
                    high: row[2].parse().ok()?,
                    low: row[3].parse().ok()?,
                    close: row[4].parse().ok()?,
                    volume: row[5].parse().ok()?,
                })
            } else {
                None
            }
        })
        .collect();

    klines.reverse();
    Ok(klines)
}

/// Compute log returns from kline close prices.
pub fn compute_log_returns(klines: &[KlineData]) -> Vec<f64> {
    klines
        .windows(2)
        .map(|w| (w[1].close / w[0].close).ln())
        .collect()
}

/// Compute mean of a slice.
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Compute covariance between two return series.
pub fn covariance(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }
    let mean_a = mean(&a[..n]);
    let mean_b = mean(&b[..n]);
    let cov: f64 = (0..n)
        .map(|i| (a[i] - mean_a) * (b[i] - mean_b))
        .sum::<f64>();
    cov / (n - 1) as f64
}

/// Build a PortfolioProblem from multiple assets' kline data.
/// Annualizes returns and covariance assuming `trading_days` days per year.
pub fn build_portfolio_problem(
    all_returns: &[Vec<f64>],
    asset_names: &[&str],
    risk_aversion: f64,
    cardinality: usize,
    penalty_strength: f64,
    trading_days: f64,
) -> PortfolioProblem {
    let n = all_returns.len();

    // Annualized expected returns
    let expected_returns: Vec<f64> = all_returns
        .iter()
        .map(|r| mean(r) * trading_days)
        .collect();

    // Annualized covariance matrix
    let mut covariance_matrix = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            covariance_matrix[i][j] = covariance(&all_returns[i], &all_returns[j]) * trading_days;
        }
    }

    println!("Portfolio Problem Setup:");
    println!("  Assets: {:?}", asset_names);
    println!(
        "  Annualized expected returns: {:?}",
        expected_returns
            .iter()
            .map(|r| format!("{:.4}", r))
            .collect::<Vec<_>>()
    );
    println!("  Covariance matrix:");
    for i in 0..n {
        let row: Vec<String> = covariance_matrix[i].iter().map(|v| format!("{:.6}", v)).collect();
        println!("    [{}]", row.join(", "));
    }

    PortfolioProblem {
        n_assets: n,
        expected_returns,
        covariance_matrix,
        risk_aversion,
        cardinality,
        penalty_strength,
    }
}

/// Format a bitstring as a human-readable portfolio selection.
pub fn format_portfolio(bitstring: usize, asset_names: &[&str]) -> String {
    let selected: Vec<&str> = asset_names
        .iter()
        .enumerate()
        .filter(|(i, _)| (bitstring >> i) & 1 == 1)
        .map(|(_, name)| *name)
        .collect();
    if selected.is_empty() {
        "None".to_string()
    } else {
        selected.join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qubo_evaluation() {
        let problem = QuboProblem {
            q_matrix: vec![vec![0.0, 1.0], vec![0.0, 0.0]],
            linear: vec![-1.0, -1.0],
            n: 2,
        };
        // bitstring 0b00 = 0: cost = 0
        assert!((problem.evaluate(0b00) - 0.0).abs() < 1e-10);
        // bitstring 0b01 (x0=1, x1=0): cost = -1
        assert!((problem.evaluate(0b01) - (-1.0)).abs() < 1e-10);
        // bitstring 0b10 (x0=0, x1=1): cost = -1
        assert!((problem.evaluate(0b10) - (-1.0)).abs() < 1e-10);
        // bitstring 0b11 (x0=1, x1=1): cost = -1 + -1 + 1 = -1
        assert!((problem.evaluate(0b11) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_qaoa_uniform_init() {
        let sim = QAOASimulator::new(2);
        let probs = sim.get_probabilities();
        for p in &probs {
            assert!((p - 0.25).abs() < 1e-10);
        }
    }

    #[test]
    fn test_qaoa_state_normalization() {
        let mut sim = QAOASimulator::new(3);
        let problem = QuboProblem {
            q_matrix: vec![
                vec![0.0, 0.5, 0.3],
                vec![0.0, 0.0, 0.2],
                vec![0.0, 0.0, 0.0],
            ],
            linear: vec![-0.5, -0.3, -0.2],
            n: 3,
        };
        sim.run_qaoa(&[0.5, 1.0, 0.3, 0.7], &problem);
        let total: f64 = sim.get_probabilities().iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_portfolio_to_qubo() {
        let problem = PortfolioProblem {
            n_assets: 3,
            expected_returns: vec![0.1, 0.2, 0.15],
            covariance_matrix: vec![
                vec![0.04, 0.01, 0.02],
                vec![0.01, 0.09, 0.03],
                vec![0.02, 0.03, 0.06],
            ],
            risk_aversion: 0.5,
            cardinality: 2,
            penalty_strength: 1.0,
        };
        let qubo = problem.to_qubo();
        assert_eq!(qubo.n, 3);
        assert_eq!(qubo.linear.len(), 3);
    }

    #[test]
    fn test_nelder_mead_simple() {
        // Minimize (x-2)^2 + (y-3)^2
        let f = |params: &[f64]| -> f64 {
            (params[0] - 2.0).powi(2) + (params[1] - 3.0).powi(2)
        };
        let (best, val) = nelder_mead(f, &[0.0, 0.0], 1000, 1e-12);
        assert!((best[0] - 2.0).abs() < 1e-4);
        assert!((best[1] - 3.0).abs() < 1e-4);
        assert!(val < 1e-8);
    }

    #[test]
    fn test_log_returns() {
        let klines = vec![
            KlineData { timestamp: 0, open: 0.0, high: 0.0, low: 0.0, close: 100.0, volume: 0.0 },
            KlineData { timestamp: 1, open: 0.0, high: 0.0, low: 0.0, close: 110.0, volume: 0.0 },
            KlineData { timestamp: 2, open: 0.0, high: 0.0, low: 0.0, close: 105.0, volume: 0.0 },
        ];
        let returns = compute_log_returns(&klines);
        assert_eq!(returns.len(), 2);
        assert!((returns[0] - (1.1_f64).ln()).abs() < 1e-10);
    }
}
