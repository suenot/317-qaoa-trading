//! QAOA Trading Example
//!
//! Fetches live data from Bybit for BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT,
//! encodes portfolio selection as a QAOA problem, and runs optimization
//! with increasing circuit depth.

use qaoa_trading::*;

fn main() -> anyhow::Result<()> {
    println!("=== QAOA Portfolio Optimization ===\n");

    let symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"];
    let interval = "D"; // Daily candles
    let limit = 31; // 31 candles -> 30 returns

    // ── Fetch market data from Bybit ──────────────────────────────────
    println!("Fetching market data from Bybit...");
    let mut all_returns: Vec<Vec<f64>> = Vec::new();

    for symbol in &symbols {
        match fetch_bybit_klines_blocking(symbol, interval, limit) {
            Ok(klines) => {
                let returns = compute_log_returns(&klines);
                println!(
                    "  {} : {} candles -> {} returns (last close: {:.2})",
                    symbol,
                    klines.len(),
                    returns.len(),
                    klines.last().map(|k| k.close).unwrap_or(0.0)
                );
                all_returns.push(returns);
            }
            Err(e) => {
                println!("  {} : Failed to fetch ({}), using synthetic data", symbol, e);
                // Generate synthetic returns as fallback
                let mut rng = rand::thread_rng();
                let returns: Vec<f64> = (0..30)
                    .map(|_| {
                        use rand::Rng;
                        rng.gen_range(-0.05..0.05)
                    })
                    .collect();
                all_returns.push(returns);
            }
        }
    }

    println!();

    // ── Build portfolio problem ───────────────────────────────────────
    let asset_names: Vec<&str> = symbols.iter().map(|s| *s).collect();
    let problem = build_portfolio_problem(
        &all_returns,
        &asset_names,
        0.5,    // risk_aversion: balanced
        2,      // select 2 out of 4 assets
        10.0,   // penalty strength for cardinality constraint
        365.0,  // crypto trades 365 days/year
    );

    println!();

    // ── Brute-force baseline ──────────────────────────────────────────
    println!("--- Brute-Force Baseline ---");
    let qubo = problem.to_qubo();
    let n_states = 1 << problem.n_assets;
    let mut best_bf_cost = f64::MAX;
    let mut best_bf_bitstring = 0;

    for bs in 0..n_states {
        let cost = qubo.evaluate(bs);
        let selected = problem.selected_assets(bs);
        if cost < best_bf_cost {
            best_bf_cost = cost;
            best_bf_bitstring = bs;
        }
        if selected.len() == problem.cardinality {
            let (ret, risk) = problem.evaluate_portfolio(bs);
            println!(
                "  Portfolio [{}] : cost={:.6}, return={:.4}, risk={:.6}",
                format_portfolio(bs, &asset_names),
                cost,
                ret,
                risk
            );
        }
    }

    let (bf_ret, bf_risk) = problem.evaluate_portfolio(best_bf_bitstring);
    println!(
        "\n  Best (brute-force): [{}] cost={:.6}, return={:.4}, risk={:.6}\n",
        format_portfolio(best_bf_bitstring, &asset_names),
        best_bf_cost,
        bf_ret,
        bf_risk
    );

    // ── QAOA optimization with increasing depth ──────────────────────
    println!("--- QAOA Optimization ---");

    for p in 1..=4 {
        println!("\n  Circuit depth p = {}", p);

        let (best_params, best_cost, sim) = optimize_qaoa(&problem, p, 500, 5);

        println!("    Optimal cost (expectation): {:.6}", best_cost);
        println!(
            "    Parameters (gamma, beta): {:?}",
            best_params
                .iter()
                .map(|v| format!("{:.4}", v))
                .collect::<Vec<_>>()
        );

        // Show top portfolios by probability
        let top = sim.top_bitstrings(4);
        println!("    Top bitstrings by probability:");
        for (bs, prob) in &top {
            let selected = problem.selected_assets(*bs);
            let (ret, risk) = problem.evaluate_portfolio(*bs);
            let cost = qubo.evaluate(*bs);
            println!(
                "      [{}] (n={}) prob={:.4}, cost={:.4}, return={:.4}, risk={:.6}",
                format_portfolio(*bs, &asset_names),
                selected.len(),
                prob,
                cost,
                ret,
                risk
            );
        }

        // Check if QAOA found the optimal solution
        let (top_bs, top_prob) = top[0];
        if top_bs == best_bf_bitstring {
            println!(
                "    >> QAOA found the optimal portfolio with probability {:.4}!",
                top_prob
            );
        }
    }

    println!("\n=== Done ===");
    Ok(())
}
