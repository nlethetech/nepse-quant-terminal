//! `nqt_floor`: an optional native speed-up for the NEPSE Quant Terminal's
//! broker floor panels. The terminal works the same without it
//! (`apps/tui/io/floor_metrics.py` has the plain-Python version).
//!
//! Build and install into the terminal's Python:
//!     pip install maturin && maturin develop --release -m native/nqt_floor/Cargo.toml

#![deny(unsafe_code)]

pub mod floor;

#[cfg(feature = "python")]
#[pyo3::pymodule]
mod nqt_floor {
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyDict;

    /// Bumped whenever the call or the result changes shape; the Python side
    /// only uses a module with the ABI it was written for.
    #[pymodule_export]
    const ABI: u32 = 1;

    #[allow(non_upper_case_globals)]
    #[pymodule_export]
    const __version__: &str = env!("CARGO_PKG_VERSION");

    fn u32s(name: &str, raw: &[u8]) -> PyResult<Vec<u32>> {
        if raw.len() % 4 != 0 {
            return Err(PyValueError::new_err(format!("{name}: expected little-endian uint32 bytes")));
        }
        Ok(raw.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    fn i64s(name: &str, raw: &[u8]) -> PyResult<Vec<i64>> {
        if raw.len() % 8 != 0 {
            return Err(PyValueError::new_err(format!("{name}: expected little-endian int64 bytes")));
        }
        Ok(raw
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect())
    }

    /// compute_floor(symbol_idx, buyer, seller, qty, n_symbols, n_brokers) -> dict
    ///
    /// Arrays arrive as bytes: symbol_idx / buyer / seller little-endian uint32
    /// indices, qty little-endian int64 shares (``numpy.ndarray.tobytes()``).
    /// Returns per-symbol lists (length n_symbols) and per-broker lists
    /// (length n_brokers).
    #[pyfunction]
    fn compute_floor<'py>(
        py: Python<'py>,
        symbol_idx: &[u8],
        buyer: &[u8],
        seller: &[u8],
        qty: &[u8],
        n_symbols: usize,
        n_brokers: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let sym = u32s("symbol_idx", symbol_idx)?;
        let buy = u32s("buyer", buyer)?;
        let sell = u32s("seller", seller)?;
        let q = i64s("qty", qty)?;
        let day = py
            .detach(move || crate::floor::compute(&sym, &buy, &sell, &q, n_symbols, n_brokers))
            .map_err(PyValueError::new_err)?;
        let m = day.symbols;
        let b = day.brokers;
        let out = PyDict::new(py);
        out.set_item("n_trades", m.n_trades)?;
        out.set_item("total_qty", m.total_qty)?;
        out.set_item("hhi_buy", m.hhi_buy)?;
        out.set_item("hhi_sell", m.hhi_sell)?;
        out.set_item("hhi_buy_norm", m.hhi_buy_norm)?;
        out.set_item("hhi_sell_norm", m.hhi_sell_norm)?;
        out.set_item("n_brokers_buy", m.n_brokers_buy)?;
        out.set_item("n_brokers_sell", m.n_brokers_sell)?;
        out.set_item("circular_score", m.circular_score)?;
        out.set_item("top_pair_pct", m.top_pair_pct)?;
        out.set_item("self_trade_pct", m.self_trade_pct)?;
        out.set_item("smart_money_score", m.smart_money_score)?;
        out.set_item("pump_score", m.pump_score)?;
        out.set_item("buy_qty", b.buy_qty)?;
        out.set_item("sell_qty", b.sell_qty)?;
        out.set_item("buy_trades", b.buy_trades)?;
        out.set_item("sell_trades", b.sell_trades)?;
        Ok(out)
    }
}
