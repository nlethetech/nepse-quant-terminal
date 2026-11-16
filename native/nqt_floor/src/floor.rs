//! One trading day's broker floor metrics, from the day's floorsheet.
//!
//! Same definitions as `backend/quant_pro/floorsheet_scraper.py`
//! (`_hhi`, `_hhi_normalized`, `_compute_signal_metrics`, `_recompute_aggregates`),
//! which is what fills the terminal's broker panels from a locally scraped
//! floorsheet, and the same arithmetic, in the same order, as the plain-Python
//! fallback in `apps/tui/io/floor_metrics.py`, so the two give identical numbers.
//!
//! Quantities are whole shares. Brokers are summed in ascending index order;
//! every other sum is over whole numbers and exact.

#![forbid(unsafe_code)]

pub const MIN_SIGNAL_TRADES: i64 = 20;
pub const MIN_SIGNAL_VOLUME: i64 = 5000;
pub const NET_RATIO_MAX: f64 = 0.15;
pub const MIN_CIRC_SHARE: f64 = 0.05;
/// Sums stay exact in f64 (and in the Python fallback) below 2^53 shares.
pub const MAX_ABS_QTY_SUM: i64 = 1 << 53;
/// Largest buyer x seller table kept for pair totals; above it pairs are sorted.
const PAIR_TABLE_MAX: usize = 1 << 20;

/// Per-symbol metrics, one entry per symbol index (zeros for a symbol with no trades).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SymbolMetrics {
    pub n_trades: Vec<i64>,
    pub total_qty: Vec<i64>,
    pub hhi_buy: Vec<f64>,
    pub hhi_sell: Vec<f64>,
    pub hhi_buy_norm: Vec<f64>,
    pub hhi_sell_norm: Vec<f64>,
    pub n_brokers_buy: Vec<i64>,
    pub n_brokers_sell: Vec<i64>,
    pub circular_score: Vec<f64>,
    pub top_pair_pct: Vec<f64>,
    pub self_trade_pct: Vec<f64>,
    pub smart_money_score: Vec<f64>,
    pub pump_score: Vec<f64>,
}

impl SymbolMetrics {
    fn zeros(n: usize) -> Self {
        SymbolMetrics {
            n_trades: vec![0; n],
            total_qty: vec![0; n],
            hhi_buy: vec![0.0; n],
            hhi_sell: vec![0.0; n],
            hhi_buy_norm: vec![0.0; n],
            hhi_sell_norm: vec![0.0; n],
            n_brokers_buy: vec![0; n],
            n_brokers_sell: vec![0; n],
            circular_score: vec![0.0; n],
            top_pair_pct: vec![0.0; n],
            self_trade_pct: vec![0.0; n],
            smart_money_score: vec![0.0; n],
            pump_score: vec![0.0; n],
        }
    }
}

/// Whole-day totals per broker index, across every symbol.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BrokerTotals {
    pub buy_qty: Vec<i64>,
    pub sell_qty: Vec<i64>,
    pub buy_trades: Vec<i64>,
    pub sell_trades: Vec<i64>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FloorDay {
    pub symbols: SymbolMetrics,
    pub brokers: BrokerTotals,
}

/// Python's `min(1.0, x)` then `max(0.0, ...)`, tie-for-tie.
fn clamp01(x: f64) -> f64 {
    let a = if x < 1.0 { x } else { 1.0 };
    if a > 0.0 {
        a
    } else {
        0.0
    }
}

/// `_hhi_normalized`: (raw HHI, normalized HHI, brokers with a positive amount).
/// `values` must be in ascending broker order.
fn hhi_normalized(values: &[i64]) -> (f64, f64, i64) {
    let mut n: i64 = 0;
    let mut total: i64 = 0;
    for &v in values {
        if v > 0 {
            n += 1;
            total += v;
        }
    }
    if n == 0 {
        return (0.0, 0.0, 0);
    }
    let tf = total as f64;
    let mut hhi = 0.0_f64;
    for &v in values {
        if v > 0 {
            let share = v as f64 / tf;
            hhi += share * share;
        }
    }
    if n == 1 {
        return (hhi, 1.0, 1);
    }
    let inv = 1.0 / n as f64;
    (hhi, clamp01((hhi - inv) / (1.0 - inv)), n)
}

fn check_index(name: &str, values: &[u32], limit: usize) -> Result<(), String> {
    match values.iter().find(|&&v| v as usize >= limit) {
        Some(v) => Err(format!("{name} index {v} is out of range (limit {limit})")),
        None => Ok(()),
    }
}

/// Largest summed quantity over the (buyer, seller) pairs in one symbol.
struct PairMax {
    n_brokers: usize,
    table: Vec<i64>,
    seen: Vec<bool>,
    touched: Vec<usize>,
    keyed: Vec<(u64, i64)>,
}

impl PairMax {
    fn new(n_brokers: usize) -> Self {
        let cells = n_brokers.checked_mul(n_brokers).filter(|&c| c <= PAIR_TABLE_MAX);
        let size = cells.unwrap_or(0);
        PairMax { n_brokers, table: vec![0; size], seen: vec![false; size], touched: Vec::new(), keyed: Vec::new() }
    }

    fn max(&mut self, trades: impl Iterator<Item = (usize, usize, i64)>) -> i64 {
        let mut best = i64::MIN;
        if !self.table.is_empty() {
            for (b, s, q) in trades {
                let key = b * self.n_brokers + s;
                if !self.seen[key] {
                    self.seen[key] = true;
                    self.touched.push(key);
                }
                self.table[key] += q;
            }
            for &key in &self.touched {
                best = best.max(self.table[key]);
                self.table[key] = 0;
                self.seen[key] = false;
            }
            self.touched.clear();
            return best;
        }
        self.keyed.clear();
        let nb = self.n_brokers as u64;
        self.keyed.extend(trades.map(|(b, s, q)| (b as u64 * nb + s as u64, q)));
        self.keyed.sort_unstable_by_key(|&(k, _)| k);
        let mut i = 0;
        while i < self.keyed.len() {
            let key = self.keyed[i].0;
            let mut sum = 0_i64;
            while i < self.keyed.len() && self.keyed[i].0 == key {
                sum += self.keyed[i].1;
                i += 1;
            }
            best = best.max(sum);
        }
        best
    }
}

/// Metrics for one day. Trade `i` is in symbol `symbol_idx[i]`, bought by
/// broker `buyer[i]` from broker `seller[i]`, `qty[i]` shares; indices are
/// below `n_symbols` / `n_brokers`.
pub fn compute(
    symbol_idx: &[u32],
    buyer: &[u32],
    seller: &[u32],
    qty: &[i64],
    n_symbols: usize,
    n_brokers: usize,
) -> Result<FloorDay, String> {
    let n = symbol_idx.len();
    if buyer.len() != n || seller.len() != n || qty.len() != n {
        return Err(format!(
            "arrays differ in length: symbol_idx {n}, buyer {}, seller {}, qty {}",
            buyer.len(),
            seller.len(),
            qty.len()
        ));
    }
    check_index("symbol", symbol_idx, n_symbols)?;
    check_index("buyer", buyer, n_brokers)?;
    check_index("seller", seller, n_brokers)?;
    let mut abs_sum: i64 = 0;
    for &q in qty {
        abs_sum = abs_sum.saturating_add(q.saturating_abs());
        if abs_sum >= MAX_ABS_QTY_SUM {
            return Err("quantities too large to add up exactly".to_string());
        }
    }

    let mut brokers = BrokerTotals {
        buy_qty: vec![0; n_brokers],
        sell_qty: vec![0; n_brokers],
        buy_trades: vec![0; n_brokers],
        sell_trades: vec![0; n_brokers],
    };
    for i in 0..n {
        let (b, s, q) = (buyer[i] as usize, seller[i] as usize, qty[i]);
        brokers.buy_qty[b] += q;
        brokers.buy_trades[b] += 1;
        brokers.sell_qty[s] += q;
        brokers.sell_trades[s] += 1;
    }

    // Group trade positions by symbol (a stable counting sort).
    let mut start = vec![0_usize; n_symbols + 1];
    for &s in symbol_idx {
        start[s as usize + 1] += 1;
    }
    for k in 0..n_symbols {
        start[k + 1] += start[k];
    }
    let mut fill = start.clone();
    let mut order = vec![0_usize; n];
    for (i, &s) in symbol_idx.iter().enumerate() {
        order[fill[s as usize]] = i;
        fill[s as usize] += 1;
    }

    let mut m = SymbolMetrics::zeros(n_symbols);
    let mut buy = vec![0_i64; n_brokers];
    let mut sell = vec![0_i64; n_brokers];
    let mut seen = vec![false; n_brokers];
    let mut touched: Vec<usize> = Vec::new();
    let mut buy_vals: Vec<i64> = Vec::new();
    let mut sell_vals: Vec<i64> = Vec::new();
    let mut pairs = PairMax::new(n_brokers);

    for s in 0..n_symbols {
        let trades = &order[start[s]..start[s + 1]];
        if trades.is_empty() {
            continue;
        }
        let mut total: i64 = 0;
        let mut self_qty: i64 = 0;
        touched.clear();
        for &i in trades {
            let (b, sl, q) = (buyer[i] as usize, seller[i] as usize, qty[i]);
            total += q;
            if b == sl {
                self_qty += q;
            }
            buy[b] += q;
            sell[sl] += q;
            for k in [b, sl] {
                if !seen[k] {
                    seen[k] = true;
                    touched.push(k);
                }
            }
        }
        touched.sort_unstable();
        buy_vals.clear();
        sell_vals.clear();
        buy_vals.extend(touched.iter().map(|&k| buy[k]));
        sell_vals.extend(touched.iter().map(|&k| sell[k]));

        let n_trades = trades.len() as i64;
        let (hhi_b, hhi_b_norm, n_b) = hhi_normalized(&buy_vals);
        let (hhi_s, hhi_s_norm, n_s) = hhi_normalized(&sell_vals);
        m.n_trades[s] = n_trades;
        m.total_qty[s] = total;
        m.hhi_buy[s] = hhi_b;
        m.hhi_sell[s] = hhi_s;
        m.hhi_buy_norm[s] = hhi_b_norm;
        m.hhi_sell_norm[s] = hhi_s_norm;
        m.n_brokers_buy[s] = n_b;
        m.n_brokers_sell[s] = n_s;

        if total > 0 {
            let self_pct = self_qty as f64 / total as f64;
            let top_pair = pairs.max(trades.iter().map(|&i| (buyer[i] as usize, seller[i] as usize, qty[i])));
            let top_pair_pct = top_pair as f64 / total as f64;

            let total_buy: i64 = buy_vals.iter().sum();
            let total_sell: i64 = sell_vals.iter().sum();
            let mut circ_qty: i64 = 0;
            for (&bq_i, &sq_i) in buy_vals.iter().zip(sell_vals.iter()) {
                let (bq, sq) = (bq_i as f64, sq_i as f64);
                let sig_buyer = total_buy > 0 && bq >= MIN_CIRC_SHARE * total_buy as f64;
                let sig_seller = total_sell > 0 && sq >= MIN_CIRC_SHARE * total_sell as f64;
                if sig_buyer && sig_seller && (bq - sq).abs() / (bq + sq + 1e-9) < NET_RATIO_MAX {
                    circ_qty += bq_i + sq_i;
                }
            }
            let circular = circ_qty as f64 / ((total_buy + total_sell) as f64 + 1e-9);

            let mut pump = 0.0;
            let mut smart = 0.0;
            if n_trades >= MIN_SIGNAL_TRADES && total >= MIN_SIGNAL_VOLUME {
                if self_pct >= 0.05 {
                    pump = self_pct;
                } else if top_pair_pct > 0.40 && circular > 0.20 {
                    pump = top_pair_pct * 0.5;
                }
                if hhi_b_norm > 0.20 && hhi_s_norm < 0.40 && circular < 0.30 && self_pct < 0.10 {
                    let x = (hhi_b_norm - 0.20) / 0.60;
                    let buy_component = if 1.0 < x { 1.0 } else { x };
                    let x = 1.0 - hhi_s_norm / 0.40;
                    let sell_component = if 0.0 > x { 0.0 } else { x };
                    let x = 1.0 - circular / 0.30;
                    let circ_penalty = if 0.0 > x { 0.0 } else { x };
                    smart = buy_component * sell_component * circ_penalty;
                }
            }
            m.self_trade_pct[s] = self_pct;
            m.top_pair_pct[s] = top_pair_pct;
            m.circular_score[s] = circular;
            m.pump_score[s] = pump;
            m.smart_money_score[s] = smart;
        }

        for &k in &touched {
            buy[k] = 0;
            sell[k] = 0;
            seen[k] = false;
        }
    }
    Ok(FloorDay { symbols: m, brokers })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, HashMap};

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn one_trade() {
        let d = compute(&[0], &[3], &[5], &[100], 1, 6).unwrap();
        let m = &d.symbols;
        assert_eq!((m.n_trades[0], m.total_qty[0]), (1, 100));
        assert_eq!((m.hhi_buy[0], m.hhi_buy_norm[0], m.n_brokers_buy[0]), (1.0, 1.0, 1));
        assert_eq!((m.hhi_sell[0], m.hhi_sell_norm[0], m.n_brokers_sell[0]), (1.0, 1.0, 1));
        assert_eq!(m.top_pair_pct[0], 1.0);
        assert_eq!((m.self_trade_pct[0], m.circular_score[0]), (0.0, 0.0));
        // one trade is below the 20-trade / 5,000-share floor
        assert_eq!((m.smart_money_score[0], m.pump_score[0]), (0.0, 0.0));
        assert_eq!(d.brokers.buy_qty[3], 100);
        assert_eq!(d.brokers.sell_trades[5], 1);
    }

    #[test]
    fn self_trades_count_on_both_sides() {
        // 25 self-trades of 300 by broker 2 plus 25 trades 1 -> 4
        let n = 50;
        let sym = vec![0_u32; n];
        let buyer: Vec<u32> = (0..n).map(|i| if i < 25 { 2 } else { 1 }).collect();
        let seller: Vec<u32> = (0..n).map(|i| if i < 25 { 2 } else { 4 }).collect();
        let qty = vec![300_i64; n];
        let d = compute(&sym, &buyer, &seller, &qty, 1, 5).unwrap();
        let m = &d.symbols;
        assert_eq!(m.self_trade_pct[0], 0.5);
        assert_eq!(m.pump_score[0], 0.5); // self-trades >= 5% -> pump = self-trade share
        assert_eq!(m.smart_money_score[0], 0.0); // self-dealing disqualifies
        assert_eq!((d.brokers.buy_trades[2], d.brokers.sell_trades[2]), (25, 25));
        // broker 2 is half of each side and perfectly balanced -> circular
        assert!(close(m.circular_score[0], 15000.0 / (30000.0 + 1e-9)));
    }

    #[test]
    fn all_one_broker() {
        let n = 30;
        let d = compute(&vec![0; n], &vec![7; n], &vec![7; n], &vec![200; n], 1, 8).unwrap();
        let m = &d.symbols;
        assert_eq!((m.hhi_buy[0], m.hhi_buy_norm[0], m.n_brokers_buy[0]), (1.0, 1.0, 1));
        assert_eq!(m.self_trade_pct[0], 1.0);
        assert_eq!(m.pump_score[0], 1.0);
        assert!(close(m.circular_score[0], 12000.0 / (12000.0 + 1e-9)));
    }

    #[test]
    fn zero_quantity_day() {
        let d = compute(&[0, 0, 1], &[1, 2, 1], &[2, 1, 3], &[0, 0, 50], 3, 4).unwrap();
        let m = &d.symbols;
        assert_eq!((m.n_trades[0], m.total_qty[0]), (2, 0));
        assert_eq!((m.hhi_buy[0], m.n_brokers_buy[0]), (0.0, 0));
        assert_eq!((m.circular_score[0], m.top_pair_pct[0], m.self_trade_pct[0]), (0.0, 0.0, 0.0));
        assert_eq!(m.n_trades[2], 0); // a symbol with no trades stays empty
        assert_eq!(m.total_qty[1], 50);
    }

    #[test]
    fn smart_money_needs_concentrated_buying_and_spread_selling() {
        // broker 0 buys 25 x 400 from 25 different sellers
        let n = 25;
        let sym = vec![0_u32; n];
        let buyer = vec![0_u32; n];
        let seller: Vec<u32> = (1..=n as u32).collect();
        let d = compute(&sym, &buyer, &seller, &vec![400; n], 1, n + 1).unwrap();
        let m = &d.symbols;
        let sell_norm = m.hhi_sell_norm[0];
        assert_eq!(sell_norm, 0.0);
        assert_eq!(m.hhi_buy_norm[0], 1.0);
        let expected = 1.0 * (1.0 - sell_norm / 0.40) * 1.0;
        assert_eq!(m.smart_money_score[0], expected);
        assert_eq!(m.pump_score[0], 0.0);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(compute(&[0], &[0], &[0], &[], 1, 1).is_err());
        assert!(compute(&[1], &[0], &[0], &[1], 1, 1).is_err());
        assert!(compute(&[0], &[0], &[3], &[1], 1, 2).is_err());
        assert!(compute(&[0], &[0], &[0], &[MAX_ABS_QTY_SUM], 1, 1).is_err());
        let empty = compute(&[], &[], &[], &[], 0, 0).unwrap();
        assert!(empty.symbols.n_trades.is_empty());
    }

    /// The Python definitions, written out the slow, obvious way.
    fn reference(sym: &[u32], buyer: &[u32], seller: &[u32], qty: &[i64]) -> BTreeMap<u32, [f64; 9]> {
        let mut out = BTreeMap::new();
        let symbols: std::collections::BTreeSet<u32> = sym.iter().copied().collect();
        for s in symbols {
            let rows: Vec<usize> = (0..sym.len()).filter(|&i| sym[i] == s).collect();
            let total: i64 = rows.iter().map(|&i| qty[i]).sum();
            let mut by: BTreeMap<u32, (i64, i64)> = BTreeMap::new();
            for &i in &rows {
                by.entry(buyer[i]).or_default().0 += qty[i];
                by.entry(seller[i]).or_default().1 += qty[i];
            }
            let buys: Vec<i64> = by.values().map(|v| v.0).collect();
            let sells: Vec<i64> = by.values().map(|v| v.1).collect();
            let (hb, hbn, _) = hhi_normalized(&buys);
            let (hs, hsn, _) = hhi_normalized(&sells);
            let mut row = [hb, hs, hbn, hsn, 0.0, 0.0, 0.0, 0.0, 0.0];
            if total > 0 {
                let self_qty: i64 = rows.iter().filter(|&&i| buyer[i] == seller[i]).map(|&i| qty[i]).sum();
                let mut pair: HashMap<(u32, u32), i64> = HashMap::new();
                for &i in &rows {
                    *pair.entry((buyer[i], seller[i])).or_default() += qty[i];
                }
                let top = *pair.values().max().unwrap() as f64 / total as f64;
                let (tb, ts): (i64, i64) = (buys.iter().sum(), sells.iter().sum());
                let mut circ_qty = 0;
                for &(b, sl) in by.values() {
                    let (bf, sf) = (b as f64, sl as f64);
                    if tb > 0 && bf >= 0.05 * tb as f64 && ts > 0 && sf >= 0.05 * ts as f64 && (bf - sf).abs() / (bf + sf + 1e-9) < 0.15 {
                        circ_qty += b + sl;
                    }
                }
                row[4] = circ_qty as f64 / ((tb + ts) as f64 + 1e-9);
                row[5] = top;
                row[6] = self_qty as f64 / total as f64;
            }
            out.insert(s, row);
        }
        out
    }

    #[test]
    fn matches_the_slow_reference_on_random_days() {
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = move |m: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % m
        };
        for round in 0..40 {
            let n = 1 + next(3000) as usize;
            let (n_sym, n_brk) = (1 + next(30) as usize, 1 + next(if round % 5 == 0 { 3 } else { 60 }) as usize);
            let sym: Vec<u32> = (0..n).map(|_| next(n_sym as u64) as u32).collect();
            let buyer: Vec<u32> = (0..n).map(|_| next(n_brk as u64) as u32).collect();
            let seller: Vec<u32> = (0..n).map(|i| if next(10) == 0 { buyer[i] } else { next(n_brk as u64) as u32 }).collect();
            let qty: Vec<i64> = (0..n).map(|_| if next(20) == 0 { 0 } else { 1 + next(2000) as i64 }).collect();
            let d = compute(&sym, &buyer, &seller, &qty, n_sym, n_brk).unwrap();
            let m = &d.symbols;
            for (s, want) in reference(&sym, &buyer, &seller, &qty) {
                let s = s as usize;
                let got = [
                    m.hhi_buy[s], m.hhi_sell[s], m.hhi_buy_norm[s], m.hhi_sell_norm[s],
                    m.circular_score[s], m.top_pair_pct[s], m.self_trade_pct[s],
                ];
                assert_eq!(&got[..], &want[..7], "round {round} symbol {s}");
            }
            let buy_total: i64 = d.brokers.buy_qty.iter().sum();
            assert_eq!(buy_total, qty.iter().sum::<i64>());
            assert_eq!(d.brokers.buy_trades.iter().sum::<i64>(), n as i64);
        }
    }

    #[test]
    fn huge_broker_codes_use_the_sorted_pair_path() {
        let n_brk = 2000; // 4M cells: above the pair table limit
        let sym = vec![0_u32; 40];
        let buyer: Vec<u32> = (0..40).map(|i| (i % 3) as u32 * 900).collect();
        let seller: Vec<u32> = (0..40).map(|i| 1999 - (i % 2) as u32).collect();
        let qty: Vec<i64> = (0..40).map(|i| 100 + i as i64).collect();
        let d = compute(&sym, &buyer, &seller, &qty, 1, n_brk).unwrap();
        let r = reference(&sym, &buyer, &seller, &qty);
        assert_eq!(d.symbols.top_pair_pct[0], r[&0][5]);
    }
}
