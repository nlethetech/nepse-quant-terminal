// Global application state types

export type TradingMode = "paper";
export type TabId =
  | "market"
  | "portfolio"
  | "signals"
  | "lookup"
  | "agents"
  | "orders"
  | "watchlist"
  | "kalimati"
  | "account"
  | "strategies";

export const TAB_LIST: { id: TabId; key: string; label: string }[] = [
  { id: "market", key: "1", label: "MARKET" },
  { id: "portfolio", key: "2", label: "PORTFOLIO" },
  { id: "signals", key: "3", label: "SIGNALS" },
  { id: "lookup", key: "4", label: "LOOKUP" },
  { id: "agents", key: "5", label: "AGENTS" },
  { id: "orders", key: "6", label: "ORDERS" },
  { id: "watchlist", key: "7", label: "WATCHLIST" },
  { id: "kalimati", key: "8", label: "KALIMATI" },
  { id: "account", key: "9", label: "ACCOUNT" },
  { id: "strategies", key: "0", label: "STRATEGIES" },
];

export interface AppState {
  activeTab: TabId;
  tradingMode: TradingMode;
  connected: boolean; // API server connection
  lastRefresh: string | null;
  lookupSymbol: string;
  chartTimeframe: "D" | "W" | "M" | "Y" | "I";
  modalOpen: string | null; // which modal is open
}

export type AppAction =
  | { type: "SET_TAB"; tab: TabId }
  | { type: "SET_MODE"; mode: TradingMode }
  | { type: "SET_CONNECTED"; connected: boolean }
  | { type: "SET_LAST_REFRESH"; time: string }
  | { type: "SET_LOOKUP_SYMBOL"; symbol: string }
  | { type: "SET_CHART_TIMEFRAME"; tf: AppState["chartTimeframe"] }
  | { type: "OPEN_MODAL"; modal: string }
  | { type: "CLOSE_MODAL" };

export const initialState: AppState = {
  activeTab: "market",
  tradingMode: "paper",
  connected: false,
  lastRefresh: null,
  lookupSymbol: "NABIL",
  chartTimeframe: "D",
  modalOpen: null,
};
