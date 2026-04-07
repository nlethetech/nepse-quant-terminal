"""
First-Order MAML (FOMAML) for regime adaptation.

Pre-trains on Indian market data (yfinance .NS tickers) to learn regime
detection patterns, then fast-adapts to NEPSE with minimal gradient steps.

Key design decisions (from Synthesis Report):
- Use FOMAML, NOT full second-order MAML (avoids Hessian, MPS-compatible)
- Use float32 throughout (MPS does not support float64)
- Pre-train on 50 Indian stocks (similar sector structure to NEPSE)
- Fast-adapt to NEPSE with 5-10 inner-loop gradient steps
- Fallback: if PyTorch unavailable, provides stub interface

Architecture:
    RegimeNet: 2-layer MLP (input_dim -> 32 -> n_regimes)
    Features: returns, volatility, skewness, kurtosis at multiple scales
    Labels: bull=0, neutral=1, bear=2 (derived from forward returns)

Usage:
    from backend.quant_pro.maml_regime import FOMAMLRegime

    model = FOMAMLRegime(input_dim=10, n_regimes=3)
    model.pretrain_on_indian_data("data/indian_markets/")
    model.adapt_to_nepse(nepse_returns, nepse_regimes)
    probs = model.predict_regime(current_features)
    # probs = {"bull": 0.6, "neutral": 0.3, "bear": 0.1}
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from backend.quant_pro.database import get_db_path
from backend.quant_pro.paths import get_project_root

logger = logging.getLogger(__name__)

# Guard PyTorch import — it may not be available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    logger.debug("PyTorch available, FOMAML regime model fully functional")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available — FOMAML regime model is in scaffold mode")

PROJECT_ROOT = get_project_root(__file__)
DEFAULT_DB_PATH = str(get_db_path())
DEFAULT_INDIAN_DATA_DIR = str(PROJECT_ROOT / "data" / "indian_markets")
DEFAULT_CHECKPOINT_DIR = str(PROJECT_ROOT / "models" / "maml_regime")

# Regime labels
REGIME_LABELS = {0: "bull", 1: "neutral", 2: "bear"}
REGIME_IDX = {"bull": 0, "neutral": 1, "bear": 2}

# Feature engineering windows
FEATURE_WINDOWS = [5, 10, 20, 60]  # trading days


# ============================================================
# Feature Engineering
# ============================================================

def prepare_features(returns: np.ndarray, window: int = 10) -> np.ndarray:
    """
    Create feature matrix from a return series.

    Features (10-dimensional):
        0: mean return over `window` days
        1: volatility (std) over `window` days
        2: skewness over `window` days
        3: kurtosis over `window` days
        4: 5-day momentum (cumulative return)
        5: 20-day momentum
        6: 60-day momentum
        7: volatility ratio (5d vol / 20d vol)
        8: max drawdown over `window` days
        9: fraction of positive days over `window` days

    Args:
        returns: 1D array of daily returns (length T)
        window: Lookback window for rolling features

    Returns:
        Feature matrix of shape (T - max_window, 10), float32
    """
    T = len(returns)
    max_window = max(FEATURE_WINDOWS)
    if T < max_window + 1:
        return np.empty((0, 10), dtype=np.float32)

    features = np.full((T, 10), np.nan, dtype=np.float32)

    ret = returns.astype(np.float32)

    # Rolling statistics
    for i in range(max_window, T):
        w = ret[i - window:i]
        features[i, 0] = np.mean(w)
        features[i, 1] = np.std(w) + 1e-8
        features[i, 2] = float(_skewness(w))
        features[i, 3] = float(_kurtosis(w))

        # Multi-scale momentum
        features[i, 4] = np.sum(ret[i - 5:i]) if i >= 5 else 0.0
        features[i, 5] = np.sum(ret[i - 20:i]) if i >= 20 else 0.0
        features[i, 6] = np.sum(ret[i - 60:i]) if i >= 60 else 0.0

        # Volatility ratio
        vol5 = np.std(ret[max(0, i - 5):i]) + 1e-8
        vol20 = np.std(ret[max(0, i - 20):i]) + 1e-8
        features[i, 7] = vol5 / vol20

        # Max drawdown
        cum = np.cumsum(w)
        features[i, 8] = float(np.min(cum - np.maximum.accumulate(cum)))

        # Fraction of positive days
        features[i, 9] = np.mean(w > 0)

    # Trim NaN rows
    valid = ~np.isnan(features).any(axis=1)
    return features[valid]


def _skewness(x: np.ndarray) -> float:
    """Compute skewness of array."""
    n = len(x)
    if n < 3:
        return 0.0
    m = np.mean(x)
    s = np.std(x)
    if s < 1e-10:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    """Compute excess kurtosis of array."""
    n = len(x)
    if n < 4:
        return 0.0
    m = np.mean(x)
    s = np.std(x)
    if s < 1e-10:
        return 0.0
    return float(np.mean(((x - m) / s) ** 4) - 3.0)


def label_regimes(
    returns: np.ndarray,
    forward_window: int = 20,
    bull_threshold: float = 0.02,
    bear_threshold: float = -0.02,
) -> np.ndarray:
    """
    Create regime labels from forward returns.

    Args:
        returns: 1D array of daily returns
        forward_window: Days to look forward for labeling
        bull_threshold: Cumulative return threshold for bull regime
        bear_threshold: Cumulative return threshold for bear regime

    Returns:
        1D array of regime labels (0=bull, 1=neutral, 2=bear)
    """
    T = len(returns)
    labels = np.full(T, 1, dtype=np.int64)  # Default: neutral

    for i in range(T - forward_window):
        fwd_ret = np.sum(returns[i + 1:i + 1 + forward_window])
        if fwd_ret > bull_threshold:
            labels[i] = 0  # bull
        elif fwd_ret < bear_threshold:
            labels[i] = 2  # bear
        # else: neutral (1)

    return labels


# ============================================================
# Neural Network Model
# ============================================================

if TORCH_AVAILABLE:

    class RegimeNet(nn.Module):
        """
        Simple 2-layer MLP for regime classification.

        Architecture: input_dim -> hidden_dim -> ReLU -> hidden_dim -> n_regimes
        Uses float32 throughout for MPS compatibility.
        """

        def __init__(self, input_dim: int = 10, hidden_dim: int = 32, n_regimes: int = 3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, n_regimes),
            )

        def forward(self, x):
            return self.net(x)

        def get_probabilities(self, x):
            """Get softmax probabilities."""
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)


# ============================================================
# FOMAML Regime Model
# ============================================================

class FOMAMLRegime:
    """
    First-Order MAML (FOMAML) for regime detection.

    Uses first-order gradient approximation (no Hessian),
    compatible with MPS backend on M3 Mac.

    Training procedure:
    1. Pre-train: Sample tasks from Indian stock data, each task = one stock's
       regime detection. Meta-learn shared initialization across all tasks.
    2. Adapt: Fine-tune on NEPSE data with a few gradient steps.
    3. Predict: Output regime probabilities for current market state.
    """

    def __init__(
        self,
        input_dim: int = 10,
        n_regimes: int = 3,
        hidden_dim: int = 32,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
        device: Optional[str] = None,
    ):
        """
        Initialize FOMAML regime model.

        Args:
            input_dim: Number of input features
            n_regimes: Number of regime classes (default 3: bull/neutral/bear)
            hidden_dim: Hidden layer dimension
            inner_lr: Learning rate for inner loop (task adaptation)
            outer_lr: Learning rate for outer loop (meta-update)
            inner_steps: Number of gradient steps in inner loop
            device: "cpu", "mps", or "cuda" (auto-detected if None)
        """
        self.input_dim = input_dim
        self.n_regimes = n_regimes
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self._trained = False

        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available — FOMAML is in scaffold mode")
            self.model = None
            self.device = "cpu"
            return

        # Select device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(f"FOMAML device: {self.device}")

        self.model = RegimeNet(input_dim, hidden_dim, n_regimes).float()
        self.model = self.model.to(self.device)
        self.outer_optimizer = torch.optim.Adam(
            self.model.parameters(), lr=outer_lr
        )

    def _clone_model_params(self) -> List:
        """Clone model parameters for inner loop."""
        return [p.clone() for p in self.model.parameters()]

    def _set_model_params(self, params: List):
        """Set model parameters from a list."""
        for p_model, p_new in zip(self.model.parameters(), params):
            p_model.data.copy_(p_new.data)

    def inner_loop(
        self,
        support_x: "torch.Tensor",
        support_y: "torch.Tensor",
    ) -> List:
        """
        Fast adaptation on support set (inner loop of FOMAML).

        Takes `inner_steps` gradient steps on the support loss.
        Returns adapted parameters (first-order only: no create_graph).

        Args:
            support_x: Features tensor (N, input_dim)
            support_y: Labels tensor (N,) with integer regime labels

        Returns:
            List of adapted parameter tensors
        """
        # Clone current parameters
        params = self._clone_model_params()

        for step in range(self.inner_steps):
            # Set cloned params
            self._set_model_params(params)

            # Forward pass
            logits = self.model(support_x)
            loss = F.cross_entropy(logits, support_y)

            # Compute gradients (first-order only, no create_graph)
            grads = torch.autograd.grad(loss, self.model.parameters())

            # Update cloned params
            params = [p - self.inner_lr * g for p, g in zip(params, grads)]

        return params

    def outer_step(
        self,
        tasks: List[Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]],
    ) -> float:
        """
        Meta-update on batch of tasks (outer loop of FOMAML).

        For each task:
        1. Adapt on support set (inner loop)
        2. Evaluate adapted model on query set
        3. Accumulate query loss for meta-gradient

        Args:
            tasks: List of (support_x, support_y, query_x, query_y) tuples

        Returns:
            Mean query loss across tasks
        """
        self.outer_optimizer.zero_grad()

        # Save original params
        original_params = self._clone_model_params()

        total_loss = 0.0

        for support_x, support_y, query_x, query_y in tasks:
            # Inner loop: adapt on support
            adapted_params = self.inner_loop(support_x, support_y)

            # Set adapted params
            self._set_model_params(adapted_params)

            # Evaluate on query set
            logits = self.model(query_x)
            query_loss = F.cross_entropy(logits, query_y)
            total_loss += query_loss.item()

            # Compute gradients of query loss w.r.t. adapted params
            # FOMAML trick: these gradients approximate the meta-gradient
            query_loss.backward()

            # Restore original params (but keep gradients accumulated)
            # The gradients from backward() are stored on the model parameters
            # We need to accumulate them across tasks

        # Average gradients
        n_tasks = len(tasks)
        if n_tasks > 1:
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.data.div_(n_tasks)

        # Restore original params before outer update
        self._set_model_params(original_params)

        # Re-accumulate gradients on original params
        # (We need to re-run since we restored originals)
        self.outer_optimizer.zero_grad()
        meta_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        for support_x, support_y, query_x, query_y in tasks:
            adapted_params = self.inner_loop(support_x, support_y)
            self._set_model_params(adapted_params)
            logits = self.model(query_x)
            meta_loss = meta_loss + F.cross_entropy(logits, query_y)

        meta_loss = meta_loss / n_tasks
        meta_loss.backward()

        # Restore original params and apply outer gradient
        self._set_model_params(original_params)
        self.outer_optimizer.step()

        return total_loss / n_tasks

    def _create_task_from_returns(
        self,
        returns: np.ndarray,
        support_ratio: float = 0.7,
    ) -> Optional[Tuple]:
        """
        Create a (support, query) task from a single stock's returns.

        Splits data chronologically: first 70% = support, last 30% = query.
        """
        features = prepare_features(returns)
        if len(features) < 100:
            return None

        # Match features with labels (offset by max_window)
        max_window = max(FEATURE_WINDOWS)
        labels = label_regimes(returns)
        labels = labels[max_window:max_window + len(features)]

        if len(labels) != len(features):
            min_len = min(len(labels), len(features))
            features = features[:min_len]
            labels = labels[:min_len]

        # Split
        split_idx = int(len(features) * support_ratio)

        support_x = torch.tensor(features[:split_idx], dtype=torch.float32).to(self.device)
        support_y = torch.tensor(labels[:split_idx], dtype=torch.long).to(self.device)
        query_x = torch.tensor(features[split_idx:], dtype=torch.float32).to(self.device)
        query_y = torch.tensor(labels[split_idx:], dtype=torch.long).to(self.device)

        return support_x, support_y, query_x, query_y

    def pretrain_on_indian_data(
        self,
        data_dir: str = DEFAULT_INDIAN_DATA_DIR,
        n_epochs: int = 50,
        tasks_per_epoch: int = 8,
        min_rows: int = 500,
        verbose: bool = True,
    ) -> Dict:
        """
        Pre-train on Indian market data using FOMAML.

        Args:
            data_dir: Directory with Indian stock CSVs
            n_epochs: Number of meta-training epochs
            tasks_per_epoch: Number of tasks (stocks) sampled per epoch
            min_rows: Minimum rows to include a stock
            verbose: Print progress

        Returns:
            Dict with training metrics
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available — cannot pretrain")
            return {"status": "skipped", "reason": "no_pytorch"}

        # Load Indian data
        from backend.quant_pro.data_scrapers.indian_data_download import load_indian_returns
        stock_data = load_indian_returns(data_dir, min_rows=min_rows)

        if not stock_data:
            logger.warning(f"No Indian data found in {data_dir}")
            return {"status": "failed", "reason": "no_data"}

        logger.info(f"Pre-training FOMAML on {len(stock_data)} Indian stocks")

        # Prepare all tasks
        all_tasks = []
        for ticker, df in stock_data.items():
            if "Return" in df.columns:
                returns = df["Return"].dropna().values
                task = self._create_task_from_returns(returns)
                if task is not None:
                    all_tasks.append(task)

        if len(all_tasks) < 3:
            logger.warning(f"Only {len(all_tasks)} valid tasks — need at least 3")
            return {"status": "failed", "reason": "insufficient_tasks"}

        logger.info(f"Created {len(all_tasks)} pre-training tasks")

        # Meta-training loop
        self.model.train()
        losses = []

        for epoch in range(n_epochs):
            # Sample a batch of tasks
            indices = np.random.choice(len(all_tasks), size=min(tasks_per_epoch, len(all_tasks)), replace=False)
            batch = [all_tasks[i] for i in indices]

            loss = self.outer_step(batch)
            losses.append(loss)

            if verbose and (epoch + 1) % 10 == 0:
                recent_loss = np.mean(losses[-10:])
                logger.info(f"  Epoch {epoch+1}/{n_epochs}: loss={recent_loss:.4f}")

        self._trained = True
        final_loss = np.mean(losses[-10:])

        metrics = {
            "status": "success",
            "n_stocks": len(stock_data),
            "n_tasks": len(all_tasks),
            "n_epochs": n_epochs,
            "final_loss": float(final_loss),
            "device": self.device,
        }
        logger.info(f"Pre-training complete: final_loss={final_loss:.4f}")
        return metrics

    def adapt_to_nepse(
        self,
        nepse_returns: np.ndarray,
        adaptation_steps: Optional[int] = None,
    ) -> Dict:
        """
        Fast-adapt pre-trained model to NEPSE market data.

        Args:
            nepse_returns: 1D array of NEPSE index daily returns
            adaptation_steps: Override inner_steps for adaptation (default: self.inner_steps)

        Returns:
            Dict with adaptation metrics
        """
        if not TORCH_AVAILABLE or self.model is None:
            return {"status": "skipped", "reason": "no_pytorch"}

        if adaptation_steps is None:
            adaptation_steps = self.inner_steps

        features = prepare_features(nepse_returns)
        if len(features) < 50:
            return {"status": "failed", "reason": "insufficient_data"}

        max_window = max(FEATURE_WINDOWS)
        labels = label_regimes(nepse_returns)
        labels = labels[max_window:max_window + len(features)]
        min_len = min(len(labels), len(features))
        features = features[:min_len]
        labels = labels[:min_len]

        # Use all data as support for adaptation
        support_x = torch.tensor(features, dtype=torch.float32).to(self.device)
        support_y = torch.tensor(labels, dtype=torch.long).to(self.device)

        # Inner loop adaptation
        old_steps = self.inner_steps
        self.inner_steps = adaptation_steps
        adapted_params = self.inner_loop(support_x, support_y)
        self._set_model_params(adapted_params)
        self.inner_steps = old_steps

        # Evaluate
        self.model.eval()
        with torch.no_grad():
            logits = self.model(support_x)
            preds = torch.argmax(logits, dim=1)
            accuracy = float((preds == support_y).float().mean())

        self._trained = True
        logger.info(f"NEPSE adaptation: accuracy={accuracy:.3f} on {len(features)} samples")

        return {
            "status": "success",
            "accuracy": accuracy,
            "n_samples": len(features),
            "adaptation_steps": adaptation_steps,
        }

    def predict_regime(
        self,
        features: np.ndarray,
    ) -> Dict[str, float]:
        """
        Predict regime probabilities from feature vector.

        Args:
            features: Feature vector of shape (input_dim,) or (1, input_dim)

        Returns:
            Dict: {"bull": 0.6, "neutral": 0.3, "bear": 0.1}
        """
        if not TORCH_AVAILABLE or self.model is None:
            # Fallback: return neutral
            return {"bull": 0.33, "neutral": 0.34, "bear": 0.33}

        self.model.eval()

        if features.ndim == 1:
            features = features.reshape(1, -1)

        x = torch.tensor(features, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            probs = self.model.get_probabilities(x)
            probs = probs.cpu().numpy()[0]

        return {
            "bull": float(probs[0]),
            "neutral": float(probs[1]),
            "bear": float(probs[2]),
        }

    def predict_regime_from_returns(
        self,
        returns: np.ndarray,
    ) -> Dict[str, float]:
        """
        Predict current regime from a recent return series.

        Args:
            returns: Recent daily returns (at least max(FEATURE_WINDOWS) + 1 days)

        Returns:
            Dict: {"bull": 0.6, "neutral": 0.3, "bear": 0.1}
        """
        features = prepare_features(returns)
        if len(features) == 0:
            return {"bull": 0.33, "neutral": 0.34, "bear": 0.33}

        # Use the last feature vector (most recent)
        return self.predict_regime(features[-1])

    def save_checkpoint(
        self,
        path: Optional[str] = None,
        db_path: str = DEFAULT_DB_PATH,
        version: int = 1,
        metrics: Optional[Dict] = None,
    ):
        """Save model checkpoint to disk and register in meta_learning_cache."""
        if not TORCH_AVAILABLE or self.model is None:
            logger.warning("Cannot save — PyTorch not available")
            return

        if path is None:
            ckpt_dir = Path(DEFAULT_CHECKPOINT_DIR)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            path = str(ckpt_dir / f"fomaml_regime_v{version}.pt")

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "n_regimes": self.n_regimes,
            "inner_lr": self.inner_lr,
            "outer_lr": self.outer_lr,
            "inner_steps": self.inner_steps,
        }, path)

        # Register in DB
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(
                """INSERT OR REPLACE INTO meta_learning_cache
                   (model_name, version, train_date, pretrain_markets,
                    nepse_finetune_date, checkpoint_path, metrics_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    "fomaml_regime",
                    version,
                    datetime.now().strftime("%Y-%m-%d"),
                    "NSE_India",
                    datetime.now().strftime("%Y-%m-%d"),
                    path,
                    json.dumps(metrics) if metrics else None,
                ),
            )
            conn.commit()
            conn.close()
            logger.info(f"Checkpoint saved: {path}")
        except Exception as e:
            logger.warning(f"Failed to register checkpoint in DB: {e}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint from disk."""
        if not TORCH_AVAILABLE:
            logger.warning("Cannot load — PyTorch not available")
            return False

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.model = RegimeNet(
                input_dim=checkpoint.get("input_dim", self.input_dim),
                n_regimes=checkpoint.get("n_regimes", self.n_regimes),
            ).float().to(self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.inner_lr = checkpoint.get("inner_lr", self.inner_lr)
            self.outer_lr = checkpoint.get("outer_lr", self.outer_lr)
            self.inner_steps = checkpoint.get("inner_steps", self.inner_steps)
            self._trained = True
            logger.info(f"Checkpoint loaded: {path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {path}: {e}")
            return False

    @staticmethod
    def load_latest_from_db(db_path: str = DEFAULT_DB_PATH) -> Optional["FOMAMLRegime"]:
        """
        Load the latest trained FOMAML model from the meta_learning_cache.

        Returns FOMAMLRegime instance or None if no checkpoint found.
        """
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT checkpoint_path, metrics_json
                FROM meta_learning_cache
                WHERE model_name = 'fomaml_regime'
                ORDER BY version DESC LIMIT 1
            """)
            row = cursor.fetchone()
            conn.close()

            if row is None:
                return None

            ckpt_path = row[0]
            model = FOMAMLRegime()
            if model.load_checkpoint(ckpt_path):
                return model
            return None
        except Exception:
            return None

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def is_available(self) -> bool:
        return TORCH_AVAILABLE
