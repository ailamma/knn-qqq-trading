"""Position sizing engine: maps QQQ predictions to discrete leverage levels.

All sizing decisions based on QQQ model predictions (prob_up).
TQQQ/SQQQ prices used ONLY for share count calculation — never for training.

Leverage levels (implemented via TQQQ/SQQQ + cash):
  +300% = 100% in TQQQ, 0% cash
  +200% = 67% in TQQQ, 33% cash
  +100% = 33% in TQQQ, 67% cash
     0% = 100% cash (no position)
  -100% = 33% in SQQQ, 67% cash
  -200% = 67% in SQQQ, 33% cash
  -300% = 100% in SQQQ, 0% cash

Volatility targeting: target 2x Nasdaq-100 daily volatility.
If realized vol * leverage exceeds target, reduce leverage.
"""

from typing import Any

import numpy as np


# Leverage levels and their TQQQ/SQQQ allocations
LEVERAGE_LEVELS = {
    300:  {"ticker": "TQQQ", "allocation": 1.00, "cash": 0.00},
    200:  {"ticker": "TQQQ", "allocation": 0.67, "cash": 0.33},
    100:  {"ticker": "TQQQ", "allocation": 0.33, "cash": 0.67},
    0:    {"ticker": None,   "allocation": 0.00, "cash": 1.00},
    -100: {"ticker": "SQQQ", "allocation": 0.33, "cash": 0.67},
    -200: {"ticker": "SQQQ", "allocation": 0.67, "cash": 0.33},
    -300: {"ticker": "SQQQ", "allocation": 1.00, "cash": 0.00},
}

# Default confidence → leverage mapping thresholds
# prob_up mapped to leverage level based on distance from 0.50
DEFAULT_THRESHOLDS = [
    (0.85, 300),   # very high confidence bullish
    (0.70, 200),   # high confidence bullish
    (0.55, 100),   # moderate confidence bullish
    (0.45, 0),     # dead zone — cash
    (0.30, -100),  # moderate confidence bearish
    (0.15, -200),  # high confidence bearish
    (0.00, -300),  # very high confidence bearish
]


class PositionSizer:
    """Maps KNN QQQ predictions to discrete leverage levels.

    Uses confidence thresholds to determine leverage, with optional
    volatility targeting to cap leverage when realized vol is too high.

    Args:
        thresholds: List of (min_prob_up, leverage_level) pairs, sorted descending.
        vol_target_multiple: Target portfolio vol as multiple of QQQ realized vol.
            E.g., 2.0 means target 2x QQQ daily vol. Set to None to disable.
    """

    def __init__(
        self,
        thresholds: list[tuple[float, int]] | None = None,
        vol_target_multiple: float | None = 2.0,
    ) -> None:
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.vol_target_multiple = vol_target_multiple

    def _prob_to_leverage(self, prob_up: float) -> int:
        """Map prob_up to a discrete leverage level.

        Args:
            prob_up: Predicted probability of QQQ going up.

        Returns:
            Leverage level: one of {-300, -200, -100, 0, 100, 200, 300}.
        """
        for min_prob, leverage in self.thresholds:
            if prob_up >= min_prob:
                return leverage
        return -300  # Fallback: most bearish

    def _vol_adjust_leverage(
        self,
        leverage: int,
        realized_vol_annual: float,
    ) -> int:
        """Reduce leverage if portfolio vol would exceed target.

        Args:
            leverage: Raw leverage level from confidence mapping.
            realized_vol_annual: QQQ annualized realized volatility.

        Returns:
            Adjusted leverage level (may be lower magnitude).
        """
        if self.vol_target_multiple is None or realized_vol_annual <= 0:
            return leverage

        # Target vol = vol_target_multiple * QQQ vol
        target_vol = self.vol_target_multiple * realized_vol_annual

        # Portfolio vol ≈ |leverage/100| * QQQ vol (since TQQQ ≈ 3x QQQ)
        # Effective leverage = (allocation * 3) since TQQQ/SQQQ are 3x
        abs_lev = abs(leverage)
        portfolio_vol = (abs_lev / 100) * realized_vol_annual

        if portfolio_vol <= target_vol:
            return leverage

        # Reduce to max leverage that keeps vol under target
        max_lev = int(target_vol / realized_vol_annual * 100)
        # Snap to valid level
        sign = 1 if leverage > 0 else -1
        valid = [0, 100, 200, 300]
        adjusted = 0
        for lev in valid:
            if lev <= max_lev:
                adjusted = lev
        return sign * adjusted

    def size(
        self,
        prob_up: float,
        account_balance: float,
        tqqq_price: float,
        sqqq_price: float,
        realized_vol_annual: float | None = None,
        current_leverage: int = 0,
    ) -> dict[str, Any]:
        """Generate leverage recommendation based on QQQ model prediction.

        Args:
            prob_up: Model's predicted P(QQQ up) — from QQQ data only.
            account_balance: Current account balance in dollars.
            tqqq_price: Current TQQQ price (for share count only).
            sqqq_price: Current SQQQ price (for share count only).
            realized_vol_annual: QQQ 20-day realized vol (annualized). Optional.
            current_leverage: Current leverage level for state tracking.

        Returns:
            Recommendation with leverage level, allocations, and share counts.
        """
        # Map confidence to leverage
        raw_leverage = self._prob_to_leverage(prob_up)

        # Apply vol targeting if we have vol data
        if realized_vol_annual is not None:
            target_leverage = self._vol_adjust_leverage(raw_leverage, realized_vol_annual)
        else:
            target_leverage = raw_leverage

        # Get allocation details
        level_info = LEVERAGE_LEVELS[target_leverage]
        ticker = level_info["ticker"]
        allocation_pct = level_info["allocation"]
        cash_pct = level_info["cash"]

        # Calculate shares
        dollar_amount = account_balance * allocation_pct
        if ticker == "TQQQ" and tqqq_price > 0:
            shares = int(dollar_amount / tqqq_price)
        elif ticker == "SQQQ" and sqqq_price > 0:
            shares = int(dollar_amount / sqqq_price)
        else:
            shares = 0

        # Determine action relative to current position
        if target_leverage > current_leverage:
            action = "INCREASE_LEVERAGE"
        elif target_leverage < current_leverage:
            action = "DECREASE_LEVERAGE"
        else:
            action = "HOLD_LEVERAGE"

        return {
            "leverage_level": target_leverage,
            "raw_leverage": raw_leverage,
            "vol_adjusted": raw_leverage != target_leverage,
            "ticker": ticker,
            "shares": shares,
            "tqqq_allocation": allocation_pct if ticker == "TQQQ" else 0.0,
            "sqqq_allocation": allocation_pct if ticker == "SQQQ" else 0.0,
            "cash_allocation": cash_pct,
            "dollar_amount": round(dollar_amount, 2),
            "action": action,
            "current_leverage": current_leverage,
            "prob_up": round(prob_up, 4),
            "realized_vol": round(realized_vol_annual, 4) if realized_vol_annual else None,
        }
