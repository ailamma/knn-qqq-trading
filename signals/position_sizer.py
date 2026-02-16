"""Position sizing engine: converts QQQ predictions to TQQQ/SQQQ share counts.

All sizing decisions are based on QQQ model predictions (prob_up).
TQQQ/SQQQ prices are used ONLY for calculating share counts — never for training.

Confidence tiers (based on KNN prob_up from QQQ data):
  - High confidence:   allocate 30% of account
  - Medium confidence: allocate 20% of account
  - Low confidence:    allocate 10% of account
  - No confidence:     0% — stay in cash
"""

from typing import Any


# Confidence tiers: (min_distance_from_threshold, allocation_pct)
# distance = how far prob_up is beyond the bull/bear threshold
TIERS = [
    (0.20, 0.30),  # High:   prob_up >= threshold + 0.20 → 30%
    (0.10, 0.20),  # Medium: prob_up >= threshold + 0.10 → 20%
    (0.00, 0.10),  # Low:    prob_up >= threshold         → 10%
]


class PositionSizer:
    """Converts KNN QQQ predictions into TQQQ/SQQQ position recommendations.

    Uses tiered allocation: 10%, 20%, or 30% of account based on how far
    the model's confidence exceeds the entry threshold. 0% if not confident.

    Args:
        bull_threshold: Minimum prob_up to go long TQQQ (default 0.55).
        bear_threshold: Maximum prob_up to go short via SQQQ (default 0.45).
    """

    def __init__(
        self,
        bull_threshold: float = 0.55,
        bear_threshold: float = 0.45,
    ) -> None:
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold

    def _get_tier(self, distance: float) -> tuple[float, str]:
        """Determine allocation tier from confidence distance.

        Args:
            distance: How far prob_up exceeds the threshold (always >= 0).

        Returns:
            Tuple of (allocation_pct, tier_label).
        """
        for min_dist, alloc in TIERS:
            if distance >= min_dist:
                pct = int(alloc * 100)
                return alloc, f"{pct}%"
        return 0.0, "0%"

    def size(
        self,
        prob_up: float,
        account_balance: float,
        tqqq_price: float,
        sqqq_price: float,
    ) -> dict[str, Any]:
        """Generate position recommendation based on QQQ model prediction.

        The model predicts QQQ direction using only QQQ + auxiliary data.
        TQQQ/SQQQ prices here are used solely to calculate share counts.

        Args:
            prob_up: Model's predicted probability of QQQ going up (from QQQ data only).
            account_balance: Current account balance in dollars.
            tqqq_price: Current TQQQ price per share (for share count only).
            sqqq_price: Current SQQQ price per share (for share count only).

        Returns:
            Recommendation dict with action, ticker, shares, allocation tier.
        """
        if prob_up >= self.bull_threshold:
            # Bullish on QQQ → buy TQQQ
            distance = prob_up - self.bull_threshold
            alloc_pct, tier = self._get_tier(distance)
            dollar_amount = account_balance * alloc_pct
            shares = int(dollar_amount / tqqq_price) if tqqq_price > 0 else 0
            return {
                "action": "BUY",
                "ticker": "TQQQ",
                "shares": shares,
                "allocation_tier": tier,
                "allocation_pct": alloc_pct,
                "dollar_amount": round(dollar_amount, 2),
                "prob_up": round(prob_up, 4),
                "confidence_distance": round(distance, 4),
            }

        elif prob_up <= self.bear_threshold:
            # Bearish on QQQ → buy SQQQ
            distance = self.bear_threshold - prob_up
            alloc_pct, tier = self._get_tier(distance)
            dollar_amount = account_balance * alloc_pct
            shares = int(dollar_amount / sqqq_price) if sqqq_price > 0 else 0
            return {
                "action": "BUY",
                "ticker": "SQQQ",
                "shares": shares,
                "allocation_tier": tier,
                "allocation_pct": alloc_pct,
                "dollar_amount": round(dollar_amount, 2),
                "prob_up": round(prob_up, 4),
                "confidence_distance": round(distance, 4),
            }

        else:
            # Dead zone — not confident enough either way
            return {
                "action": "CASH",
                "ticker": None,
                "shares": 0,
                "allocation_tier": "0%",
                "allocation_pct": 0.0,
                "dollar_amount": 0.0,
                "prob_up": round(prob_up, 4),
                "confidence_distance": 0.0,
            }
