"""Position sizing engine: converts QQQ predictions to TQQQ/SQQQ share counts."""

from typing import Any


class PositionSizer:
    """Converts KNN model predictions into TQQQ/SQQQ position recommendations.

    Scales position size by confidence level. Higher confidence = larger position.
    Caps max position at a configurable fraction of account value.

    Args:
        bull_threshold: Minimum prob_up to go long TQQQ.
        bear_threshold: Maximum prob_up to go short via SQQQ.
        max_position_pct: Maximum position size as fraction of account.
        scaling: Position scaling method ('linear', 'quadratic', 'step').
    """

    def __init__(
        self,
        bull_threshold: float = 0.58,
        bear_threshold: float = 0.42,
        max_position_pct: float = 0.50,
        scaling: str = "linear",
    ) -> None:
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold
        self.max_position_pct = max_position_pct
        self.scaling = scaling

    def _scale_confidence(self, prob_up: float, is_bull: bool) -> float:
        """Scale confidence to position size fraction.

        Args:
            prob_up: Predicted probability of up move.
            is_bull: True if bullish signal, False if bearish.

        Returns:
            Position size as fraction of max_position_pct (0 to 1).
        """
        if is_bull:
            # Distance from threshold to 1.0
            raw = (prob_up - self.bull_threshold) / (1.0 - self.bull_threshold)
        else:
            # Distance from threshold to 0.0
            raw = (self.bear_threshold - prob_up) / self.bear_threshold

        raw = max(0.0, min(1.0, raw))

        if self.scaling == "linear":
            return raw
        elif self.scaling == "quadratic":
            return raw ** 2
        elif self.scaling == "step":
            if raw > 0.66:
                return 1.0
            elif raw > 0.33:
                return 0.5
            else:
                return 0.25
        return raw

    def size(
        self,
        prob_up: float,
        account_balance: float,
        tqqq_price: float,
        sqqq_price: float,
    ) -> dict[str, Any]:
        """Generate position recommendation.

        Args:
            prob_up: Model's predicted probability of QQQ going up.
            account_balance: Current account balance in dollars.
            tqqq_price: Current TQQQ price per share.
            sqqq_price: Current SQQQ price per share.

        Returns:
            Recommendation dict with action, ticker, shares, confidence, dollar_amount.
        """
        if prob_up >= self.bull_threshold:
            # Bullish — buy TQQQ
            scale = self._scale_confidence(prob_up, is_bull=True)
            dollar_amount = account_balance * self.max_position_pct * scale
            shares = int(dollar_amount / tqqq_price)
            return {
                "action": "BUY",
                "ticker": "TQQQ",
                "shares": shares,
                "confidence": round(prob_up, 4),
                "scale_factor": round(scale, 4),
                "dollar_amount": round(dollar_amount, 2),
                "position_pct": round(dollar_amount / account_balance, 4),
            }
        elif prob_up <= self.bear_threshold:
            # Bearish — buy SQQQ
            scale = self._scale_confidence(prob_up, is_bull=False)
            dollar_amount = account_balance * self.max_position_pct * scale
            shares = int(dollar_amount / sqqq_price)
            return {
                "action": "BUY",
                "ticker": "SQQQ",
                "shares": shares,
                "confidence": round(1 - prob_up, 4),
                "scale_factor": round(scale, 4),
                "dollar_amount": round(dollar_amount, 2),
                "position_pct": round(dollar_amount / account_balance, 4),
            }
        else:
            # Dead zone — stay in cash
            return {
                "action": "CASH",
                "ticker": None,
                "shares": 0,
                "confidence": round(abs(prob_up - 0.5), 4),
                "scale_factor": 0.0,
                "dollar_amount": 0.0,
                "position_pct": 0.0,
            }
