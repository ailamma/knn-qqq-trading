"""Regime overlay: modifies KNN leverage based on trend state.

Sits between the KNN model output and final position sizing.
KNN handles short-term signal quality; overlay handles regime awareness.

The overlay never *increases* leverage — it only caps or reduces it
when the trend context disagrees with the KNN signal.
"""

from typing import Any


def apply_regime_overlay(
    leverage: int,
    above_sma200: bool,
    golden_cross: bool,
    days_below_sma200: int = 0,
    mode: str = "moderate",
) -> int:
    """Adjust leverage based on trend regime.

    The overlay reduces long leverage in bear markets and short leverage
    in bull markets. It never increases the absolute leverage.

    Args:
        leverage: Raw leverage from KNN position sizer.
        above_sma200: Whether price is above 200-day SMA.
        golden_cross: Whether 50-day SMA > 200-day SMA.
        days_below_sma200: Consecutive days below 200-day SMA.
        mode: Overlay aggressiveness. One of:
            - "light": only cap +300% longs in bear
            - "moderate": cap longs at +100% in bear, shorts at -100% in bull
            - "aggressive": no longs in confirmed bear (death cross + extended below)
            - "symmetric": moderate + also reduce shorts in bull

    Returns:
        Adjusted leverage level.
    """
    if mode == "light":
        return _light_overlay(leverage, above_sma200, golden_cross)
    elif mode == "moderate":
        return _moderate_overlay(leverage, above_sma200, golden_cross, days_below_sma200)
    elif mode == "aggressive":
        return _aggressive_overlay(leverage, above_sma200, golden_cross, days_below_sma200)
    elif mode == "symmetric":
        return _symmetric_overlay(leverage, above_sma200, golden_cross, days_below_sma200)
    else:
        return leverage


def _light_overlay(
    leverage: int,
    above_sma200: bool,
    golden_cross: bool,
) -> int:
    """Light touch: only cap extreme longs in bear regime."""
    if not above_sma200 and leverage > 200:
        return 200
    if not above_sma200 and not golden_cross and leverage > 100:
        return 100
    return leverage


def _moderate_overlay(
    leverage: int,
    above_sma200: bool,
    golden_cross: bool,
    days_below_sma200: int,
) -> int:
    """Moderate: cap longs in bear, cap shorts in bull."""
    # Bear regime: price below 200 SMA
    if not above_sma200:
        if not golden_cross:
            # Confirmed bear (death cross): cap longs at +100%
            if leverage > 100:
                return 100
        else:
            # Below 200 SMA but golden cross still active: cap at +200%
            if leverage > 200:
                return 200

    # Bull regime: price above 200 SMA with golden cross
    if above_sma200 and golden_cross:
        # Cap shorts at -100% in confirmed bull
        if leverage < -100:
            return -100

    return leverage


def _aggressive_overlay(
    leverage: int,
    above_sma200: bool,
    golden_cross: bool,
    days_below_sma200: int,
) -> int:
    """Aggressive: force cash/short in extended bear, force cash/long in bull."""
    # Extended bear: death cross AND 20+ days below 200 SMA
    if not above_sma200 and not golden_cross and days_below_sma200 >= 20:
        # No longs allowed in confirmed extended bear
        if leverage > 0:
            return 0

    # Moderate bear: below 200 SMA
    elif not above_sma200:
        if leverage > 100:
            return 100

    # Confirmed bull: above 200 SMA with golden cross
    if above_sma200 and golden_cross:
        if leverage < -100:
            return -100

    return leverage


def _symmetric_overlay(
    leverage: int,
    above_sma200: bool,
    golden_cross: bool,
    days_below_sma200: int,
) -> int:
    """Symmetric: moderate overlay applied equally to both directions."""
    # Bear regime
    if not above_sma200:
        if not golden_cross:
            if leverage > 100:
                return 100
        else:
            if leverage > 200:
                return 200

    # Bull regime
    if above_sma200:
        if golden_cross:
            if leverage < -100:
                return -100
        else:
            if leverage < -200:
                return -200

    return leverage
