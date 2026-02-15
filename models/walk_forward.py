"""Walk-forward train/test split infrastructure for time series validation."""

from typing import Generator

import numpy as np
import pandas as pd


class WalkForwardSplitter:
    """Sliding window walk-forward splitter for time series data.

    Generates (train_indices, test_indices) pairs where the training window
    slides forward one day at a time. Guarantees no future data leakage:
    test date is always strictly after all training dates.

    Args:
        training_window: Number of days in the training window.
        test_window: Number of days in each test fold (default 1).
    """

    def __init__(self, training_window: int = 500, test_window: int = 1) -> None:
        self.training_window = training_window
        self.test_window = test_window

    def split(
        self, X: pd.DataFrame | np.ndarray
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate train/test index pairs.

        Args:
            X: Feature matrix (used only for length).

        Yields:
            Tuple of (train_indices, test_indices) as numpy arrays.
        """
        n = len(X)

        for start in range(0, n - self.training_window - self.test_window + 1):
            train_end = start + self.training_window
            test_end = min(train_end + self.test_window, n)

            train_idx = np.arange(start, train_end)
            test_idx = np.arange(train_end, test_end)

            yield train_idx, test_idx

    def get_n_splits(self, X: pd.DataFrame | np.ndarray) -> int:
        """Return the number of splits.

        Args:
            X: Feature matrix (used only for length).

        Returns:
            Number of train/test splits.
        """
        n = len(X)
        return max(0, n - self.training_window - self.test_window + 1)

    def split_by_date(
        self,
        df: pd.DataFrame,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate train/test splits filtered by test date range.

        Args:
            df: DataFrame with DatetimeIndex.
            start_date: Only yield splits where test date >= start_date.
            end_date: Only yield splits where test date <= end_date.

        Yields:
            Tuple of (train_indices, test_indices) as numpy arrays.
        """
        for train_idx, test_idx in self.split(df):
            test_date = df.index[test_idx[0]]

            if start_date and test_date < pd.Timestamp(start_date):
                continue
            if end_date and test_date > pd.Timestamp(end_date):
                break

            yield train_idx, test_idx
