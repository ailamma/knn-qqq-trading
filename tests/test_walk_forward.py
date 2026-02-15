"""Unit tests for WalkForwardSplitter."""

import numpy as np
import pandas as pd
import pytest

from models.walk_forward import WalkForwardSplitter


def test_basic_split_count():
    """Test that number of splits is correct."""
    X = np.random.randn(100, 5)
    splitter = WalkForwardSplitter(training_window=50, test_window=1)
    assert splitter.get_n_splits(X) == 50


def test_train_test_sizes():
    """Test that train and test windows have correct sizes."""
    X = np.random.randn(100, 5)
    splitter = WalkForwardSplitter(training_window=50, test_window=1)

    for train_idx, test_idx in splitter.split(X):
        assert len(train_idx) == 50
        assert len(test_idx) == 1


def test_no_future_leakage():
    """Test that test indices are always strictly after train indices."""
    X = np.random.randn(200, 5)
    splitter = WalkForwardSplitter(training_window=100, test_window=1)

    for train_idx, test_idx in splitter.split(X):
        assert test_idx[0] > train_idx[-1], "Test data must come after training data"


def test_sliding_window():
    """Test that the window slides forward by 1 each iteration."""
    X = np.random.randn(55, 5)
    splitter = WalkForwardSplitter(training_window=50, test_window=1)

    splits = list(splitter.split(X))
    assert len(splits) == 5

    # First split: train [0..49], test [50]
    assert splits[0][0][0] == 0
    assert splits[0][0][-1] == 49
    assert splits[0][1][0] == 50

    # Second split: train [1..50], test [51]
    assert splits[1][0][0] == 1
    assert splits[1][0][-1] == 50
    assert splits[1][1][0] == 51


def test_multi_day_test_window():
    """Test with test_window > 1."""
    X = np.random.randn(60, 5)
    splitter = WalkForwardSplitter(training_window=50, test_window=5)

    splits = list(splitter.split(X))
    assert len(splits) == 6

    for train_idx, test_idx in splits:
        assert len(train_idx) == 50
        assert len(test_idx) == 5 or test_idx[-1] == len(X) - 1


def test_no_overlap():
    """Test that train and test indices never overlap."""
    X = np.random.randn(100, 5)
    splitter = WalkForwardSplitter(training_window=50, test_window=3)

    for train_idx, test_idx in splitter.split(X):
        overlap = set(train_idx) & set(test_idx)
        assert len(overlap) == 0, f"Train and test overlap: {overlap}"


def test_split_by_date():
    """Test date-filtered splitting."""
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    df = pd.DataFrame(np.random.randn(100, 5), index=dates)

    splitter = WalkForwardSplitter(training_window=50, test_window=1)
    splits = list(splitter.split_by_date(df, start_date="2020-04-01"))

    for train_idx, test_idx in splits:
        test_date = df.index[test_idx[0]]
        assert test_date >= pd.Timestamp("2020-04-01")


def test_empty_data():
    """Test with data smaller than training window."""
    X = np.random.randn(10, 5)
    splitter = WalkForwardSplitter(training_window=50, test_window=1)
    assert splitter.get_n_splits(X) == 0
    assert list(splitter.split(X)) == []


def test_with_dataframe():
    """Test that splitter works with pandas DataFrames."""
    df = pd.DataFrame(np.random.randn(100, 5))
    splitter = WalkForwardSplitter(training_window=50, test_window=1)

    splits = list(splitter.split(df))
    assert len(splits) == 50
