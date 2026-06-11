import pandas as pd
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_processing import (
    clean_dataframe,
    normalize_column
)


# ---------------------------
# clean_dataframe tests
# ---------------------------

def test_clean_dataframe_fills_missing_values():

    data = {
        "A": [1, 2, None, 4],
        "B": [5, None, 7, 8]
    }

    df = pd.DataFrame(data)

    result = clean_dataframe(df, ["A", "B"])

    assert result["A"].isnull().sum() == 0
    assert result["B"].isnull().sum() == 0


def test_clean_dataframe_ignores_other_columns():

    data = {
        "A": [1, None, 3],
        "B": [4, 5, 6]
    }

    df = pd.DataFrame(data)

    result = clean_dataframe(df, ["A"])

    assert result["B"].tolist() == [4, 5, 6]


def test_clean_dataframe_empty_dataframe():

    df = pd.DataFrame({
        "A": [],
        "B": []
    })

    result = clean_dataframe(df, ["A"])

    assert result.empty


# ---------------------------
# normalize_column tests
# ---------------------------

def test_normalize_column_scales_between_0_and_1():

    data = {
        "X": [10, 20, 30, 40]
    }

    df = pd.DataFrame(data)

    result = normalize_column(df, "X")

    assert result["X"].min() == 0.0
    assert result["X"].max() == 1.0


def test_normalize_column_same_values():

    data = {
        "Y": [5, 5, 5, 5]
    }

    df = pd.DataFrame(data)

    result = normalize_column(df, "Y")

    assert result["Y"].tolist() == [
        0.0,
        0.0,
        0.0,
        0.0
    ]


def test_normalize_column_not_exists():

    data = {
        "A": [1, 2, 3]
    }

    df = pd.DataFrame(data)

    result = normalize_column(df, "B")

    assert result["A"].tolist() == [1, 2, 3]