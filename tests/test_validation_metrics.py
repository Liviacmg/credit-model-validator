import pytest
import numpy as np
import pandas as pd
from src.validation_framework.metrics_calculator import (
    calculate_ks,
    calculate_gini,
    calculate_psi
)


def test_calculate_ks():
    # Dados de teste
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    ks = calculate_ks(y_true, y_proba)
    assert pytest.approx(ks, 0.01) == 1.0


def test_calculate_gini():
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    gini = calculate_gini(y_true, y_proba)
    assert pytest.approx(gini, 0.01) == 1.0


def test_calculate_psi():
    expected = np.random.normal(0.5, 0.1, 1000)
    actual = np.random.normal(0.5, 0.1, 1000)

    psi_low = calculate_psi(expected, actual)
    assert psi_low < 0.1

    # Teste com distribuição diferente
    actual_drift = np.random.normal(0.7, 0.1, 1000)
    psi_high = calculate_psi(expected, actual_drift)
    assert psi_high > 0.2


def test_backtesting():
    from src.validation_framework.backtesting import perform_backtesting

    # Criar dados sintéticos
    n = 100
    y_true = np.random.choice([0, 1], size=n, p=[0.8, 0.2])
    y_proba = np.where(y_true == 1,
                       np.random.uniform(0.5, 1.0, n),
                       np.random.uniform(0.0, 0.5, n))

    results = perform_backtesting(y_true, y_proba, window_size=12)
    assert len(results) == n - 12
    assert 'ks' in results.columns
    assert 'accuracy' in results.columns