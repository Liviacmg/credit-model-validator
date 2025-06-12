import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def perform_backtesting(y_true, y_pred_proba, window_size=12):
    """
    Realiza backtesting de modelos de crédito com janela móvel

    Args:
        y_true (array): Valores reais de default
        y_pred_proba (array): Probabilidades previstas de default
        window_size (int): Tamanho da janela de backtesting

    Returns:
        dict: Resultados com KS, precisão e taxa de captura
    """
    results = []
    n_periods = len(y_true) - window_size

    for i in range(n_periods):
        start_idx = i
        end_idx = i + window_size

        actual_window = y_true[start_idx:end_idx]
        predicted_window = y_pred_proba[start_idx:end_idx]

        # Calcular KS para a janela
        ks_stat, _ = ks_2samp(
            predicted_window[actual_window == 1],
            predicted_window[actual_window == 0]
        )

        # Calcular precisão
        predicted_class = (predicted_window > 0.5).astype(int)
        accuracy = np.mean(predicted_class == actual_window)

        # Calcular taxa de captura (capture rate)
        default_rate = np.mean(actual_window)
        predicted_default_rate = np.mean(predicted_class)
        capture_rate = predicted_default_rate / default_rate if default_rate > 0 else 0

        results.append({
            'start_period': start_idx,
            'end_period': end_idx,
            'ks': ks_stat,
            'accuracy': accuracy,
            'capture_rate': capture_rate,
            'default_rate': default_rate
        })

    return pd.DataFrame(results)