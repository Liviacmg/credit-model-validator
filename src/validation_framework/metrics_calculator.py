import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score


def calculate_ks(y_true, y_proba):
    """
    Calcula a estatística KS para modelos de classificação
    """
    df = pd.DataFrame({'true': y_true, 'proba': y_proba})
    df = df.sort_values('proba')
    df['cum_bad'] = (1 - df['true']).cumsum() / (1 - df['true']).sum()
    df['cum_good'] = df['true'].cumsum() / df['true'].sum()
    return np.max(np.abs(df['cum_bad'] - df['cum_good']))


def calculate_psi(expected, actual, bins=10):
    """
    Calcula o Population Stability Index (PSI)
    """
    breaks = np.linspace(0, 1, bins + 1)
    expected_counts = np.histogram(expected, breaks)[0]
    actual_counts = np.histogram(actual, breaks)[0]

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    return np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))


def backtesting(y_true, y_pred, window=12):
    """
    Realiza backtesting de modelos de crédito
    """
    results = []
    for i in range(len(y_true) - window):
        actual = y_true[i:i + window]
        predicted = y_pred[i:i + window]
        ks = calculate_ks(actual, predicted)
        results.append(ks)
    return results