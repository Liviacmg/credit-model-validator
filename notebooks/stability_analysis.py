import numpy as np
import pandas as pd


def calculate_psi(expected, actual, bins=10, epsilon=1e-6):
    """
    Calcula o Population Stability Index (PSI) entre duas distribuições

    Args:
        expected (array): Distribuição esperada (base)
        actual (array): Distribuição atual
        bins (int): Número de bins
        epsilon (float): Valor pequeno para evitar divisão por zero

    Returns:
        float: Valor do PSI
    """
    # Definir breakpoints baseados na distribuição esperada
    breakpoints = np.linspace(0, 1, bins + 1)

    # Calcular histogramas
    expected_hist, _ = np.histogram(expected, bins=breakpoints)
    actual_hist, _ = np.histogram(actual, bins=breakpoints)

    # Normalizar para proporções
    expected_prop = expected_hist / (len(expected) + epsilon)
    actual_prop = actual_hist / (len(actual) + epsilon)

    # Calcular PSI
    psi = np.sum((actual_prop - expected_prop) * np.log((actual_prop + epsilon) / (expected_prop + epsilon)))

    return psi


def calculate_feature_stability(X_train, X_test, features, bins=10):
    """
    Calcula a estabilidade de features entre conjuntos de treino e teste

    Args:
        X_train (DataFrame): Dados de treino
        X_test (DataFrame): Dados de teste
        features (list): Lista de features para análise
        bins (int): Número de bins para PSI

    Returns:
        DataFrame: Resultados de estabilidade para cada feature
    """
    stability_results = []

    for feature in features:
        psi = calculate_psi(
            X_train[feature].values,
            X_test[feature].values,
            bins=bins
        )

        stability_results.append({
            'feature': feature,
            'psi': psi,
            'mean_train': X_train[feature].mean(),
            'mean_test': X_test[feature].mean(),
            'std_train': X_train[feature].std(),
            'std_test': X_test[feature].std()
        })

    return pd.DataFrame(stability_results)