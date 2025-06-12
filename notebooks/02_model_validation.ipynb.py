# # Validação de Modelos de Crédito - BCB 303
# ## Conformidade com Resolução BCB 303 e Parâmetros de Basileia II

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from scipy.stats import ks_2samp

# ## 1. Carregamento de Dados
# Dataset sintético de portfólio de crédito com PD, LGD e EAD

# Gerar dados sintéticos
np.random.seed(42)
n_samples = 10000

data = pd.DataFrame({
    'idade': np.random.randint(18, 70, n_samples),
    'renda': np.random.lognormal(mean=8, sigma=0.4, size=n_samples),
    'score_credito': np.random.normal(650, 100, n_samples),
    'divida_total': np.random.uniform(1000, 100000, n_samples),
    'historico_credito': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
    'garantias': np.random.uniform(0, 1, n_samples),
    'tipo_emprestimo': np.random.choice(['pessoal', 'imobiliario', 'automovel'], n_samples),
})

# Calcular PD (Probability of Default)
data['PD'] = 1 / (1 + np.exp(-(
        0.02 * data['idade'] +
        0.0001 * data['renda'] -
        0.005 * data['score_credito'] +
        0.000003 * data['divida_total'] -
        0.5 * data['historico_credito'] +
        np.random.normal(0, 0.2, n_samples)
))

# Calcular LGD (Loss Given Default)
data['LGD'] = np.clip(0.8 - 0.1 * data['garantias'] - 0.05 * data['score_credito'] / 800 + np.random.normal(0, 0.1, n_samples), 0.2, 0.9)

# Calcular EAD (Exposure at Default)
data['EAD'] = np.clip(data['divida_total'] * (0.8 + 0.2 * data['PD']) + np.random.normal(0, 1000, n_samples), 1000,
                      100000)

# Gerar variável target (default)
data['default'] = np.random.binomial(1, data['PD'])

# Salvar dataset
data.to_csv('../data/sample_loan_portfolio.csv', index=False)
data = pd.read_csv('../data/sample_loan_portfolio.csv')

# ## 2. Treinamento do Modelo de PD
# Modelo Random Forest para Probability of Default

# Pré-processamento
X = data[['idade', 'renda', 'score_credito', 'divida_total', 'historico_credito', 'garantias']]
y = data['default']

# Codificar variáveis categóricas
X = pd.get_dummies(X, drop_first=True)

# Split temporal (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False, random_state=42
)

# Treinar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Previsões
y_pred_proba = model.predict_proba(X_test)[:, 1]
test_data = X_test.copy()
test_data['PD_predito'] = y_pred_proba
test_data['default_real'] = y_test

# ## 3. Cálculo de Métricas de Validação
# Conforme requisitos BCB 303

def calculate_ks(y_true, y_proba):
    df = pd.DataFrame({'true': y_true, 'proba': y_proba})
    df = df.sort_values('proba')
    df['cum_bad'] = (1 - df['true']).cumsum() / (1 - df['true']).sum()
    df['cum_good'] = df['true'].cumsum() / df['true'].sum()
    ks = np.max(np.abs(df['cum_bad'] - df['cum_good']))
    return ks


def calculate_gini(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    return 2 * roc_auc - 1


def calculate_psi(expected, actual, bins=10):
    breakpoints = np.linspace(0, 1, bins + 1)
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_ratio = expected_counts / len(expected)
    actual_ratio = actual_counts / len(actual)

    psi = np.sum((actual_ratio - expected_ratio) * np.log(actual_ratio / expected_ratio))
    return psi


def backtesting(y_true, y_pred, window=12):
    results = []
    for i in range(len(y_true) - window):
        actual = y_true[i:i + window]
        predicted = y_pred[i:i + window]
        ks = calculate_ks(actual, predicted)
        results.append(ks)
    return results


# Calcular métricas
ks = calculate_ks(y_test, y_pred_proba)
gini = calculate_gini(y_test, y_pred_proba)

# PSI entre treino e teste
train_proba = model.predict_proba(X_train)[:, 1]
psi = calculate_psi(train_proba, y_pred_proba)

# Backtesting
backtesting_results = backtesting(y_test.values, y_pred_proba)

# ## 4. Visualização e Análise
# Gráficos para validação de modelos

# Plot KS
plt.figure(figsize=(10, 6))
sns.kdeplot(y_pred_proba[y_test == 0], label='Bons Pagadores', color='green')
sns.kdeplot(y_pred_proba[y_test == 1], label='Maus Pagadores', color='red')
plt.title(f'Distribuição de PD - KS: {ks:.3f}')
plt.xlabel('Probabilidade de Default (PD)')
plt.legend()
plt.show()

# Plot Backtesting
plt.figure(figsize=(10, 6))
plt.plot(backtesting_results, marker='o')
plt.axhline(y=0.3, color='r', linestyle='--', label='Limite BCB 303 (KS>0.3)')
plt.title('Backtesting de Modelo - Evolução do KS')
plt.xlabel('Período')
plt.ylabel('Estatística KS')
plt.legend()
plt.grid(True)
plt.show()

# ## 5. Relatório de Conformidade BCB 303
# Verificação de requisitos regulatórios

# Verificar conformidade
compliance = {
    'KS > 0.25': ks > 0.25,
    'Gini > 0.20': gini > 0.20,
    'PSI < 0.10': psi < 0.10,
    'Backtesting consistente': np.mean(backtesting_results) > 0.25
}

print("\nRelatório de Conformidade BCB 303:")
print("==================================")
for metric, status in compliance.items():
    print(f"{metric}: {'✅ Aprovado' if status else '❌ Reprovado'}")

print(f"\nMétricas Detalhadas:")
print(f"- KS: {ks:.4f}")
print(f"- Gini: {gini:.4f}")
print(f"- PSI: {psi:.4f}")
print(f"- Média Backtesting: {np.mean(backtesting_results):.4f}")