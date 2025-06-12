# Credit Model Validator

Solução completa para validação de modelos de crédito conforme regulamentações do Banco Central (BCB 303) e Basel II, com integração de IA Generativa.

## 🚀 Principais Funcionalidades

- **Validação de Modelos de Crédito** (PD, LGD, EAD)
- **Cálculo de Métricas Reguladas**: KS, Gini, PSI
- **Consulta Regulamentar com IA Generativa** (RAG com PDFs da BCB)
- **Dashboard Interativo** com Streamlit
- **Ambiente Reproduzível com Docker**
- **Testes Automatizados para Garantia de Qualidade**

## 🧪 Como Rodar Localmente

```bash
# Clone o repositório

git clone https://github.com/seu-usuario/credit-model-validator.git

cd credit-model-validator

# Instale as dependências
pip install -r requirements.txt

# Execute o dashboard
streamlit run src/app.py 
```

## Uso

```
#Validação de Modelos

from src.validation_framework.metrics_calculator import calculate_ks, calculate_gini

ks = calculate_ks(y_true, y_pred_proba)
gini = calculate_gini(y_true, y_pred_proba)

#Consulta Regulatória

from src.generative_ai.regulatory_rag import RegulatoryRAG

rag = RegulatoryRAG("data/regulations")
response = rag.query_regulation("Requisitos para validação de modelos PD")

#Dashboard

streamlit run src/app.py
```

## 🐳 Docker (opcional)

```bash
docker build -t credit-validator .

docker run -p 8501:8501 credit-validator
```

## Estrutura Geral do Projeto
```
credit-model-validator/
├── data/                   # Dados e documentos regulatórios
├── notebooks/              # Jupyter notebooks de demonstração
├── src/                    # Código-fonte principal
│   ├── validation_framework # Módulo de validação
│   └── generative_ai       # IA Generativa
├── tests/                  # Testes automatizados
├── Dockerfile              # Configuração de container
└── requirements.txt        # Dependências
```

## 📬 Contato

Conecte-se comigo no LinkedIn (https://www.linkedin.com/in/liviacavalcanti/) ou envie um email para livia.cavalcanti.gama@gmail.com

