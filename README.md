# Credit Model Validator

Projeto para validação de modelos de crédito com foco em requisitos regulatórios (ex: BCB 303, Basileia II) e aplicação de IA generativa para consulta normativa.

## Estrutura do Projeto

```
credit-model-validator/
├── data/                        # Dados sintéticos e regulatórios
│   └── regulations/            # PDFs de resoluções e guias
├── notebooks/                  # Jupyter Notebooks de modelagem
├── src/                        # Código-fonte do projeto
│   ├── validation_framework/   # Módulos de validação (KS, Gini, PSI)
│   └── generative_ai/          # Implementação RAG e IA Generativa
├── tests/                      # Testes automatizados
├── requirements.txt            # Dependências do projeto
├── Dockerfile                  # Configuração para container Docker
└── README.md                   # Documentação do projeto
```

## Como usar

```bash
docker build -t credit-validator .

docker run -p 8501:8501 credit-validator
```

