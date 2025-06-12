from langchain.prompts import PromptTemplate


def get_validation_prompt_template():
    """
    Retorna template de prompt para validação de modelos
    """
    return PromptTemplate.from_template(
        """Você é um especialista em risco de crédito e conformidade regulatória BCB 303. 
        Analise os seguintes resultados de validação de modelo e gere um relatório técnico:

        **Contexto:**
        - Modelo: {model_type}
        - Parâmetro: {parameter} (PD/LGD/EAD)
        - Dataset: {dataset_description}
        {validation_results}

        **Tarefa:**
        1. Avalie a conformidade com BCB 303 considerando:
           - Discriminação (KS, Gini)
           - Estabilidade (PSI)
           - Precisão (Backtesting)
        2. Destaque pontos fortes e vulnerabilidades
        3. Recomende ações de melhoria
        4. Formate a saída em Markdown com seções claras

        **Estrutura Esperada:**
        # Relatório de Validação - {parameter}
        ## Conformidade Regulatória
        ## Análise de Performance
        ## Recomendações
        """
    )


def get_regulatory_qa_prompt_template():
    """
    Retorna template de prompt para Q&A regulatório
    """
    return PromptTemplate.from_template(
        """Você é um especialista em regulamentações do Banco Central (BCB 303 e Basiléia II).
        Responda EXCLUSIVAMENTE com base no contexto fornecido:

        **Contexto:**
        {context}

        **Pergunta:**
        {question}

        **Instruções:**
        - Cite os artigos/seções relevantes
        - Explique as implicações para validação de modelos
        - Seja técnico mas claro
        - Formate referências como [Documento, Página X]
        """
    )


def get_report_summary_prompt_template():
    """
    Retorna template para sumarização de relatórios
    """
    return PromptTemplate.from_template(
        """Resuma o seguinte relatório técnico para executivos de risco:

        **Relatório Completo:**
        {full_report}

        **Instruções:**
        - Máximo de 200 palavras
        - Destaque pontos críticos de conformidade
        - Use linguagem acessível para não-especialistas
        - Inclua recomendação principal
        - Formato: Markdown com tópicos
        """
    )