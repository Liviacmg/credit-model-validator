import streamlit as st
import pandas as pd
from src.validation_framework import metrics_calculator
from src.generative_ai import regulatory_rag

# Configuração da página
st.set_page_config(page_title="Validador de Modelos BCB 303", layout="wide")

# Sidebar
st.sidebar.header("Parâmetros de Validação")
confidence_level = st.sidebar.slider("Nível de Confiança", 0.90, 0.99, 0.95)
ks_threshold = st.sidebar.number_input("Limite KS", 0.25, 0.50, 0.30)

# Página principal
tab1, tab2, tab3 = st.tabs(["Validação", "Regulatório", "Documentação"])

with tab1:
    # Upload de dados
    uploaded_file = st.file_uploader("Carregar portfólio de crédito")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Cálculo automático de métricas
        ks = metrics_calculator.calculate_ks(df['default'], df['PD'])
        gini = metrics_calculator.calculate_gini(df['default'], df['PD'])

        # Visualização
        col1, col2 = st.columns(2)
        col1.metric("Estatística KS", f"{ks:.3f}",
                    "Aprovado" if ks > ks_threshold else "Reprovado")
        col2.metric("Coeficiente Gini", f"{gini:.2%}")

with tab2:
    # Assistente regulatório
    st.header("Consultor BCB 303")
    query = st.text_input("Faça sua pergunta regulatória:")
    if query:
        rag = regulatory_rag.RegulatoryAssistant("data/regulations")
        response = rag.query_regulation(query)
        st.markdown(response)