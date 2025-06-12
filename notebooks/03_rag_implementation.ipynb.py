# # Sistema RAG para Consulta Regulatória
# ## Implementação de IA Generativa para BCB 303 e Basileia II

import os
import re
import numpy as np
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import requests
from io import BytesIO

# ## 1. Carregamento de Documentos Regulatórios
# Download e processamento de PDFs do Banco Central


# URLs dos documentos regulatórios
regulation_urls = {
    'BCB_Resolution_303': 'https://www.bcb.gov.br/pre/normativos/res/2023/pdf/res_303_v1_P.pdf',
    'Basel_II_Framework': 'https://www.bis.org/publ/bcbs128.pdf'
}

# Criar diretório se não existir
os.makedirs('../data/regulations', exist_ok=True)

# Download e salvar documentos
for name, url in regulation_urls.items():
    response = requests.get(url)
    with open(f'../data/regulations/{name}.pdf', 'wb') as f:
        f.write(response.content)
    print(f'Downloaded {name}.pdf')

# Carregar e processar documentos
loaders = [
    PyPDFLoader('../data/regulations/BCB_Resolution_303.pdf'),
    PyPDFLoader('../data/regulations/Basel_II_Framework.pdf')
]

documents = []
for loader in loaders:
    documents.extend(loader.load())

# Pré-processar texto
for doc in documents:
    doc.page_content = re.sub(r'\s+', ' ', doc.page_content)  # Remover espaços múltiplos
    doc.page_content = re.sub(r'\n', ' ', doc.page_content)  # Remover quebras de linha


# Dividir documentos em chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len
)

chunks = text_splitter.split_documents(documents)
print(f"Total de chunks: {len(chunks)}")

# ## 2. Configuração do Sistema RAG
# Criação de vetor store e engenharia de prompts


# Criar vetor store com embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local("../data/regulations_faiss_index")

# Configurar LLM
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.3)

# Template de prompt especializado
prompt_template = PromptTemplate.from_template(
    """Você é um especialista em regulamentações bancárias do Banco Central do Brasil (BCB) e normas de Basileia.
    Responda exclusivamente com base no contexto fornecido abaixo, citando os artigos e seções relevantes.

    **Contexto:**
    {context}

    **Pergunta:**
    {question}

    **Estruture sua resposta em:**
    1. Fundamentação regulatória (cite artigo/seção)
    2. Requisitos específicos
    3. Implicações para validação de modelos
    4. Referências completas

    **Exemplo de resposta para referência:**
    "Conforme o Artigo 15 da Resolução BCB 303/2023, as instituições devem... [detalhes]"
    """
)


# ## 3. Implementação da Cadeia RAG
# Sistema completo de recuperação e geração


class RegulatoryRAGSystem:
    def __init__(self):
        self.vector_store = FAISS.load_local(
            "../data/regulations_faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        self.llm_chain = prompt_template | llm | StrOutputParser()

    def query(self, question, k=4):
        # Recuperação de documentos relevantes
        docs = self.vector_store.similarity_search(question, k=k)
        context = "\n\n".join([d.page_content for d in docs])

        # Geração da resposta
        response = self.llm_chain.invoke({
            "context": context,
            "question": question
        })

        # Formatar referências
        references = "\n\n**Referências:**\n"
        for i, doc in enumerate(docs):
            source = doc.metadata['source'].split('/')[-1]
            page = doc.metadata.get('page', 'N/A')
            references += f"{i + 1}. {source} (Página {page})\n"

        return response + references



# Testar o sistema
rag_system = RegulatoryRAGSystem()

# ## 4. Demonstração do Sistema
# Testes com perguntas regulatórias típicas


questions = [
    "Quais são os requisitos mínimos para validação de modelos PD segundo BCB 303?",
    "Como deve ser realizada a análise de estabilidade para modelos de crédito?",
    "Quais as exigências de Basileia II para cálculo de LGD?",
    "Como documentar os processos de validação de modelos?",
    "Quais são os testes obrigatórios para modelos IRB?"
]

for i, question in enumerate(questions):
    print(f"\n{'=' * 50}")
    print(f"Pergunta {i + 1}: {question}")
    print(f"{'=' * 50}")
    response = rag_system.query(question)
    print(response)
    print("\n")

# ## 5. Avaliação do Sistema
# Métricas de qualidade de respostas

# Exemplo de avaliação (em ambiente real seria mais completo)
sample_questions = {
    "Quais são os componentes do capital regulatório segundo Basileia II?": {
        "expected_keywords": ["Tier 1", "Tier 2", "capital mínimo", "requisitos de adequação"],
        "expected_sources": ["Basel_II_Framework"]
    },
    "Qual o prazo para implementação da Resolução 303?": {
        "expected_keywords": ["vigência", "180 dias", "implementação gradativa"],
        "expected_sources": ["BCB_Resolution_303"]
    }
}

def evaluate_response(question, response, expected):
    # Verificar presença de keywords
    keywords_score = sum(1 for kw in expected["expected_keywords"] if kw.lower() in response.lower())
    keywords_score = keywords_score / len(expected["expected_keywords"])

    # Verificar fontes
    sources_score = sum(1 for src in expected["expected_sources"] if src in response)
    sources_score = sources_score / len(expected["expected_sources"])

    return {
        "keywords_score": keywords_score,
        "sources_score": sources_score,
        "total_score": (keywords_score + sources_score) / 2
    }


# Executar avaliação
results = []
for question, expected in sample_questions.items():
    response = rag_system.query(question)
    evaluation = evaluate_response(question, response, expected)
    results.append({
        "question": question,
        "response": response[:500] + "..." if len(response) > 500 else response,
        **evaluation
    })

# Exibir resultados
results_df = pd.DataFrame(results)
print("\nResultados da Avaliação do Sistema RAG:")
print(results_df[['question', 'keywords_score', 'sources_score', 'total_score']])