from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate


class RegulatoryAssistant:
    def __init__(self, regulations_path):
        self.vector_store = self._load_documents(regulations_path)
        self.llm = ChatOpenAI(model="gpt-4-turbo")

    def _load_documents(self, path):
        loaders = [PyPDFLoader(os.path.join(path, f))
                   for f in os.listdir(path) if f.endswith('.pdf')]
        documents = [page for loader in loaders for page in loader.load()]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)

        return FAISS.from_documents(chunks, OpenAIEmbeddings())

    def query_regulation(self, question):
        # Recuperação de contexto
        docs = self.vector_store.similarity_search(question, k=4)
        context = "\n\n".join([d.page_content for d in docs])

        # Engenharia de prompt especializada
        prompt_template = """
        Você é um especialista em regulamentações do Banco Central (BCB 303 e Basiléia II).
        Responda com base EXCLUSIVAMENTE no contexto fornecido:

        Contexto:
        {context}

        Pergunta: {question}

        Estruture sua resposta em:
        1. Fundamentação regulatória
        2. Requisitos aplicáveis
        3. Implicações para validação de modelos
        """
        prompt = PromptTemplate.from_template(prompt_template)

        # Geração da resposta
        chain = prompt | self.llm
        return chain.invoke({"context": context, "question": question})