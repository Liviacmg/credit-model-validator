import pytest
import os
from src.generative_ai.regulatory_rag import RegulatoryRAGSystem
from langchain_community.document_loaders import PyPDFLoader


@pytest.fixture(scope="module")
def rag_system():
    # Setup - carregar sistema RAG
    os.makedirs("test_data/regulations", exist_ok=True)

    # Criar documento de teste
    with open("test_data/regulations/test_doc.pdf", "w") as f:
        f.write("Test document content")

    system = RegulatoryRAGSystem(regulations_path="test_data/regulations")
    yield system
    # Teardown
    import shutil
    shutil.rmtree("test_data")


def test_query_regulation(rag_system):
    response = rag_system.query_regulation("Test query")
    assert "Test document" in response
    assert "Fundamentação regulatória" in response


def test_report_generation():
    from src.generative_ai.report_generator import ReportGenerator

    # Mock de resultados
    validation_results = {
        "KS": 0.35,
        "GINI": 0.45,
        "PSI": 0.08,
        "AUC": 0.82
    }

    generator = ReportGenerator()
    report = generator.generate_validation_report(
        validation_results,
        "Random Forest",
        "PD",
        "Portfólio de crédito corporativo - 10k amostras"
    )

    assert "Relatório de Validação" in report
    assert "BCB 303" in report
    assert "Recomendações" in report


def test_prompt_templates():
    from src.generative_ai.prompt_engineering import (
        get_validation_prompt_template,
        get_regulatory_qa_prompt_template
    )

    val_prompt = get_validation_prompt_template()
    assert "BCB 303" in val_prompt.template

    qa_prompt = get_regulatory_qa_prompt_template()
    assert "Banco Central" in qa_prompt.template