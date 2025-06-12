import markdown
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from .prompt_engineering import (
    get_validation_prompt_template,
    get_report_summary_prompt_template
)


class ReportGenerator:
    def __init__(self, llm_model="gpt-4-turbo"):
        self.llm = ChatOpenAI(model=llm_model, temperature=0.2)
        self.output_parser = StrOutputParser()

    def generate_validation_report(self, validation_results, model_type, parameter, dataset_description):
        """
        Gera relatório de validação de modelo

        Args:
            validation_results (dict): Resultados das métricas de validação
            model_type (str): Tipo de modelo (Random Forest, XGBoost, etc.)
            parameter (str): PD, LGD ou EAD
            dataset_description (str): Descrição do dataset

        Returns:
            str: Relatório formatado em Markdown
        """
        prompt = get_validation_prompt_template()
        chain = prompt | self.llm | self.output_parser

        return chain.invoke({
            "model_type": model_type,
            "parameter": parameter,
            "dataset_description": dataset_description,
            "validation_results": self._format_validation_results(validation_results)
        })

    def generate_executive_summary(self, full_report):
        """
        Gera sumário executivo de um relatório técnico

        Args:
            full_report (str): Relatório técnico completo

        Returns:
            str: Sumário executivo em Markdown
        """
        prompt = get_report_summary_prompt_template()
        chain = prompt | self.llm | self.output_parser
        return chain.invoke({"full_report": full_report})

    def _format_validation_results(self, results):
        """Formata resultados para inclusão no prompt"""
        formatted = "\n".join([f"- **{k}**: {v}" for k, v in results.items()])
        return f"**Métricas de Validação:**\n{formatted}"

    def convert_to_html(self, markdown_report):
        """Converte relatório Markdown para HTML"""
        return markdown.markdown(markdown_report)