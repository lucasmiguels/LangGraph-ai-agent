import re
from ..models import AgentState
from ..config import logger, FORBIDDEN_SQL_KEYWORDS, ALLOWED_TABLES

def sql_validator(state: AgentState) -> dict:
    """
    Verifica o SQL gerado contra um conjunto de regras de segurança:
    1. Impede a execução de comandos que modificam dados (UPDATE, DELETE, etc.).
    2. Garante que apenas as tabelas permitidas sejam consultadas.
    """
    logger.info(">> Nó: Validador de SQL")
    sql_query = state.get("sql_query", "").upper()

    for keyword in FORBIDDEN_SQL_KEYWORDS:
        # \b para garantir que estamos verificando a palavra inteira 
        if re.search(r'\b' + keyword + r'\b', sql_query):
            error_message = f"Validação falhou: A consulta contém a palavra-chave proibida '{keyword}'."
            logger.warning(error_message)
            return {"error": error_message}

    # Extrai todos os nomes de tabelas entre crases (`) da consulta
    tables_in_query = re.findall(r'`([^`]+)`', sql_query.lower())
    for table in tables_in_query:
        if table not in ALLOWED_TABLES:
            error_message = f"Validação falhou: A consulta tenta acessar uma tabela não permitida: '{table}'."
            logger.warning(error_message)
            return {"error": error_message}
            
    logger.info("Validação do SQL bem-sucedida. A consulta é segura para execução.")
    # Não deve alterar nada se a consulta for segura
    return {}