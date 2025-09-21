from ..config import logger
from ..bigquery import get_bq_client
from ..models import AgentState

def sql_executor(state: AgentState) -> dict:
    """
    Executa a consulta no BigQuery e retorna o resultado.
    """
    logger.info(">> Nó: Executor de SQL")
    sql_query = state["sql_query"]

    try:
        client = get_bq_client()
        logger.info("   Conectado ao BigQuery.")
        query_job = client.query(sql_query)
        
        results = query_job.to_dataframe()
        query_result = results.to_dict('records')
        
        logger.info(f"   Consulta executada com sucesso. {len(query_result)} linhas retornadas.")
        logger.debug(f"Resultado da consulta (amostra): {query_result[:5]}") # Loga as 5 primeiras linhas
        return {"query_result": query_result}
    except Exception as e:
        logger.error(f"   Erro na execução da consulta: {e}")
        return {"error": f"Erro ao executar a consulta no BigQuery: {e}"}