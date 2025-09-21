from ..config import logger, CHAMADOS_TABLE_FULL_PATH, CATEGORICAL_COLUMNS
from ..bigquery import get_bq_client
from ..models import AgentState

def category_fetcher(state: AgentState) -> dict:
    """
    Busca os valores únicos das colunas 'tipo', 'subtipo' e 'categoria'
    para dar contexto ao LLM sobre os filtros de texto corretos.
    """
    logger.info(">> Nó: Buscador de Categorias (Category Fetcher)")
    
    context_details = []
    
    try:
        client = get_bq_client()
        logger.info("   Conectado ao BigQuery para buscar categorias.")
        
        for column in CATEGORICAL_COLUMNS:
            # Esta consulta busca os valores distintos da coluna especificada.
            query = f"SELECT DISTINCT {column} FROM `{CHAMADOS_TABLE_FULL_PATH}` WHERE {column} IS NOT NULL ORDER BY {column}"
            
            results = client.query(query).to_dataframe()
            
            # Formata os valores em uma lista de strings para o prompt
            values = results[column].tolist()
            context_details.append(f"Valores possíveis para a coluna '{column}':\n{values}\n")

        formatted_context = "\n".join(context_details)
        logger.info("   Contexto de categorias obtido com sucesso.")
        logger.debug(f"   Contexto de categorias: {formatted_context}")
        
        return {"category_context": formatted_context}

    except Exception as e:
        logger.error(f"   Erro ao buscar categorias: {e}")
        # Retorna um contexto vazio em caso de erro para não quebrar o fluxo.
        return {"category_context": ""}
