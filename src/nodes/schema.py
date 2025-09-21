from ..config import logger, CHAMADOS_TABLE_FULL_PATH, BAIRROS_TABLE_FULL_PATH
from ..bigquery import get_bq_client
from ..models import AgentState

def schema_fetcher(state: AgentState) -> dict:
    """
    Busca o esquema (colunas e tipos) das tabelas no BigQuery
    e o formata em uma string para ser usada no prompt do LLM.
    """
    logger.info(">> Nó: Buscador de Esquema (Schema Fetcher)")
    

    table_full_paths = [CHAMADOS_TABLE_FULL_PATH, BAIRROS_TABLE_FULL_PATH]
    
    schema_details = []
    
    try:
        client = get_bq_client()
        logger.info(f"   Conectado ao BigQuery para buscar esquema.")
        
        for table_path in table_full_paths:
            table = client.get_table(table_path)
            
            # Formata o nome da tabela
            schema_details.append(f"Tabela: `{table.full_table_id.replace(':', '.')}`")
            
            # Formata cada coluna e seu tipo
            for column in table.schema:
                schema_details.append(f"- {column.name} ({column.field_type})")
            
            schema_details.append("") 

        formatted_schema = "\n".join(schema_details)
        logger.debug(f"   Esquema obtido com sucesso:\n{formatted_schema}")
        
        return {"schema": formatted_schema}

    except Exception as e:
        logger.error(f"   Erro ao buscar o esquema: {e}")
        return {"error": f"Não foi possível buscar o esquema das tabelas: {e}"}

