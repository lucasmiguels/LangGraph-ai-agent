from ..config import logger
from ..llm import make_llm
from ..models import AgentState, format_chat_history

_llm = make_llm()

def sql_generator(state: AgentState) -> dict:
    """
    Gera uma consulta SQL válida e eficiente para o BigQuery com base na pergunta e no schema.
    """
    logger.info(">> Nó: Gerador de SQL")
    question = state['messages'][-1].content
    chat_history = format_chat_history(state['messages'][:-1])
    schema = state["schema"]
    category_context = state.get("category_context", "")
    context_section = ""
    if category_context:
        context_section = f"""
        INFORMAÇÕES DE CONTEXTO SOBRE AS CATEGORIAS:
        Use a lista de valores abaixo para encontrar o termo e a coluna corretos para a pergunta do usuário.
        {category_context}
        """

    prompt = f"""
    Sua tarefa é ser um especialista em SQL do Google BigQuery. Seu objetivo principal é gerar uma única consulta SQL que seja **correta e funcional**.
    {chat_history}

    ESQUEMA DO BANCO DE DADOS:
    {schema}
    {context_section}

    REGRAS ESSENCIAIS:
    1.  **Nomes de Tabela:** SEMPRE use o nome completo da tabela (ex: `projeto.dataset.tabela`) nas cláusulas `FROM` e `JOIN`.

    2.  **Filtros de Texto:** Para buscas em colunas de texto (STRING), a seção "INFORMAÇÕES DE CONTEXTO SOBRE AS CATEGORIAS" contém os termos mais relevantes encontrados no banco de dados para a pergunta do usuário. Sua tarefa é usar essa informação para construir o filtro `WHERE`.
        - Use o termo mais provável da lista e, crucialmente, a coluna de origem ('source_column') informada.
        - Se o contexto fornecer uma correspondência exata, use o operador de igualdade (`=`) e a função `LOWER()`.

    3.  **Filtros de Data:** Para todos os filtros de data, use a coluna `data_inicio`.
        - Para um **dia específico** (ex: "no dia 28/11/2024"), use a função `DATE`. Ex: `WHERE DATE(data_inicio) = '2024-11-28'`.
        - Para **intervalos de datas**, use `DATE(data_inicio) BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'`.

    4.  **Seleção de Colunas:** NUNCA use `SELECT *`. Selecione apenas as colunas necessárias para responder à pergunta.

    5.  **Formato de Saída:** Retorne APENAS o código SQL, sem nenhuma explicação ou formatação de markdown.

    PERGUNTA DO USUÁRIO:
    "{question}"

    SQL GERADO:
    """
    
    logger.info("--- INÍCIO DO PROMPT PARA O GERADOR DE SQL ---")
    logger.info(prompt)
    logger.info("--- FIM DO PROMPT PARA O GERADOR DE SQL ---")

    try:
        sql_query = _llm.invoke(prompt).content
        cleaned_sql_query = sql_query.strip().replace("```sql", "").replace("```", "").strip()
        logger.info(f"   SQL Gerado: \n{cleaned_sql_query}")
        
        return {"sql_query": cleaned_sql_query}
    
    except Exception as e:
        logger.error(f"   Erro na geração de SQL: {e}")
        return {"error": "Falha ao gerar a consulta SQL."}