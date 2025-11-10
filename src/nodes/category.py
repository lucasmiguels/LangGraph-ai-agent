import os
import chromadb
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from ..config import logger, CHAMADOS_TABLE_FULL_PATH, CATEGORICAL_COLUMNS
from ..bigquery import get_bq_client
from ..models import AgentState, format_chat_history
from ..llm import make_llm

load_dotenv()
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
CHROMA_PATH = os.path.join(project_root, "chroma_db_index")
COLLECTION_NAME = "categories_1746"

def _expand_query_with_context(state: AgentState) -> str:
    """
    Expande a pergunta atual com contexto relevante da conversa anterior.
    Retorna a pergunta original se não houver histórico.
    """
    current_question = state['messages'][-1].content
    chat_history = format_chat_history(state['messages'][:-1])
    
    if not chat_history:
        logger.info("Sem histórico de conversa. Usando pergunta original.")
        return current_question
    
    # Se houver histórico, expande a query usando LLM
    llm = make_llm()
    prompt = f"""
        {chat_history}

        A pergunta atual do usuário é: "{current_question}"

        Sua tarefa é criar uma versão expandida da pergunta atual que inclua APENAS os termos-chave relevantes mencionados na conversa anterior que são necessários para entender a pergunta.

        IMPORTANTE:
        - Se a pergunta atual for completa e auto-suficiente, retorne-a exatamente como está.
        - Se a pergunta atual for contextual (ex: "e os com menos chamados?", "e quais os bairros?"), extraia APENAS os termos-chave da conversa anterior que completam o sentido (ex: tipo de chamado, período, categoria mencionada).
        - NÃO adicione contexto conversacional, explicações, ou informações irrelevantes.
        - NÃO adicione informações sobre resultados anteriores ou comparações.
        - Foque apenas em termos que seriam úteis para buscar categorias em um banco de dados

        Retorne APENAS a pergunta expandida com os termos-chave necessários, sem explicações adicionais.
        """
    
    try:
        expanded = llm.invoke(prompt).content.strip()
        expanded = expanded.strip('"').strip("'")
        logger.info(f"Query expandida: '{current_question}' -> '{expanded}'")
        return expanded
    except Exception as e:
        logger.warning(f"Falha ao expandir query: {e}. Usando pergunta original.")
        return current_question

def _fetch_from_rag(state: AgentState) -> str | None:
    """
    Busca categorias textuais e suas respectivas colunas utilizando um banco vetorial (RAG).
    """
    logger.info("Tentando buscar categorias via RAG (ChromaDB)...")
    expanded_query = _expand_query_with_context(state)
    try:
        embedding_function = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, api_key=os.getenv("OPENAI_API_KEY"))
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)

        results = collection.query(
            query_texts=[expanded_query],
            n_results=5,
            include=["documents", "metadatas"]
        )

        if not results or not results.get('documents') or not results['documents'][0]:
            logger.warning("Busca RAG não retornou resultados.")
            return None

        context_details = ""
        docs = results['documents'][0]
        metadatas = results['metadatas'][0]

        for i, doc in enumerate(docs):
            source_column = metadatas[i].get('source_column', 'desconhecida')
            context_details += f"- Termo encontrado: '{doc}' (obtido da coluna '{source_column}')\n"
        
        logger.info("Contexto obtido com sucesso via RAG.")
        return context_details

    except Exception as e:
        logger.warning(f"Falha ao buscar categorias via RAG: {e}. Acionando fallback.")
        return None

def _fetch_from_bigquery() -> str:
    """
    Busca o contexto diretamente do BigQuery (método de fallback).
    """
    logger.info("Método de fallback acionado: buscando categorias do BigQuery...")
    context_details = []

    try:
        client = get_bq_client()
        logger.info(f"Conectado ao BigQuery para buscar categorias.")
        
        for column in CATEGORICAL_COLUMNS:
            query = f"SELECT DISTINCT {column} FROM `{CHAMADOS_TABLE_FULL_PATH}` WHERE {column} IS NOT NULL ORDER BY {column}"
            results = client.query(query).to_dataframe()
            values = results[column].tolist()
            context_details.append(f"Valores possíveis para a coluna '{column}':\n{values}\n")
        
        formatted_context = "\n".join(context_details)
        logger.info("   Contexto de categorias obtido com sucesso.")
        logger.debug(f"   Contexto de categorias: {formatted_context}")
        
        return formatted_context
    except Exception as e:
        logger.error(f"Falha também no método de fallback (BigQuery): {e}", exc_info=True)
        return ""

def category_fetcher(state: AgentState) -> dict:
    """
    Orquestra a busca de categorias: tenta o método RAG primeiro e,
    se falhar, usa a consulta direta ao BigQuery como fallback.
    """
    logger.info(">> Nó: Buscador de Categorias (com Fallback)")

    rag_context = _fetch_from_rag(state)

    if rag_context is not None:
        return {"category_context": rag_context}
    else:
        # Falha no RAG. Aciona o método secundário e mais lento (BigQuery).
        bq_context = _fetch_from_bigquery()
        return {"category_context": bq_context}
