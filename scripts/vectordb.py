import os
import sys
import chromadb
from google.cloud import bigquery
from langchain_openai import OpenAIEmbeddings

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.config import CHAMADOS_TABLE_FULL_PATH, CATEGORICAL_COLUMNS, logger
from src.bigquery import get_bq_client

CHROMA_PATH = os.path.join(project_root, "chroma_db_index")
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
COLLECTION_NAME = "categories_1746"

def main():
    """
    Busca categorias do BigQuery, gera embeddings e os armazena no ChromaDB.
    """
    logger.info("Iniciando processo de indexação de categorias para RAG...")

    try:
        client = get_bq_client()
        all_documents = []
        for column in CATEGORICAL_COLUMNS:
            logger.info(f"Buscando valores distintos para a coluna: {column}")
            query = f"SELECT DISTINCT {column} FROM `{CHAMADOS_TABLE_FULL_PATH}` WHERE {column} IS NOT NULL"
            results = client.query(query).to_dataframe()
            for value in results[column].tolist():
                all_documents.append({
                    "text": value,
                    "metadata": {"source_column": column}
                })
        logger.info(f"Total de {len(all_documents)} categorias únicas encontradas.")
    except Exception as e:
        logger.error(f"Falha ao buscar dados do BigQuery: {e}")
        return

    logger.info(f"Carregando modelo de embedding: {EMBEDDING_MODEL_NAME} (isso pode levar um momento)...")
    embedding_function = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, api_key=os.getenv("OPENAI_API_KEY"))

    logger.info(f"Configurando ChromaDB no diretório: {CHROMA_PATH}")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    if COLLECTION_NAME in [c.name for c in chroma_client.list_collections()]:
        logger.warning(f"Coleção '{COLLECTION_NAME}' já existe. Removendo para recriar.")
        chroma_client.delete_collection(name=COLLECTION_NAME)

    collection = chroma_client.create_collection(name=COLLECTION_NAME)
    
    logger.info("Iniciando a adição de documentos ao ChromaDB (pode levar vários minutos)...")
    
    batch_size = 200
    for i in range(0, len(all_documents), batch_size):
        batch = all_documents[i:i + batch_size]
        
        collection.add(
            documents=[item['text'] for item in batch],
            metadatas=[item['metadata'] for item in batch],
            ids=[f"cat_{i+j}" for j in range(len(batch))] # IDs únicos são necessários
        )
        logger.info(f"Processado lote {i//batch_size + 1}/{(len(all_documents)//batch_size) + 1}")

    logger.info("Indexação concluída com sucesso!")
    count = collection.count()
    logger.info(f"A coleção '{COLLECTION_NAME}' agora contém {count} itens.")

if __name__ == "__main__":
    main()