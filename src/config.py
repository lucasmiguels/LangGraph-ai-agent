import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("DataAgentLogger")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler("agent.log", mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(funcName)s - %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BIGQUERY_PROJECT = os.getenv("BIGQUERY_PROJECT")

CHAMADOS_TABLE_FULL_PATH = "datario.adm_central_atendimento_1746.chamado"
BAIRROS_TABLE_FULL_PATH  = "datario.dados_mestres.bairro"

CATEGORICAL_COLUMNS = ["tipo", "categoria", "subtipo"]

FORBIDDEN_SQL_KEYWORDS = [
    "UPDATE", "DELETE", "INSERT", "DROP", "CREATE", 
    "ALTER", "TRUNCATE", "MERGE", "GRANT", "REVOKE"
]

ALLOWED_TABLES = [
    CHAMADOS_TABLE_FULL_PATH,
    BAIRROS_TABLE_FULL_PATH
]