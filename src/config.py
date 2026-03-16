import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = PROJECT_ROOT / "agent.log"

logger = logging.getLogger("DataAgentLogger")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(funcName)s - %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BIGQUERY_PROJECT = os.getenv("BIGQUERY_PROJECT")

USE_VECTOR_DB = os.getenv("USE_VECTOR_DB", "true").lower() in ("true", "1", "yes")

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