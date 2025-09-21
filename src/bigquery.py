from google.cloud import bigquery
from .config import BIGQUERY_PROJECT

def get_bq_client():
    return bigquery.Client(project=BIGQUERY_PROJECT)
