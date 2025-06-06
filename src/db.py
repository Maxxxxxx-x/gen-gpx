from psycopg2.extensions import connection
from typing import List, Tuple, Any
import config  # 改為絕對匯入
import psycopg2
import os


SQL_QUERY = """
SELECT id, trails
FROM records
WHERE trails IN (
    SELECT trails
    FROM records
    WHERE trails LIKE '臺北大縱走%' AND distance > 400 AND duration > 0
    GROUP BY trails
    HAVING COUNT(*) >= 10
)
ORDER BY trails, id
LIMIT %s OFFSET %s;
"""

CONNECTION_STR = "host=%s port=%s user=%s password=%s dbname=%s"


def get_connection_str() -> str:
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    username = os.getenv("DB_USER")
    password = os.getenv("DB_PASS")
    db_name = os.getenv("DB_NAME")
    if not (host and port and username and password and db_name):
        print("Missing ENV variable")
        return ""
    return CONNECTION_STR % (host, port, username, password, db_name)


def connect_database(conn_str: str) -> connection:
    return psycopg2.connect(conn_str)


def get_records_from_database(
        conn: connection,
        offset: int
) -> List[Tuple[Any, ...]]:
    cursor = conn.cursor()
    cursor.execute(SQL_QUERY, (config.query_limit, offset))  # 使用 config.query_limit
    result = cursor.fetchall()
    cursor.close()
    return result
