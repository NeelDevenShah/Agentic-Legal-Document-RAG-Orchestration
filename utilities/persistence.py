from __future__ import annotations

from urllib.parse import unquote, urlparse

from config import AppConfig


def _parse_mysql_url(mysql_url: str) -> dict[str, str | int]:
    parsed = urlparse(mysql_url.replace("mysql://", "http://", 1))
    database = parsed.path.lstrip("/").split("?", 1)[0]
    if not database:
        raise ValueError(f"Invalid MYSQL_URL (missing database): {mysql_url!r}")

    return {
        "user": unquote(parsed.username or ""),
        "password": unquote(parsed.password or ""),
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 3306,
        "database": database,
    }


def clear_mysql_tables(config: AppConfig) -> str:
    if not config.mysql_url:
        return "MySQL: not configured (skipped)."

    try:
        import pymysql
    except ImportError as exc:
        raise RuntimeError(
            "pymysql is required to clear MySQL tables. Install dependencies from requirements.txt."
        ) from exc

    connection_params = _parse_mysql_url(config.mysql_url)
    tables = (
        config.mysql_messages_table,
        config.mysql_memory_table,
        config.mysql_sessions_table,
    )

    connection = pymysql.connect(
        host=str(connection_params["host"]),
        port=int(connection_params["port"]),
        user=str(connection_params["user"]),
        password=str(connection_params["password"]),
        database=str(connection_params["database"]),
        autocommit=True,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute("SET FOREIGN_KEY_CHECKS=0")
            for table in tables:
                cursor.execute(f"TRUNCATE TABLE `{table}`")
            cursor.execute("SET FOREIGN_KEY_CHECKS=1")
    finally:
        connection.close()

    return f"MySQL: cleared tables {', '.join(tables)}."


def clear_persistence_stores(config: AppConfig, *, qdrant_clear_message: str) -> str:
    lines = [qdrant_clear_message]
    if config.mysql_url:
        lines.append(clear_mysql_tables(config))
    return "\n".join(lines)
