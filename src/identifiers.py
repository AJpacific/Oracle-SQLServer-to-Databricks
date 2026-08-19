"""
identifiers.py - safe identifier quoting and literal escaping.

Oracle and Databricks have different identifier rules:
  * Databricks (Delta / Unity Catalog) quotes identifiers with backticks.
  * Oracle quotes identifiers with double quotes and folds *unquoted*
    identifiers to UPPER CASE. Because our source metadata (ALL_TAB_COLUMNS)
    stores names exactly as created (usually upper case), we quote them as-is
    and let the caller pass the stored casing.

We deliberately reject obviously invalid / dangerous identifiers instead of
silently sanitizing them, so bad metadata fails loudly. Oracle object names may
contain letters, digits, underscore, dollar ($) and hash (#); spaces are allowed
for quoted identifiers. Hyphens are rejected because "--" is a SQL comment.

All functions here are pure (no Spark / no dbutils) so they can be unit tested.
"""

from __future__ import annotations

import re

# Allow-list for object names coming from the Oracle data dictionary.
# Letters, digits, underscore, space, dollar and hash are permitted.
_VALID_IDENTIFIER = re.compile(r"^[A-Za-z_#$][A-Za-z0-9_ #$]*$")

# Allow-list for SQL Server object names. Same base set plus square brackets,
# because bracket-quoting neutralizes ``]`` by doubling it. Dangerous characters
# for T-SQL (``;``, quotes, hyphen '--' comment) are still rejected so an
# identifier can never break out of its bracket quoting.
_VALID_SQLSERVER_IDENTIFIER = re.compile(r"^[A-Za-z_#$\[][A-Za-z0-9_ #$\[\]]*$")


class IdentifierError(ValueError):
    "Raised when an identifier fails validation."


def validate_identifier(name: str) -> str:
    "Validate a raw identifier and return it unchanged (stripped), or raise."
    if name is None:
        raise IdentifierError("Identifier is None")
    if not isinstance(name, str):
        raise IdentifierError(f"Identifier must be a string, got {type(name)!r}")
    stripped = name.strip()
    if not stripped:
        raise IdentifierError("Identifier is empty")
    if len(stripped) > 128:
        raise IdentifierError(f"Identifier too long: {stripped[:40]}...")
    if not _VALID_IDENTIFIER.match(stripped):
        raise IdentifierError(f"Invalid identifier characters: {name!r}")
    return stripped


def validate_sqlserver_identifier(name: str) -> str:
    "Validate a raw SQL Server identifier (allows ``]``) and return it, or raise."
    if name is None:
        raise IdentifierError("Identifier is None")
    if not isinstance(name, str):
        raise IdentifierError(f"Identifier must be a string, got {type(name)!r}")
    stripped = name.strip()
    if not stripped:
        raise IdentifierError("Identifier is empty")
    if len(stripped) > 128:
        raise IdentifierError(f"Identifier too long: {stripped[:40]}...")
    if not _VALID_SQLSERVER_IDENTIFIER.match(stripped):
        raise IdentifierError(f"Invalid SQL Server identifier characters: {name!r}")
    return stripped


def quote_databricks(identifier: str) -> str:
    """Quote a single Databricks identifier with backticks, escaping backticks."""
    validated = validate_identifier(identifier)
    return "`" + validated.replace("`", "``") + "`"


def quote_oracle(identifier: str) -> str:
    """Quote a single Oracle identifier with double quotes, escaping quotes.

    We do NOT upper-case here: Oracle stores names as created and our metadata
    reads them back exactly, so quoting preserves the true stored name.
    """
    validated = validate_identifier(identifier)
    return '"' + validated.replace('"', '""') + '"'


def quote_sqlserver(identifier: str) -> str:
    """Quote a single SQL Server identifier with brackets, escaping ``]`` as ``]]``.

    SQL Server folds *unquoted* identifiers per the database collation, so we
    never upper-case here: the metadata reader returns names exactly as stored
    (dbo, Employees, ...) and bracket-quoting preserves that casing. The name is
    validated first so unvalidated identifiers can never be interpolated.
    """
    validated = validate_sqlserver_identifier(identifier)
    return "[" + validated.replace("]", "]]") + "]"


def oracle_fqn(schema: str, table: str) -> str:
    """Fully qualified, quoted Oracle object name: "OWNER"."TABLE"."""
    return f"{quote_oracle(schema)}.{quote_oracle(table)}"


def sqlserver_fqn(schema: str, table: str, database: str = None) -> str:
    """Fully qualified, bracket-quoted SQL Server object name.

    ``[schema].[table]`` or, when a database is supplied,
    ``[database].[schema].[table]``.
    """
    if database:
        return (
            f"{quote_sqlserver(database)}."
            f"{quote_sqlserver(schema)}."
            f"{quote_sqlserver(table)}"
        )
    return f"{quote_sqlserver(schema)}.{quote_sqlserver(table)}"


def databricks_fqn(catalog: str, schema: str, table: str) -> str:
    """Fully qualified, quoted Databricks name: `cat`.`schema`.`table`."""
    return (
        f"{quote_databricks(catalog)}."
        f"{quote_databricks(schema)}."
        f"{quote_databricks(table)}"
    )


def escape_string_literal(value) -> str:
    """Return a SQL string literal (single quoted, quotes doubled).

    None becomes SQL NULL. Used when interpolating run_id etc. into spark.sql.
    """
    if value is None:
        return "NULL"
    return "'" + str(value).replace("'", "''") + "'"


# Server (host[\instance][,port] / host:port) and database names are interpolated
# into JDBC connection strings, so they get their own conservative validation to
# stop connection-string injection. We permit the small set of characters that
# legitimately appear in a SQL Server / Oracle server or database name and reject
# anything that could break out of the URL (';', '=', quotes, whitespace, etc.).
_VALID_SERVER = re.compile(r"^[A-Za-z0-9_.\-\\]+(?:,[0-9]{1,5})?(?::[0-9]{1,5})?$")
_VALID_DATABASE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$#]*$")


def validate_server(server: str) -> str:
    """Validate a source server/host token before it is placed in a JDBC URL.

    Accepts ``host``, ``host\\instance``, ``host,port`` and ``host:port`` forms.
    Rejects anything containing characters that could inject extra JDBC
    connection properties (';', '=', whitespace, quotes, ...).
    """
    if server is None:
        raise IdentifierError("Server is None")
    if not isinstance(server, str):
        raise IdentifierError(f"Server must be a string, got {type(server)!r}")
    stripped = server.strip()
    if not stripped:
        raise IdentifierError("Server is empty")
    if len(stripped) > 255:
        raise IdentifierError("Server name too long")
    if not _VALID_SERVER.match(stripped):
        raise IdentifierError(f"Invalid server value: {server!r}")
    return stripped


def validate_database(database: str) -> str:
    """Validate a source database name before it is placed in a JDBC URL.

    Only letters, digits, underscore, ``$`` and ``#`` are permitted (must start
    with a letter or underscore), which is safe to embed as ``databaseName=...``.
    """
    if database is None:
        raise IdentifierError("Database is None")
    if not isinstance(database, str):
        raise IdentifierError(f"Database must be a string, got {type(database)!r}")
    stripped = database.strip()
    if not stripped:
        raise IdentifierError("Database is empty")
    if len(stripped) > 128:
        raise IdentifierError("Database name too long")
    if not _VALID_DATABASE.match(stripped):
        raise IdentifierError(f"Invalid database value: {database!r}")
    return stripped
