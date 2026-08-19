# Databricks notebook source
# MAGIC %md
# MAGIC # 00_TEST_SQLSERVER_CONNECTION
# MAGIC Run this first for SQL Server sources. It validates the shared adapter,
# MAGIC JDBC driver, TLS/login, a sample read, column metadata, and PK metadata.

# COMMAND ----------
# MAGIC %run ./_common

# COMMAND ----------
# MAGIC %md ### 1. Module and driver validation

# COMMAND ----------
import json

modules = [
    "src.identifiers",
    "src.sqlserver_sql_builder",
    "src.ddl_builder",
    "src.strategy",
    "src.crosssourcetypemapper",
    "src.control_repository",
    "src.source_adapters.factory",
]

for module_name in modules:
    try:
        __import__(module_name)
        print(f"OK  {module_name}")
    except Exception:
        flat_name = module_name.replace("src.", "")
        __import__(flat_name)
        print(f"OK  {module_name} (flat)")

loader = (
    spark._jvm.java.lang.Thread
    .currentThread()
    .getContextClassLoader()
)
driver_class = loader.loadClass(
    "com.microsoft.sqlserver.jdbc.SQLServerDriver"
)
print(
    "Microsoft SQL Server JDBC driver loaded successfully:",
    driver_class.getName(),
)

# COMMAND ----------
# MAGIC %md ### 2. Source-table test configuration

# COMMAND ----------
for widget_name in (
    "test_server",
    "test_database",
    "test_schema",
    "test_table",
):
    try:
        dbutils.widgets.remove(widget_name)
    except Exception:
        pass

# Host can remain blank because the adapter reads sqlserver-host from secrets.
dbutils.widgets.text(
    "test_server",
    "",
    "SQL Server host (blank = sqlserver-host secret)",
)
dbutils.widgets.text(
    "test_database",
    "free-sql-db-8215349",
    "SQL Server database",
)
dbutils.widgets.text(
    "test_schema",
    "accelerator_demo",
    "SQL Server schema",
)
dbutils.widgets.text(
    "test_table",
    "Customers",
    "SQL Server table",
)

test_server = dbutils.widgets.get("test_server").strip() or None
test_database = dbutils.widgets.get("test_database").strip()
test_schema = dbutils.widgets.get("test_schema").strip()
test_table = dbutils.widgets.get("test_table").strip()

print(
    "Effective SQL Server object:",
    f"{test_database}.{test_schema}.{test_table}",
)

adapter = get_source_adapter(
    "sqlserver",
    source_server=test_server,
    source_database=test_database,
    secret_provider=_secret_provider,
    secret_scope=SQLSERVER_SECRET_SCOPE,
)
print(
    "Adapter:", adapter.source_system,
    "| driver:", adapter.DRIVER,
    "| scope:", SQLSERVER_SECRET_SCOPE,
)

_url, _props = adapter.get_jdbc_url_and_props(
    test_server,
    test_database,
)
print(
    "JDBC URL (redacted):",
    adapter.redact_jdbc_url(_url),
)

# COMMAND ----------
# MAGIC %md ### 3. SELECT 1 connectivity test

# COMMAND ----------
ping = read_source_jdbc(
    adapter,
    "(SELECT 1 AS CONNECTION_OK) q",
    source_server=test_server,
    source_database=test_database,
    fetchsize=1,
)
ping_rows = ping.collect()
if not ping_rows or int(ping_rows[0]["CONNECTION_OK"]) != 1:
    raise RuntimeError(
        "SQL Server SELECT 1 returned an unexpected value."
    )
ping.show(n=1, truncate=False)
print("SQL Server TLS connection and authentication succeeded.")

# COMMAND ----------
# MAGIC %md ### 4. Source table TOP sample test

# COMMAND ----------
probe = adapter.top_n_probe_query(
    test_database,
    test_schema,
    test_table,
    5,
)
print(
    "Testing SQL Server object:",
    f"{test_database}.{test_schema}.{test_table}",
)
sample_df = read_source_jdbc(
    adapter,
    probe,
    source_server=test_server,
    source_database=test_database,
    fetchsize=5,
)
sample_df.show(n=5, truncate=False)
sample_count = sample_df.count()
print("Sample rows returned:", sample_count)

# COMMAND ----------
# MAGIC %md ### 5. Column metadata test

# COMMAND ----------
meta = read_source_jdbc(
    adapter,
    adapter.columns_metadata_query(
        test_database,
        test_schema,
        test_table,
    ),
    source_server=test_server,
    source_database=test_database,
)
meta.show(truncate=False)
meta_count = meta.count()
print("Column metadata rows:", meta_count)
if meta_count == 0:
    raise RuntimeError(
        "No SQL Server column metadata was returned. Verify the "
        "database/schema/table names and object visibility."
    )

# COMMAND ----------
# MAGIC %md ### 6. Primary-key metadata test

# COMMAND ----------
pk = read_source_jdbc(
    adapter,
    adapter.primary_key_query(
        test_database,
        test_schema,
        test_table,
    ),
    source_server=test_server,
    source_database=test_database,
)
pk.show(truncate=False)
pk_count = pk.count()
print("Primary-key columns:", pk_count)
if pk_count == 0:
    print(
        "No primary key found. The accelerator may select WATERMARK "
        "or FULL_LOAD depending on available temporal columns."
    )

# COMMAND ----------
# MAGIC %md ### 7. Final success result

# COMMAND ----------
result = {
    "status": "SUCCEEDED",
    "database": test_database,
    "schema": test_schema,
    "table": test_table,
    "sample_rows": sample_count,
    "metadata_columns": meta_count,
    "primary_key_columns": pk_count,
}
print(json.dumps(result, indent=2))
dbutils.notebook.exit(json.dumps(result))
