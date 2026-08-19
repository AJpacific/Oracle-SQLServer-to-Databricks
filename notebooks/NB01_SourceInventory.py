# Databricks notebook source
# MAGIC %md
# MAGIC # NB01_SourceInventory
# MAGIC Reads Oracle metadata (columns, primary keys) directly over JDBC from the
# MAGIC data dictionary (ALL_TAB_COLUMNS / ALL_CONS_COLUMNS) for every active
# MAGIC table in source_table_control, detects the load strategy, and writes
# MAGIC source_inventory plus strategy/PK/watermark back to control.

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

from pyspark.sql import functions as F

run_id = get_run_id()
print("run_id:", run_id)
repo = control_repo()

def ctrl(t):
    return f"{quote_databricks(CATALOG)}.{quote_databricks(CONTROL_SCHEMA)}.{quote_databricks(t)}"

# COMMAND ----------

active = repo.active_tables().collect()
print(f"Active tables to inventory: {len(active)}")

# COMMAND ----------

inventory_rows = []
for r in active:
    d = r.asDict()
    src_system = d.get("source_system") or "oracle"
    src_server = d.get("source_server")
    src_db = d.get("source_database")
    src_schema = r["source_schema"]
    src_table = r["source_table"]
    src_id = d.get("source_table_id")

    # Validate the source and obtain its adapter first. A bad/unsupported
    # source_system isolates only this row - other rows keep processing.
    try:
        if not src_id:
            src_id = compute_source_table_id(src_system, src_server, src_db,
                                             src_schema, src_table)
            # Persist the id onto the (legacy) row so later updates key on it.
            repo.update_control_by_identity(src_system, src_server, src_db,
                                            src_schema, src_table,
                                            {"source_table_id": src_id})
        adapter = get_source_adapter_for_row(r)
    except Exception as e:
        try:
            if src_id:
                repo.update_control(src_id, {
                    "current_status": "INVENTORY_FAILED",
                    "error_message": f"source routing failed: {str(e)[:900]}",
                })
        except Exception as ctrl_err:
            print(f"  [warn] control update failed for {src_schema}.{src_table}: {ctrl_err}")
        print(f"  FAILED routing {src_system} {src_schema}.{src_table}: {e}")
        continue

    try:
        cols_df = read_source_jdbc(
            adapter,
            adapter.columns_metadata_query(src_db, src_schema, src_table),
            source_server=src_server, source_database=src_db)
        cols = cols_df.collect()
        if not cols:
            repo.update_control(src_id, {
                "current_status": "INVENTORY_FAILED",
                "error_message": "No columns returned from source metadata "
                                 "(check schema/owner casing and SELECT grants)",
            })
            continue

        # ---- primary keys ----
        pk_df = read_source_jdbc(
            adapter,
            adapter.primary_key_query(src_db, src_schema, src_table),
            source_server=src_server, source_database=src_db)
        pk_cols = [pr["COLUMN_NAME"] for pr in pk_df.collect()]

        # ---- build column dicts for strategy detection ----
        col_dicts = []
        for c in cols:
            col_dicts.append({
                "column_name": c["COLUMN_NAME"],
                "data_type": c["DATA_TYPE"],
                "scale": c["NUMERIC_SCALE"],
                "ordinal_position": int(c["ORDINAL_POSITION"]),
                "datetime_precision": (int(c["DATETIME_PRECISION"])
                                       if c["DATETIME_PRECISION"] is not None else None),
            })
            inventory_rows.append((
                run_id, src_id, src_system, src_server, src_db,
                src_schema, src_table, c["COLUMN_NAME"],
                int(c["ORDINAL_POSITION"]), c["IS_NULLABLE"], c["DATA_TYPE"],
                (int(c["CHARACTER_MAXIMUM_LENGTH"]) if c["CHARACTER_MAXIMUM_LENGTH"] is not None else None),
                (int(c["NUMERIC_PRECISION"]) if c["NUMERIC_PRECISION"] is not None else None),
                (int(c["NUMERIC_SCALE"]) if c["NUMERIC_SCALE"] is not None else None),
                (int(c["DATETIME_PRECISION"]) if c["DATETIME_PRECISION"] is not None else None),
            ))

        # Validate any previously stored watermark first, otherwise rank all
        # eligible temporal columns using this source's temporal policy.
        # Invalid/stale values (e.g. numeric IDs, SQL Server rowversion) are
        # rejected and cleared. All ranking logic stays in src/strategy.py.
        configured_wm = d.get("watermark_column")
        decision = adapter.resolve_watermark_decision(col_dicts, pk_cols, configured_wm)
        strategy = decision["strategy"]
        wm_col = decision["watermark_column"]
        wm_family = decision["watermark_data_type"]

        repo.update_control(src_id, {
            "source_table_id": src_id,
            "load_strategy": strategy,
            "primary_key_columns": pk_cols if pk_cols else None,
            "watermark_column": wm_col,
            "watermark_data_type": wm_family,
            "current_status": "INVENTORIED",
            "error_message": None,
        })
        print(f"  [{src_system}] {src_schema}.{src_table}: {len(cols)} cols, "
              f"strategy={strategy}, wm={wm_col} "
              f"(orig_type={decision['source_type']}, norm_type={wm_family}, "
              f"{decision['source'] or 'NONE'}); reason={decision['reason']}")

    except Exception as e:
        try:
            repo.update_control(src_id, {
                "current_status": "INVENTORY_FAILED",
                "error_message": str(e)[:1000],
            })
        except Exception as ctrl_err:
            print(f"  [warn] control update failed for {src_schema}.{src_table}: {ctrl_err}")
        print(f"  FAILED {src_system} {src_schema}.{src_table}: {e}")

# COMMAND ----------

# MAGIC %md ### Persist raw inventory (append this run's rows, keep history by run_id)

# COMMAND ----------

from pyspark.sql.types import *

inventory_schema = StructType([
    StructField("run_id", StringType(), True),
    StructField("source_table_id", StringType(), True),
    StructField("source_system", StringType(), True),
    StructField("source_server", StringType(), True),
    StructField("source_database", StringType(), True),
    StructField("source_schema", StringType(), True),
    StructField("source_table", StringType(), True),
    StructField("column_name", StringType(), True),
    StructField("ordinal_position", IntegerType(), True),
    StructField("is_nullable", StringType(), True),
    StructField("data_type", StringType(), True),
    StructField("character_maximum_length", IntegerType(), True),
    StructField("numeric_precision", IntegerType(), True),
    StructField("numeric_scale", IntegerType(), True),
    StructField("datetime_precision", IntegerType(), True)
])

inv_df = spark.createDataFrame(
    inventory_rows,
    schema=inventory_schema
)

# COMMAND ----------

if inventory_rows:
    inv_df = inv_df.withColumn("captured_ts", F.current_timestamp())

    inv_df.write.format("delta").mode("append").saveAsTable(
        ctrl("source_inventory").replace("`", "")
    )

    print(f"Wrote {len(inventory_rows)} inventory rows.")
else:
    print("No inventory rows produced.")