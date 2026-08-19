# Oracle → Databricks Migration Accelerator

A metadata-driven, control-table-orchestrated accelerator that migrates Oracle
schemas to Databricks Delta (Unity Catalog). It mirrors the SQL Server → Databricks
accelerator you already have, but every source-side concern (data dictionary,
dialect, type system) has been re-implemented for **Oracle**.

The pure-logic layer ships with a **59-test suite** (all passing) so the risky
parts — type mapping and load-strategy detection — are provably correct before
you ever touch a cluster.

---

## What changed vs the SQL Server accelerator

| Concern | SQL Server | **Oracle (this build)** |
|---|---|---|
| JDBC driver | `com.microsoft.sqlserver...` | `oracle.jdbc.OracleDriver` |
| JDBC URL | `jdbc:sqlserver://...` | `jdbc:oracle:thin:@//host:port/service` |
| Column metadata | `INFORMATION_SCHEMA.COLUMNS` | `ALL_TAB_COLUMNS` |
| Primary keys | `INFORMATION_SCHEMA` constraints | `ALL_CONSTRAINTS` + `ALL_CONS_COLUMNS` |
| Row-limit probe | `SELECT TOP 5` | `FETCH FIRST 5 ROWS ONLY` |
| Identifier quoting | `[brackets]` | `"double quotes"` (case-sensitive) |
| Numeric types | many distinct types | almost everything is `NUMBER(p,s)` → resolved in code |
| `DATE` | date only | Oracle `DATE` **has a time component** → `TIMESTAMP` |
| Secret scope | `sql-migration` | `oracle-migration` |

The **target** side is still Databricks Delta, so `ddl_builder.py` is essentially
unchanged.

---

## Layout

```
oracle_to_databricks/
├── notebooks/                 # Databricks notebooks (import as a Git folder / Repo)
│   ├── _common.py             # bootstrap: src path, imports, Oracle JDBC, logging
│   ├── 00_TEST_ORACLE_CONNECTION.py
│   ├── NB00_ControlTableInit.py
│   ├── NB01_SourceInventory.py
│   ├── NB02_TypeNormalization.py
│   ├── NB03_MappingRulesGeneration.py
│   ├── NB04_MappingValidation.py
│   ├── NB07_TableDecisionGeneration.py
│   ├── NB08_TargetProvisioning.py
│   ├── NB09_FullLoad.py
│   ├── NB10_PostFullLoadState.py
│   ├── NB11a_DeltaSyncPrep.py
│   ├── NB11b_DeltaSyncApply.py
│   └── NB12_ValidationAndReconciliation.py
├── src/                       # pure logic (no Spark / no dbutils) — unit tested
│   ├── identifiers.py
│   ├── crosssourcetypemapper.py
│   ├── strategy.py
│   ├── sql_builder.py
│   ├── ddl_builder.py
│   └── control_repository.py
├── jobs/                      # Databricks (Lakeflow) Job definitions
│   ├── JOB01_ONBOARD_AND_FULL_LOAD.json
│   └── JOB02_INCREMENTAL_SYNC.json
├── config/
│   └── type_rules.yaml        # Oracle → Delta deterministic type rules
├── tests/
│   └── test_accelerator.py    # 71 tests, all passing
└── requirements-dev.txt
```

---

## Environment

- **Catalog:** `da_accelerators` — an **existing** Unity Catalog catalog that is
  reused, **not** created (no `CREATE CATALOG` privilege required).
- **Control schema:** `control`
- **Full control namespace:** `da_accelerators.control`
- **Secret scope:** `oracle-migration`
- **Compute:** Azure Databricks all-purpose/classic or job compute (**not**
  serverless — the Oracle JDBC JAR and raw JDBC require classic compute).
- **Oracle driver:** `oracle.jdbc.OracleDriver` from Maven
  `com.oracle.database.jdbc:ojdbc17:<approved-version>`.

---

## One-time setup

1. **Install the Oracle JDBC driver** on the classic/job cluster — Maven
   coordinate `com.oracle.database.jdbc:ojdbc17:<approved-version>` (pin to a
   version approved in your org's Maven allow-list, e.g. `23.6.0.24.10`), or
   upload the `ojdbc17.jar`.

2. **Create the secret scope** and add the connection secrets. Preferred: store a
   single complete JDBC URL as `oracle-jdbc-url`; otherwise provide
   host/port/service, which are used only as a fallback:
   ```bash
   databricks secrets create-scope oracle-migration

   # Preferred: one complete JDBC URL (service-name or SID form)
   databricks secrets put-secret oracle-migration oracle-jdbc-url   # jdbc:oracle:thin:@//host:1521/service

   # Fallback (used only when oracle-jdbc-url is absent)
   databricks secrets put-secret oracle-migration oracle-host
   databricks secrets put-secret oracle-migration oracle-port      # e.g. 1521
   databricks secrets put-secret oracle-migration oracle-service   # service name

   # Always required
   databricks secrets put-secret oracle-migration oracle-user
   databricks secrets put-secret oracle-migration oracle-password
   ```
   > For a **SID**, put the SID form (`jdbc:oracle:thin:@host:port:SID`) in the
   > `oracle-jdbc-url` secret. Never commit hosts, usernames, passwords, or
   > connection strings to source control — they live only in the secret scope.

3. **Import** this folder as a Databricks **Repo / Git folder** so `src/` imports
   resolve automatically. (If not in a Repo, set the `src_path` widget to the
   absolute path of `src/`.)

---

## Run order (linear)

Run once to prove connectivity, then the pipeline:

```
00_TEST_ORACLE_CONNECTION      ← prove network/login/SELECT + dictionary access
        │
NB00_ControlTableInit          ← create control schema + control tables (catalog reused)
NB01_SourceInventory           ← read ALL_TAB_COLUMNS + PKs, detect load strategy
NB02_TypeNormalization         ← normalize types, per-table schema hash
NB03_MappingRulesGeneration    ← apply type_rules.yaml → Delta types
NB04_MappingValidation         ← ERROR/WARNING/INFO per column
NB07_TableDecisionGeneration   ← AUTO_MIGRATE / MANUAL_REVIEW / BLOCKED
NB08_TargetProvisioning        ← CREATE target Delta tables from approved mappings
NB09_FullLoad                  ← JDBC read → Delta overwrite + count logging
NB12_ValidationAndReconciliation (mode=full)   ← source vs target parity
NB10_PostFullLoadState         ← commit initial_load_completed + seed watermark
        │  (recurring incremental sync)
NB11a_DeltaSyncPrep            ← build incremental queue (watermark/PK/hybrid)
NB11b_DeltaSyncApply           ← append or MERGE-by-PK, advance watermark on success
NB12_ValidationAndReconciliation (mode=delta)
```

### Ready-made Job definitions (`jobs/`)
Two Lakeflow Job JSONs are included, mirroring the SQL Server accelerator:
- `jobs/JOB01_ONBOARD_AND_FULL_LOAD.json` — NB00 → NB01…NB07 → NB08 → NB09 →
  NB12 (`mode=full`) → NB10, one-shot onboarding + full load.
- `jobs/JOB02_INCREMENTAL_SYNC.json` — NB11a → NB11b → NB12 (`mode=delta`),
  scheduled hourly (starts **PAUSED**).

Import via the Jobs UI ("Create Job" → JSON) or the CLI
(`databricks jobs create --json @jobs/JOB01_ONBOARD_AND_FULL_LOAD.json`). Update
`notebook_path` prefixes, attach compute, and confirm the parameter defaults
(`catalog=da_accelerators`, `secret_scope=oracle-migration`) before running.

### Suggested Databricks Job (Workflows)
- `T00_Init_Control`  → NB00 (sets the shared `run_id` task value; every other
  notebook reuses it via `get_run_id()`).
- `T01..T07` chained one after another (NB01 → NB07).
- `T08_Provision` → NB08.
- `T09_FullLoad` → NB09 as a **ForEach** over AUTO_MIGRATE tables, passing
  `only_source_schema` / `only_source_table` for per-table parallelism.
- `T12_ReconFull` → NB12 (`mode=full`), then `T10_State` → NB10.
- Separate scheduled job for incremental: `T11a` → NB11a, `T11b` → NB11b,
  `T12_ReconDelta` → NB12 (`mode=delta`).

---

## Load strategies (auto-detected in NB01)

| Detected | Condition | NB11b behaviour |
|---|---|---|
| `FULL_LOAD` | no PK, no watermark | always full replace |
| `WATERMARK` | watermark col, no PK | append rows `> last_watermark` |
| `PRIMARY_KEY` | PK, no watermark | re-extract + MERGE by PK (upsert) |
| `HYBRID` | PK **and** watermark | watermark slice + MERGE by PK (safest) |

Watermark candidates: `DATE`, `TIMESTAMP*`, and **integer-like** `NUMBER`
(scale 0 — e.g. sequence/version). A `NUMBER(19,4)` money column is deliberately
**never** chosen as a watermark.

---

## Type mapping highlights (Oracle → Delta)

- `NUMBER(p,0)` → `SMALLINT`/`INT`/`BIGINT` by precision, else `DECIMAL(p,0)`;
  `p>38` is **BLOCKED**.
- `NUMBER(p,s)` → `DECIMAL(p,s)`; unconstrained `NUMBER` → `DECIMAL(38,10)` (**REVIEW**).
- `DATE` → `TIMESTAMP` (**WIDENED** — carries time).
- `TIMESTAMP WITH [LOCAL] TIME ZONE` → `TIMESTAMP` (**REVIEW / LOSSY** — offset dropped).
- `VARCHAR2/NVARCHAR2/CHAR/CLOB/LONG/XMLTYPE/ROWID` → `STRING`.
- `RAW/LONG RAW/BLOB` → `BINARY`; `BFILE/SDO_GEOMETRY/ANYDATA` → **BLOCKED**.

Full catalog: `config/type_rules.yaml`.

---

## Known limitations & production notes

- **Human-review boundary.** Tables classified `MANUAL_REVIEW` / `BLOCKED` by
  NB07 are written to the `review_queue` control table (status
  `PENDING_REVIEW`) and are **never** loaded — only `AUTO_MIGRATE` tables reach
  NB08/NB09. A table auto-resolves out of the queue once its mapping becomes
  clean on a later run.
- **Type casts are lossless by construction.** NB09/NB11b conform the JDBC
  data to the NB08-provisioned Delta schema. Because a table only auto-migrates
  when every column maps `AUTO`/`EXACT`, these casts cannot truncate (the Delta
  type is always wide enough for the Oracle type).
- **Delete propagation (opt-in).** By default deleted source rows are kept
  (`delete_policy = IGNORE_DELETES`). Set a table's `delete_policy` to
  `HARD_DELETE` in `source_table_control` to remove them: for `PRIMARY_KEY`
  tables (which re-extract a full snapshot every sync) NB11b adds
  `WHEN NOT MATCHED BY SOURCE THEN DELETE` to the MERGE, so any target row whose
  PK is absent from the source snapshot is deleted. It is deliberately **not**
  applied to `WATERMARK`/`HYBRID`, which read only a changed slice and cannot
  tell a delete from an unchanged row (give such a table a full `PRIMARY_KEY`
  snapshot pass if you need deletes there).
- **PRIMARY_KEY re-extracts the whole table each sync** (full snapshot + MERGE).
  Correct, but consider Oracle change tracking / a watermark (→ HYBRID) for very
  large tables.
- **Concurrency.** Both Job definitions set `max_concurrent_runs: 1`, so
  scheduled runs never overlap on the same control state.

---

## Testing

```bash
cd oracle_to_databricks
PYTHONPATH=src python -m pytest tests -v
# 71 passed
```

The suite covers identifier quoting, Oracle dictionary SQL (incl. an
SQL-injection escaping check and a "no `TOP`" guard), DDL generation, the full
`NUMBER` decision tree, every non-numeric type, and all four strategy paths.
