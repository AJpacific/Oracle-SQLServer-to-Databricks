"""
Focused SQL Server and mixed-source tests for the accelerator add-on.

These complement tests/test_accelerator.py (which stays the Oracle suite). They
cover the source-adapter boundary, source identity, SQL Server SQL generation,
SQL Server type mapping, SQL Server temporal watermark policy, SQL Server
partition planning, JDBC connection safety, and the mixed-source notebook wiring.

Run:  cd oracle_to_databricks && python -m pytest tests -v
"""

import os
import sys

HERE = os.path.dirname(__file__)
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CONFIG_SS = os.path.join(ROOT, "config", "type_rules_sqlserver.yaml")
CONFIG_ORA = os.path.join(ROOT, "config", "type_rules_oracle.yaml")
# Import everything through the ``src`` package so these tests share the exact
# module objects the source adapters/factory load (the adapters use
# ``from src.<mod> import ...``). Mixing flat and package imports would create
# duplicate module objects and break isinstance / exception-identity checks.
sys.path.insert(0, ROOT)

import pytest

import src.identifiers as ids
import src.sqlserver_sql_builder as ssb
import src.strategy as strat
import src.partitioning as part
from src.crosssourcetypemapper import CrossSourceTypeMapper
from src.source_identity import (
    compute_source_table_id, normalize_source_system,
)
from src.source_adapters.factory import get_source_adapter
from src.source_adapters.oracle import OracleSourceAdapter
from src.source_adapters.sqlserver import SqlServerSourceAdapter


NB = os.path.join(ROOT, "notebooks")


def _read_nb(name):
    with open(os.path.join(NB, name), "r", encoding="utf-8") as fh:
        return fh.read()


# ============================================================ adapter factory
class TestAdapterFactory:
    def test_oracle_returns_oracle_adapter(self):
        assert isinstance(get_source_adapter("oracle"), OracleSourceAdapter)

    def test_sqlserver_returns_sqlserver_adapter(self):
        assert isinstance(get_source_adapter("sqlserver"), SqlServerSourceAdapter)

    @pytest.mark.parametrize("alias", ["sql_server", "mssql", "SQL Server", "MSSQL"])
    def test_synonyms_normalize_to_sqlserver(self, alias):
        assert normalize_source_system(alias) == "sqlserver"
        assert isinstance(get_source_adapter(alias), SqlServerSourceAdapter)

    def test_oracle_case_insensitive(self):
        assert normalize_source_system("ORACLE") == "oracle"

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError):
            get_source_adapter("db2")
        with pytest.raises(ValueError):
            normalize_source_system("postgres")

    def test_null_source_raises(self):
        with pytest.raises(ValueError):
            normalize_source_system(None)
        with pytest.raises(ValueError):
            normalize_source_system("")

    def test_unknown_never_defaults_to_oracle(self):
        with pytest.raises(ValueError):
            get_source_adapter("mysql")

    def test_adapter_source_system_attr(self):
        assert get_source_adapter("oracle").source_system == "oracle"
        assert get_source_adapter("mssql").source_system == "sqlserver"


# ============================================================ source identity
class TestSourceIdentity:
    def test_oracle_vs_sqlserver_same_schema_table_differ(self):
        a = compute_source_table_id("oracle", "svr", "db", "dbo", "Customers")
        b = compute_source_table_id("sqlserver", "svr", "db", "dbo", "Customers")
        assert a != b

    def test_two_databases_differ(self):
        a = compute_source_table_id("sqlserver", "svr", "AdventureWorks", "dbo", "Customers")
        b = compute_source_table_id("sqlserver", "svr", "Northwind", "dbo", "Customers")
        assert a != b

    def test_two_servers_differ(self):
        a = compute_source_table_id("sqlserver", "server-a", "db", "dbo", "Orders")
        b = compute_source_table_id("sqlserver", "server-b", "db", "dbo", "Orders")
        assert a != b

    def test_deterministic(self):
        a = compute_source_table_id("sqlserver", "SVR", "DB", "dbo", "Customers")
        b = compute_source_table_id("sqlserver", "svr", "db", "dbo", "Customers")
        # system/server/database are case-insensitive; schema/table preserved.
        assert a == b

    def test_schema_table_case_preserved(self):
        a = compute_source_table_id("sqlserver", "svr", "db", "dbo", "Customers")
        b = compute_source_table_id("sqlserver", "svr", "db", "DBO", "customers")
        assert a != b

    def test_legacy_oracle_null_server_db(self):
        a = compute_source_table_id("oracle", None, None, "HR", "EMPLOYEES")
        b = compute_source_table_id("oracle", "", "", "HR", "EMPLOYEES")
        assert a == b  # null and empty normalize identically

    def test_is_sha256_hex(self):
        v = compute_source_table_id("oracle", None, None, "HR", "EMPLOYEES")
        assert len(v) == 64 and all(c in "0123456789abcdef" for c in v)

    def test_missing_schema_or_table_raises(self):
        with pytest.raises(ValueError):
            compute_source_table_id("oracle", None, None, None, "T")
        with pytest.raises(ValueError):
            compute_source_table_id("oracle", None, None, "S", None)


# ============================================================ SQL Server identifiers
class TestSqlServerIdentifiers:
    def test_bracket_quoting(self):
        assert ids.quote_sqlserver("Employees") == "[Employees]"

    def test_closing_bracket_escaped(self):
        assert ids.quote_sqlserver("we]ird") == "[we]]ird]"

    def test_fqn_two_part(self):
        assert ids.sqlserver_fqn("dbo", "Employees") == "[dbo].[Employees]"

    def test_fqn_three_part(self):
        assert (ids.sqlserver_fqn("dbo", "Employees", "AdventureWorks")
                == "[AdventureWorks].[dbo].[Employees]")

    def test_rejects_semicolon(self):
        with pytest.raises(ids.IdentifierError):
            ids.quote_sqlserver("x; DROP TABLE t")

    def test_does_not_uppercase(self):
        assert ids.quote_sqlserver("mixedCase") == "[mixedCase]"

    def test_validate_server_ok(self):
        assert ids.validate_server("sql-server-01") == "sql-server-01"
        assert ids.validate_server("host,1433") == "host,1433"
        assert ids.validate_server("host\\INSTANCE") == "host\\INSTANCE"

    def test_validate_server_rejects_injection(self):
        with pytest.raises(ids.IdentifierError):
            ids.validate_server("host;password=leak")

    def test_validate_database_ok(self):
        assert ids.validate_database("AdventureWorks") == "AdventureWorks"

    def test_validate_database_rejects_injection(self):
        with pytest.raises(ids.IdentifierError):
            ids.validate_database("db;encrypt=false")


# ============================================================ SQL Server SQL
class TestSqlServerSql:
    def test_top_n_probe(self):
        q = ssb.build_top_n_probe("AdventureWorks", "dbo", "Employees", 5)
        assert "SELECT TOP (5)" in q
        assert "[AdventureWorks].[dbo].[Employees]" in q
        assert "FETCH FIRST" not in q

    def test_top_n_floor(self):
        q = ssb.build_top_n_probe("db", "dbo", "T", 0)
        assert "TOP (1)" in q

    def test_columns_metadata_uses_catalog_views(self):
        q = ssb.columns_metadata_query("AdventureWorks", "dbo", "Employees")
        for v in ["sys.columns", "sys.tables", "sys.schemas", "sys.types"]:
            assert v in q
        for c in ["COLUMN_NAME", "ORDINAL_POSITION", "IS_NULLABLE", "DATA_TYPE",
                  "CHARACTER_MAXIMUM_LENGTH", "NUMERIC_PRECISION", "NUMERIC_SCALE",
                  "DATETIME_PRECISION", "IS_IDENTITY", "IS_COMPUTED",
                  "IS_ROWVERSION", "SOURCE_TYPE_SCHEMA"]:
            assert c in q
        assert "ALL_TAB_COLUMNS" not in q.upper()

    def test_columns_metadata_char_length_in_chars(self):
        q = ssb.columns_metadata_query("db", "dbo", "T")
        # n[var]char byte length halved; MAX (-1) preserved
        assert "c.max_length / 2" in q
        assert "= -1 THEN -1" in q

    def test_pk_query_uses_indexes(self):
        q = ssb.primary_key_query("db", "dbo", "Orders")
        assert "sys.indexes" in q and "sys.index_columns" in q
        assert "is_primary_key = 1" in q
        assert "is_included_column = 0" in q
        assert "KEY_POSITION" in q
        assert "ORDER BY ic.key_ordinal" in q

    def test_count_query(self):
        q = ssb.build_count_query("db", "dbo", "T")
        assert "COUNT_BIG(*)" in q and "ROW_COUNT" in q

    def test_min_max_query(self):
        q = ssb.build_min_max_query("db", "dbo", "T", "Id")
        assert "MIN([Id])" in q and "MAX([Id])" in q
        assert "MIN_VAL" in q and "MAX_VAL" in q

    def test_upper_watermark_query(self):
        q = ssb.build_upper_watermark_query("db", "dbo", "T", "ModifiedDate", "datetime2")
        assert "MAX(CAST([ModifiedDate] AS datetime2(6)))" in q and "UPPER_WATERMARK" in q

    def test_full_extract(self):
        q = ssb.build_full_extract_query("db", "dbo", "T")
        assert q == "(SELECT * FROM [db].[dbo].[T]) q"

    def test_full_extract_columns(self):
        q = ssb.build_full_extract_query("db", "dbo", "T", ["A", "B"])
        assert "[A], [B]" in q

    def test_incremental_interval(self):
        q = ssb.build_incremental_extract_query(
            "db", "dbo", "T", "ModifiedDate", "datetime2",
            "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z",
            columns=["Id", "ModifiedDate"])
        assert "CAST([ModifiedDate] AS datetime2(6)) >" in q and "CAST([ModifiedDate] AS datetime2(6)) <=" in q
        assert "CAST(" in q and "datetime2" in q

    def test_incremental_requires_both_bounds(self):
        with pytest.raises(ValueError):
            ssb.build_incremental_extract_query(
                "db", "dbo", "T", "M", "datetime2", None, "2026-01-02T00:00:00Z",
                columns=["M"])

    def test_no_oracle_syntax_anywhere(self):
        qs = [
            ssb.build_top_n_probe("db", "dbo", "T", 5),
            ssb.columns_metadata_query("db", "dbo", "T"),
            ssb.primary_key_query("db", "dbo", "T"),
            ssb.build_count_query("db", "dbo", "T"),
            ssb.build_min_max_query("db", "dbo", "T", "Id"),
            ssb.build_full_extract_query("db", "dbo", "T"),
            ssb.build_incremental_extract_query(
                "db", "dbo", "T", "M", "datetime2",
                "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z",
                columns=["M"]),
        ]
        for q in qs:
            assert "FETCH FIRST" not in q
            assert "TO_TIMESTAMP" not in q
            assert "ALL_TAB_COLUMNS" not in q.upper()
            assert '"' not in q  # no Oracle double-quote identifiers

    def test_incremental_value_quote_escaped(self):
        # single quotes in a temporal bound would be canonicalized/parsed; a
        # non-parseable injection attempt must raise, never interpolate raw.
        with pytest.raises(Exception):
            ssb.build_incremental_extract_query(
                "db", "dbo", "T", "M", "datetime2",
                "2026-01-01'; DROP--", "2026-01-02'; DROP--",
                columns=["M"])


class TestSqlServerWatermarkLiterals:
    def test_date_literal(self):
        lit = ssb._format_watermark_literal("2026-08-18T00:00:00Z", "date")
        assert lit == "CAST('2026-08-18' AS date)"

    def test_datetime_literal_millis(self):
        lit = ssb._format_watermark_literal("2026-08-18T07:00:45.123456Z", "datetime")
        assert lit.startswith("CAST('2026-08-18 07:00:45.123'") and "AS datetime)" in lit

    def test_smalldatetime_literal(self):
        lit = ssb._format_watermark_literal("2026-08-18T07:00:45Z", "smalldatetime")
        assert "AS smalldatetime)" in lit

    def test_datetime2_literal_micros(self):
        lit = ssb._format_watermark_literal("2026-08-18T07:00:45.123456Z", "datetime2")
        assert "07:00:45.123456" in lit and "AS datetime2(6))" in lit

    def test_datetimeoffset_preserves_offset(self):
        lit = ssb._format_watermark_literal("2026-08-18T07:00:45.123456Z", "datetimeoffset")
        assert "+00:00" in lit and "AS datetimeoffset)" in lit

    @pytest.mark.parametrize("family", [
        "timestamp", "rowversion", "time", "int", "varchar", "", None])
    def test_non_temporal_raises(self, family):
        with pytest.raises(ValueError):
            ssb._format_watermark_literal("2026-08-18T07:00:45Z", family)


class TestSqlServerSixDigitConsistency:
    def test_full_extract_projects_datetime2_watermark(self):
        q = ssb.build_full_extract_query(
            "db", "dbo", "T", ["Id", "ModifiedDate"],
            watermark_column="ModifiedDate", watermark_type="datetime2")
        assert "CAST([ModifiedDate] AS datetime2(6)) AS [ModifiedDate]" in q

    def test_upper_date_preserves_native_expression(self):
        q = ssb.build_upper_watermark_query("db", "dbo", "T", "BusinessDate", "date")
        assert "MAX([BusinessDate])" in q and "datetime2(6)" not in q

    def test_empty_checkpoint_policy(self):
        assert get_source_adapter("sqlserver").initial_watermark_value("datetime2") ==             "1900-01-01T00:00:00.000000Z"

# ============================================================ SQL Server JDBC
def _fake_secrets(mapping):
    def provider(scope, key):
        return mapping.get(key)
    return provider


class TestSqlServerJdbc:
    def _adapter(self, secrets, server="sql-01", database="AdventureWorks",
                 config=None):
        return get_source_adapter(
            "sqlserver", source_server=server, source_database=database,
            secret_provider=_fake_secrets(secrets), secret_scope="sqlserver-migration",
            config=config)

    def test_microsoft_driver(self):
        a = self._adapter({"sqlserver-user": "u", "sqlserver-password": "p"})
        url, props = a.get_jdbc_url_and_props("sql-01", "AdventureWorks")
        assert props["driver"] == "com.microsoft.sqlserver.jdbc.SQLServerDriver"

    def test_url_built_safely(self):
        a = self._adapter({"sqlserver-user": "u", "sqlserver-password": "p"})
        url, _ = a.get_jdbc_url_and_props("sql-01", "AdventureWorks")
        assert url.startswith("jdbc:sqlserver://sql-01:1433;")
        assert "databaseName=AdventureWorks;" in url
        assert "encrypt=true;" in url
        assert "trustServerCertificate=false" in url

    def test_database_comes_from_row_not_secret(self):
        a = self._adapter({"sqlserver-user": "u", "sqlserver-password": "p"},
                          database="Northwind")
        url, _ = a.get_jdbc_url_and_props(None, "Northwind")
        assert "databaseName=Northwind;" in url

    def test_missing_database_raises(self):
        a = get_source_adapter(
            "sqlserver", source_server="s", source_database=None,
            secret_provider=_fake_secrets({"sqlserver-user": "u",
                                           "sqlserver-password": "p"}),
            secret_scope="sqlserver-migration")
        with pytest.raises(ValueError):
            a.get_jdbc_url_and_props(None, None)

    def test_url_template_fills_database(self):
        secrets = {"sqlserver-user": "u", "sqlserver-password": "p",
                   "sqlserver-jdbc-url": "jdbc:sqlserver://h:1433;encrypt=true"}
        a = self._adapter(secrets)
        url, _ = a.get_jdbc_url_and_props("sql-01", "AdventureWorks")
        assert "databaseName=AdventureWorks" in url

    def test_url_template_placeholder(self):
        secrets = {"sqlserver-user": "u", "sqlserver-password": "p",
                   "sqlserver-jdbc-url": "jdbc:sqlserver://h:1433;databaseName={database};encrypt=true"}
        a = self._adapter(secrets)
        url, _ = a.get_jdbc_url_and_props("sql-01", "Northwind")
        assert "databaseName=Northwind" in url and "{database}" not in url

    def test_trust_server_certificate_opt_in(self):
        a = self._adapter({"sqlserver-user": "u", "sqlserver-password": "p"},
                          config={"trust_server_certificate": True})
        url, _ = a.get_jdbc_url_and_props("sql-01", "AdventureWorks")
        assert "trustServerCertificate=true" in url

    def test_no_oracle_read_options(self):
        a = self._adapter({"sqlserver-user": "u", "sqlserver-password": "p"})
        assert a.extra_read_options() == {}

    def test_oracle_has_map_date_option(self):
        assert OracleSourceAdapter().extra_read_options().get(
            "oracle.jdbc.mapDateToTimestamp") == "true"

    def test_credentials_not_in_redacted_url(self):
        url = ("jdbc:sqlserver://h:1433;databaseName=DB;user=admin;"
               "password=SuperSecret;encrypt=true")
        red = SqlServerSourceAdapter.redact_jdbc_url(url)
        assert "SuperSecret" not in red and "admin" not in red
        assert "***" in red

    def test_invalid_server_rejected(self):
        a = self._adapter({"sqlserver-user": "u", "sqlserver-password": "p"},
                          server="host;inject=1")
        with pytest.raises(ids.IdentifierError):
            a.get_jdbc_url_and_props("host;inject=1", "AdventureWorks")


# ============================================================ SQL Server mappings
@pytest.fixture(scope="module")
def ss_mapper():
    assert os.path.exists(CONFIG_SS)
    return CrossSourceTypeMapper.from_yaml_path(CONFIG_SS)


class TestSqlServerMappings:
    @pytest.mark.parametrize("sql_type,expected", [
        ("bit", "BOOLEAN"),
        ("tinyint", "SMALLINT"),
        ("smallint", "SMALLINT"),
        ("int", "INT"),
        ("bigint", "BIGINT"),
        ("money", "DECIMAL(19,4)"),
        ("smallmoney", "DECIMAL(10,4)"),
        ("real", "FLOAT"),
        ("float", "DOUBLE"),
        ("uniqueidentifier", "STRING"),
        ("date", "DATE"),
        ("datetime2", "TIMESTAMP"),
        ("timestamp", "BINARY"),
        ("rowversion", "BINARY"),
    ])
    def test_direct(self, ss_mapper, sql_type, expected):
        assert ss_mapper.map_column(sql_type).databricks_delta_type == expected

    def test_decimal_precision_scale(self, ss_mapper):
        r = ss_mapper.map_column("decimal", precision=10, scale=2)
        assert r.databricks_delta_type == "DECIMAL(10,2)" and r.status == "AUTO"

    def test_numeric_precision_scale(self, ss_mapper):
        r = ss_mapper.map_column("numeric", precision=18, scale=4)
        assert r.databricks_delta_type == "DECIMAL(18,4)"

    def test_decimal_precision_over_38_blocked(self, ss_mapper):
        assert ss_mapper.map_column("decimal", precision=40, scale=2).status == "BLOCKED"

    def test_decimal_default_when_missing(self, ss_mapper):
        r = ss_mapper.map_column("decimal")
        assert r.databricks_delta_type == "DECIMAL(18,0)"

    def test_varchar_max_string(self, ss_mapper):
        # length -1 (MAX) still maps to STRING
        assert ss_mapper.map_column("varchar", length=-1).databricks_delta_type == "STRING"
        assert ss_mapper.map_column("nvarchar", length=-1).databricks_delta_type == "STRING"

    def test_binary_varbinary(self, ss_mapper):
        assert ss_mapper.map_column("binary").databricks_delta_type == "BINARY"
        assert ss_mapper.map_column("varbinary").databricks_delta_type == "BINARY"

    def test_datetimeoffset_review(self, ss_mapper):
        r = ss_mapper.map_column("datetimeoffset")
        assert r.databricks_delta_type == "TIMESTAMP" and r.status == "REVIEW"

    def test_text_ntext_review(self, ss_mapper):
        assert ss_mapper.map_column("text").status == "REVIEW"
        assert ss_mapper.map_column("ntext").status == "REVIEW"

    def test_image_review(self, ss_mapper):
        assert ss_mapper.map_column("image").status == "REVIEW"

    def test_time_review_string(self, ss_mapper):
        r = ss_mapper.map_column("time")
        assert r.databricks_delta_type == "STRING" and r.status == "REVIEW"

    @pytest.mark.parametrize("t", [
        "sql_variant", "hierarchyid", "geometry", "geography", "cursor", "table"])
    def test_special_types_blocked(self, ss_mapper, t):
        assert ss_mapper.map_column(t).status == "BLOCKED"

    def test_unknown_type_blocked_not_string(self, ss_mapper):
        r = ss_mapper.map_column("my_clr_udt")
        assert r.status == "BLOCKED" and r.databricks_delta_type is None

    def test_case_insensitive(self, ss_mapper):
        assert ss_mapper.map_column("INT").databricks_delta_type == "INT"
        assert ss_mapper.map_column("DateTime2").databricks_delta_type == "TIMESTAMP"

    def test_oracle_number_path_not_used(self, ss_mapper):
        # SQL Server has no bare NUMBER; passing 'numeric' must not go through the
        # Oracle NUMBER logic (which would AUTO an unconstrained value).
        r = ss_mapper.map_column("numeric", precision=5, scale=0)
        assert r.databricks_delta_type == "DECIMAL(5,0)"


class TestMapperDialectIsolation:
    def test_oracle_mapper_still_number_aware(self):
        m = CrossSourceTypeMapper.from_yaml_path(CONFIG_ORA)
        assert m.map_column("NUMBER", precision=4, scale=0).databricks_delta_type == "SMALLINT"
        assert m.map_column("NUMBER", precision=None, scale=None).status == "AUTO"

    def test_sqlserver_int_not_in_oracle_mapper(self):
        m = CrossSourceTypeMapper.from_yaml_path(CONFIG_ORA)
        # 'int' is not an Oracle type -> BLOCKED under the Oracle dialect.
        assert m.map_column("int").status == "BLOCKED"

    def test_oracle_number_not_in_sqlserver_mapper(self):
        m = CrossSourceTypeMapper.from_yaml_path(CONFIG_SS)
        # bare Oracle NUMBER is not a SQL Server type -> BLOCKED.
        assert m.map_column("NUMBER").status == "BLOCKED"


# ============================================================ SQL Server strategy
class TestSqlServerStrategy:
    def _adapter(self):
        return get_source_adapter("sqlserver")

    def _oracle(self):
        return get_source_adapter("oracle")

    def test_oracle_temporal_unchanged(self):
        a = self._oracle()
        assert a.is_supported_watermark_type("TIMESTAMP(6)") is True
        assert a.is_supported_watermark_type("NUMBER") is False

    @pytest.mark.parametrize("t", [
        "datetime2", "datetimeoffset", "datetime", "date", "smalldatetime"])
    def test_sqlserver_temporal_eligible(self, t):
        assert self._adapter().is_supported_watermark_type(t) is True

    @pytest.mark.parametrize("t", ["time", "timestamp", "rowversion", "int", "varchar"])
    def test_sqlserver_non_temporal(self, t):
        assert self._adapter().is_supported_watermark_type(t) is False

    def test_pk_plus_datetime2_hybrid(self):
        a = self._adapter()
        cols = [{"column_name": "Id", "data_type": "int", "ordinal_position": 1},
                {"column_name": "ModifiedDate", "data_type": "datetime2", "ordinal_position": 2}]
        d = strat.resolve_watermark_decision(cols, ["Id"], source=a)
        assert d["strategy"] == "HYBRID" and d["watermark_column"] == "ModifiedDate"
        assert d["watermark_data_type"] == "DATETIME2"

    def test_no_pk_plus_datetime2_watermark(self):
        a = self._adapter()
        cols = [{"column_name": "ModifiedDate", "data_type": "datetime2", "ordinal_position": 1}]
        d = strat.resolve_watermark_decision(cols, [], source=a)
        assert d["strategy"] == "WATERMARK"

    def test_pk_no_temporal_primary_key(self):
        a = self._adapter()
        cols = [{"column_name": "Id", "data_type": "int", "ordinal_position": 1},
                {"column_name": "Ver", "data_type": "rowversion", "ordinal_position": 2}]
        d = strat.resolve_watermark_decision(cols, ["Id"], source=a)
        assert d["strategy"] == "PRIMARY_KEY" and d["watermark_column"] is None

    def test_no_pk_no_temporal_full_load(self):
        a = self._adapter()
        cols = [{"column_name": "Name", "data_type": "varchar", "ordinal_position": 1}]
        d = strat.resolve_watermark_decision(cols, [], source=a)
        assert d["strategy"] == "FULL_LOAD"

    def test_rowversion_named_updated_not_temporal(self):
        a = self._adapter()
        cols = [{"column_name": "UpdatedAt", "data_type": "rowversion", "ordinal_position": 1}]
        assert strat.pick_watermark_column(cols, policy=a) is None

    def test_datetimeoffset_ranks_above_datetime2(self):
        a = self._adapter()
        cols = [{"column_name": "a_dt2", "data_type": "datetime2", "ordinal_position": 1},
                {"column_name": "b_dto", "data_type": "datetimeoffset", "ordinal_position": 2}]
        assert strat.pick_watermark_column(cols, policy=a)["column_name"] == "b_dto"

    def test_datetime2_ranks_above_date(self):
        a = self._adapter()
        cols = [{"column_name": "a_date", "data_type": "date", "ordinal_position": 1},
                {"column_name": "b_dt2", "data_type": "datetime2", "ordinal_position": 2}]
        assert strat.pick_watermark_column(cols, policy=a)["column_name"] == "b_dt2"


# ============================================================ SQL Server partitioning
class TestSqlServerPartitioning:
    def _plan(self, data_type, target, lo, hi, n=8):
        a = get_source_adapter("sqlserver")
        return a.resolve_partition_plan(
            {"data_type": data_type}, target, lo, hi, n)

    def test_int_pk_eligible(self):
        eff, lo, hi, reason = self._plan("int", "INT", 1, 1000)
        assert reason is None and eff > 1 and lo == 1 and hi == 1000

    @pytest.mark.parametrize("t", ["tinyint", "smallint", "int", "bigint"])
    def test_native_integral_eligible(self, t):
        eff, lo, hi, reason = self._plan(t, "BIGINT", 1, 500)
        assert reason is None

    def test_decimal_pk_unpartitioned(self):
        eff, lo, hi, reason = self._plan("decimal", "DECIMAL(10,2)", 1, 1000)
        assert eff is None and reason is not None

    def test_non_integral_target_rejected(self):
        eff, lo, hi, reason = self._plan("int", "STRING", 1, 1000)
        assert eff is None

    def test_min_ge_max_rejected(self):
        eff, lo, hi, reason = self._plan("int", "INT", 5, 5)
        assert eff is None

    def test_oracle_partitioning_unchanged(self):
        a = get_source_adapter("oracle")
        eff, lo, hi, reason = a.resolve_partition_plan(
            {"data_type": "NUMBER", "numeric_precision": 9, "numeric_scale": 0},
            "INT", 1, 1000, 8)
        assert reason is None and eff > 1
        # unconstrained NUMBER stays unpartitioned
        eff2, _, _, r2 = a.resolve_partition_plan(
            {"data_type": "NUMBER", "numeric_precision": None, "numeric_scale": None},
            "DECIMAL(38,0)", 1, 1000, 8)
        assert eff2 is None


# ============================================================ mixed-source wiring
class TestMixedSourceNotebookWiring:
    def test_nb01_uses_adapter_per_row(self):
        src = _read_nb("NB01_SourceInventory.py")
        assert "get_source_adapter_for_row(r)" in src
        assert "adapter.columns_metadata_query" in src
        assert "adapter.resolve_watermark_decision" in src
        assert "read_jdbc(sqlb." not in src  # no Oracle-only global read

    def test_nb03_selects_mapper_per_source_system(self):
        src = _read_nb("NB03_MappingRulesGeneration.py")
        assert "load_type_mapper(key)" in src or "mapper_for(" in src
        assert "source_system" in src

    def test_nb09_routes_per_adapter(self):
        src = _read_nb("NB09_FullLoad.py")
        assert "get_source_adapter_for_row(r)" in src
        assert "adapter.full_extract_query" in src
        assert "adapter.resolve_partition_plan" in src
        assert "read_jdbc(sqlb." not in src

    def test_nb11a_queues_source_identity(self):
        src = _read_nb("NB11a_DeltaSyncPrep.py")
        for f in ["source_table_id", "source_system", "source_server", "source_database"]:
            assert f in src
        assert "get_source_adapter_for_row(r)" in src

    def test_nb11b_reads_with_queue_identity(self):
        src = _read_nb("NB11b_DeltaSyncApply.py")
        assert "get_source_adapter_for_row(q)" in src
        assert "read_source_jdbc(" in src
        assert 'read_jdbc(q["source_query"])' not in src

    def test_nb12_counts_through_adapter(self):
        src = _read_nb("NB12_ValidationAndReconciliation.py")
        assert "adapter.count_query" in src
        assert "source_table_id" in src

    def test_nb00_queue_self_contained(self):
        src = _read_nb("NB00_ControlTableInit.py")
        # delta_sync_queue carries full source identity for routing.
        qi = src.index("delta_sync_queue")
        seg = src[qi:qi + 600]
        for f in ["source_table_id", "source_system", "source_server", "source_database"]:
            assert f in seg

    def test_nb08_collision_is_config_error(self):
        src = _read_nb("NB08_TargetProvisioning.py")
        assert "collision" in src.lower()
        assert "PROVISION_CONFIG_ERROR" in src

    def test_notebooks_have_no_oracle_only_global_read(self):
        # shared notebooks must not call the Oracle-only read_jdbc(sqlb....) form
        for name in ["NB01_SourceInventory.py", "NB09_FullLoad.py",
                     "NB11a_DeltaSyncPrep.py", "NB11b_DeltaSyncApply.py",
                     "NB12_ValidationAndReconciliation.py"]:
            src = _read_nb(name)
            assert "read_jdbc(sqlb." not in src, name

    def test_control_updates_keyed_by_source_table_id(self):
        for name in ["NB01_SourceInventory.py", "NB07_TableDecisionGeneration.py",
                     "NB08_TargetProvisioning.py", "NB09_FullLoad.py",
                     "NB10_PostFullLoadState.py", "NB11a_DeltaSyncPrep.py",
                     "NB11b_DeltaSyncApply.py"]:
            src = _read_nb(name)
            assert "update_control(src_id" in src, name
