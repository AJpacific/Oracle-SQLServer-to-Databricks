"""
Full test suite for the Oracle -> Databricks accelerator pure-logic modules.
Run:  cd oracle_to_databricks && PYTHONPATH=src python -m pytest tests -v
"""

import os
import sys

HERE = os.path.dirname(__file__)
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
CONFIG = os.path.abspath(os.path.join(HERE, "..", "config", "type_rules.yaml"))
sys.path.insert(0, SRC)

import pytest

import identifiers as ids
import sql_builder as sqlb
import ddl_builder as ddl
import strategy as strat
import partitioning as part
from crosssourcetypemapper import CrossSourceTypeMapper
from control_repository import new_run_id


# ============================================================ identifiers
class TestIdentifiers:
    def test_quote_databricks_basic(self):
        assert ids.quote_databricks("col") == "`col`"

    def test_quote_databricks_rejects_backtick(self):
        # backtick is not on the allow-list -> rejected loudly (safer)
        with pytest.raises(ids.IdentifierError):
            ids.quote_databricks("we`ird")

    def test_quote_oracle_basic(self):
        assert ids.quote_oracle("EMP") == '"EMP"'

    def test_quote_oracle_rejects_embedded_quote(self):
        # embedded double-quote is not on the allow-list -> rejected loudly
        with pytest.raises(ids.IdentifierError):
            ids.quote_oracle('A"B')

    def test_oracle_fqn(self):
        assert ids.oracle_fqn("HR", "EMPLOYEES") == '"HR"."EMPLOYEES"'

    def test_databricks_fqn(self):
        assert ids.databricks_fqn("cat", "sch", "tbl") == "`cat`.`sch`.`tbl`"

    def test_escape_string_literal_none(self):
        assert ids.escape_string_literal(None) == "NULL"

    def test_escape_string_literal_quote(self):
        assert ids.escape_string_literal("O'Brien") == "'O''Brien'"


# ============================================================ sql_builder
class TestSqlBuilder:
    def test_columns_metadata_uses_all_tab_columns(self):
        q = sqlb.columns_metadata_query("HR", "EMPLOYEES")
        assert "all_tab_columns" in q
        assert "owner = 'HR'" in q
        assert "table_name = 'EMPLOYEES'" in q
        # target-neutral output names must be present
        for c in ["column_name", "ordinal_position", "is_nullable", "data_type",
                  "character_maximum_length", "numeric_precision",
                  "numeric_scale", "datetime_precision"]:
            assert c in q

    def test_pk_query_uses_all_cons_columns(self):
        q = sqlb.primary_key_query("HR", "EMPLOYEES")
        assert "all_constraints" in q and "all_cons_columns" in q
        assert "constraint_type = 'P'" in q

    def test_top_n_probe_uses_fetch_first(self):
        q = sqlb.build_top_n_probe("SALES", "CUSTOMERS", 5)
        assert "FETCH FIRST 5 ROWS ONLY" in q
        assert '"SALES"."CUSTOMERS"' in q
        assert "TOP" not in q  # this is the SQL Server-ism we must not emit

    def test_top_n_probe_floor(self):
        q = sqlb.build_top_n_probe("S", "T", 0)
        assert "FETCH FIRST 1 ROWS ONLY" in q

    def test_count_query(self):
        assert "COUNT(*)" in sqlb.build_count_query("S", "T")

    def test_max_watermark_select(self):
        q = sqlb.build_max_watermark_select("HR", "EMP", "LAST_UPDATED")
        assert 'MAX("LAST_UPDATED")' in q and "max_wm" in q

    def test_upper_watermark_query_quotes_identifiers(self):
        q = sqlb.build_upper_watermark_query("HR", "EMP", "LAST_UPDATED")
        assert q == ('(SELECT MAX("LAST_UPDATED") AS UPPER_WATERMARK '
                     'FROM "HR"."EMP") q')

    def test_upper_watermark_query_rejects_invalid_identifier(self):
        with pytest.raises(ValueError):
            sqlb.build_upper_watermark_query("HR", "EMP", "LAST_UPDATED;DROP")

    def test_min_max_query(self):
        q = sqlb.build_min_max_query("HR", "EMP", "EMP_ID")
        assert 'MIN("EMP_ID")' in q and 'MAX("EMP_ID")' in q
        assert "min_val" in q and "max_val" in q
        assert '"HR"."EMP"' in q

    def test_incremental_timestamp_literal(self):
        q = sqlb.build_incremental_extract_query(
            "HR", "EMP", "LAST_UPDATED", "TIMESTAMP",
            "2026-01-01 00:00:00.000", "2026-01-02 00:00:00.000")
        assert "TO_TIMESTAMP(" in q
        assert '"LAST_UPDATED" >' in q
        assert '"LAST_UPDATED" <=' in q

    def test_incremental_date_literal(self):
        q = sqlb.build_incremental_extract_query(
            "HR", "EMP", "HIRE_DATE", "DATE",
            "2026-01-01 00:00:00", "2026-01-02 00:00:00")
        assert "TO_DATE(" in q

    def test_incremental_numeric_rejected(self):
        # numeric watermark families are no longer supported (temporal-only)
        with pytest.raises(ValueError):
            sqlb.build_incremental_extract_query(
                "HR", "EMP", "VERSION", "NUMBER", 42, 84)

    def test_incremental_requires_both_bounds(self):
        with pytest.raises(ValueError):
            sqlb.build_incremental_extract_query(
                "HR", "EMP", "VERSION", "NUMBER", None, 84)
        with pytest.raises(ValueError):
            sqlb.build_incremental_extract_query(
                "HR", "EMP", "VERSION", "NUMBER", 42, None)

    def test_incremental_timestamp_with_time_zone(self):
        q = sqlb.build_incremental_extract_query(
            "HR", "EMP", "UPDATED_AT", "TIMESTAMP WITH TIME ZONE",
            "2026-01-01 00:00:00.000000 +00:00",
            "2026-01-02 00:00:00.000000 +00:00")
        assert "TO_TIMESTAMP_TZ(" in q
        assert "TZH:TZM" in q

    def test_incremental_numeric_rejects_invalid_value(self):
        with pytest.raises(ValueError):
            sqlb.build_incremental_extract_query(
                "HR", "EMP", "VERSION", "NUMBER", "not-a-number", 84)

    def test_incremental_temporal_value_quote_escaped(self):
        # a single quote embedded in a temporal bound must be doubled (escaped)
        q = sqlb.build_incremental_extract_query(
            "HR", "EMP", "UPDATED_AT", "TIMESTAMP",
            "2026-01-01 00:00:00'; DROP--", "2026-01-02 00:00:00'; DROP--")
        assert "''" in q  # single quote doubled -> neutralised


# ============================================================ ddl_builder
class TestDdlBuilder:
    def test_create_schema(self):
        s = ddl.build_create_schema("cat", "control", "hi")
        assert "CREATE SCHEMA IF NOT EXISTS `cat`.`control`" in s
        assert "COMMENT 'hi'" in s

    def test_create_table_nullability(self):
        cols = [("id", "BIGINT", False), ("name", "STRING", True)]
        s = ddl.build_create_table("c", "s", "t", cols)
        assert "`id` BIGINT NOT NULL" in s
        assert "`name` STRING" in s and "`name` STRING NOT NULL" not in s
        assert "USING DELTA" in s

    def test_merge_condition_single_key(self):
        assert ddl.build_merge_condition(["id"]) == "t.`id` = s.`id`"

    def test_merge_condition_composite_key(self):
        c = ddl.build_merge_condition(["a", "b"])
        assert c == "t.`a` = s.`a` AND t.`b` = s.`b`"

    def test_merge_condition_empty_raises(self):
        with pytest.raises(ValueError):
            ddl.build_merge_condition([])

    def test_merge_sql_shape(self):
        s = ddl.build_merge_sql("cat", "sch", "tgt", "tgt_stage", ["id"])
        assert "MERGE INTO `cat`.`sch`.`tgt`" in s
        assert "USING `cat`.`sch`.`tgt_stage`" in s
        assert "WHEN MATCHED THEN UPDATE SET *" in s
        assert "WHEN NOT MATCHED THEN INSERT *" in s

    def test_merge_sql_no_delete_by_default(self):
        s = ddl.build_merge_sql("cat", "sch", "tgt", "tgt_stage", ["id"])
        assert "BY SOURCE" not in s

    def test_merge_sql_delete_propagation(self):
        s = ddl.build_merge_sql("cat", "sch", "tgt", "tgt_stage", ["id"],
                                delete_unmatched=True)
        assert "WHEN NOT MATCHED BY SOURCE THEN DELETE" in s

    def test_create_like(self):
        s = ddl.build_create_like("c", "s", "t_stage", "t")
        assert "CREATE TABLE IF NOT EXISTS `c`.`s`.`t_stage` LIKE `c`.`s`.`t`" in s

    def test_count_sql(self):
        assert "COUNT(*)" in ddl.build_count_sql("c", "s", "t")


class TestIdentifierValidation:
    def test_valid_plain(self):
        assert ids.validate_identifier("EMPLOYEES") == "EMPLOYEES"

    def test_valid_oracle_dollar_hash(self):
        # Oracle system-ish names use $ and #
        assert ids.validate_identifier("SYS$COL") == "SYS$COL"
        assert ids.validate_identifier("COL#1") == "COL#1"

    def test_rejects_hyphen(self):
        with pytest.raises(ids.IdentifierError):
            ids.validate_identifier("bad-name")

    def test_rejects_semicolon_injection(self):
        with pytest.raises(ids.IdentifierError):
            ids.validate_identifier("x; DROP TABLE t")

    def test_rejects_none_and_empty(self):
        with pytest.raises(ids.IdentifierError):
            ids.validate_identifier(None)
        with pytest.raises(ids.IdentifierError):
            ids.validate_identifier("   ")

    def test_quote_oracle_validates(self):
        with pytest.raises(ids.IdentifierError):
            ids.quote_oracle("a-b")


# ============================================================ type mapper
@pytest.fixture(scope="module")
def mapper():
    assert os.path.exists(CONFIG), f"missing {CONFIG}"
    return CrossSourceTypeMapper.from_yaml_path(CONFIG)


class TestNumberMapping:
    def test_number_scale0_p4_smallint(self, mapper):
        r = mapper.map_column("NUMBER", precision=4, scale=0)
        assert r.databricks_delta_type == "SMALLINT" and r.status == "AUTO"

    def test_number_scale0_p9_int(self, mapper):
        r = mapper.map_column("NUMBER", precision=9, scale=0)
        assert r.databricks_delta_type == "INT"

    def test_number_scale0_p18_bigint(self, mapper):
        r = mapper.map_column("NUMBER", precision=18, scale=0)
        assert r.databricks_delta_type == "BIGINT"

    def test_number_scale0_p28_decimal(self, mapper):
        r = mapper.map_column("NUMBER", precision=28, scale=0)
        assert r.databricks_delta_type == "DECIMAL(28,0)"

    def test_number_scale0_p10_bigint(self, mapper):
        r = mapper.map_column("NUMBER", precision=10, scale=0)
        assert r.databricks_delta_type == "BIGINT"
        assert r.status == "AUTO" and r.fidelity == "EXACT"

    def test_number_scale0_p18_bigint_preserved(self, mapper):
        r = mapper.map_column("NUMBER", precision=18, scale=0)
        assert r.databricks_delta_type == "BIGINT"
        assert r.status == "AUTO" and r.fidelity == "EXACT"

    def test_number_p10_s2_decimal(self, mapper):
        r = mapper.map_column("NUMBER", precision=10, scale=2)
        assert r.databricks_delta_type == "DECIMAL(10,2)"
        assert r.status == "AUTO" and r.fidelity == "EXACT"

    def test_number_money(self, mapper):
        r = mapper.map_column("NUMBER", precision=19, scale=4)
        assert r.databricks_delta_type == "DECIMAL(19,4)" and r.status == "AUTO"

    def test_number_precision_over_38_blocked(self, mapper):
        r = mapper.map_column("NUMBER", precision=40, scale=2)
        assert r.status == "BLOCKED"

    def test_number_unconstrained_whole_number(self, mapper):
        # Both Oracle metadata values missing -> approved whole-number policy.
        r = mapper.map_column("NUMBER", precision=None, scale=None)
        assert r.databricks_delta_type == "DECIMAL(38,0)"
        assert r.status == "AUTO" and r.fidelity == "EXACT"

    def test_number_scale_gt_precision_clamped(self, mapper):
        r = mapper.map_column("NUMBER", precision=5, scale=7)
        assert r.databricks_delta_type == "DECIMAL(5,5)"

    def test_number_negative_scale_review(self, mapper):
        r = mapper.map_column("NUMBER", precision=5, scale=-2)
        assert r.status == "REVIEW" and r.fidelity == "WIDENED"


class TestOtherTypeMapping:
    @pytest.mark.parametrize("otype,expected", [
        ("VARCHAR2", "STRING"),
        ("NVARCHAR2", "STRING"),
        ("CHAR", "STRING"),
        ("CLOB", "STRING"),
        ("DATE", "TIMESTAMP"),
        ("TIMESTAMP(6)", "TIMESTAMP"),
        ("BINARY_FLOAT", "FLOAT"),
        ("BINARY_DOUBLE", "DOUBLE"),
        ("RAW", "BINARY"),
        ("BLOB", "BINARY"),
    ])
    def test_direct_mappings(self, mapper, otype, expected):
        assert mapper.map_column(otype).databricks_delta_type == expected

    def test_date_is_widened(self, mapper):
        assert mapper.map_column("DATE").fidelity == "WIDENED"

    def test_tstz_review_lossy(self, mapper):
        r = mapper.map_column("TIMESTAMP(6) WITH TIME ZONE")
        assert r.status == "REVIEW" and r.fidelity == "LOSSY"

    def test_tsltz_review(self, mapper):
        r = mapper.map_column("TIMESTAMP WITH LOCAL TIME ZONE")
        assert r.status == "REVIEW"

    def test_bfile_blocked(self, mapper):
        assert mapper.map_column("BFILE").status == "BLOCKED"

    def test_sdo_geometry_blocked(self, mapper):
        assert mapper.map_column("SDO_GEOMETRY").status == "BLOCKED"

    def test_long_review(self, mapper):
        assert mapper.map_column("LONG").status == "REVIEW"

    def test_xmltype_review(self, mapper):
        assert mapper.map_column("XMLTYPE").status == "REVIEW"

    def test_unknown_type_blocked(self, mapper):
        assert mapper.map_column("SOME_UDT").status == "BLOCKED"

    def test_nullable_flag_passthrough(self, mapper):
        assert mapper.map_column("VARCHAR2", is_nullable=False).is_nullable is False


class TestOracle26aiMapping:
    def test_boolean_auto_exact(self, mapper):
        r = mapper.map_column("BOOLEAN")
        assert r.databricks_delta_type == "BOOLEAN"
        assert r.status == "AUTO" and r.fidelity == "EXACT"

    def test_boolean_case_insensitive(self, mapper):
        assert mapper.map_column("boolean").databricks_delta_type == "BOOLEAN"

    def test_native_json_review_lossy(self, mapper):
        r = mapper.map_column("JSON")
        assert r.databricks_delta_type == "STRING"
        assert r.status == "REVIEW" and r.fidelity == "LOSSY"

    def test_vector_blocked_no_target(self, mapper):
        r = mapper.map_column("VECTOR")
        assert r.databricks_delta_type is None
        assert r.status == "BLOCKED" and r.fidelity == "UNKNOWN"

    def test_unknown_udt_blocked_none(self, mapper):
        r = mapper.map_column("MY_CUSTOM_OBJECT_TYPE")
        assert r.databricks_delta_type is None
        assert r.status == "BLOCKED" and r.fidelity == "UNKNOWN"
        assert ("explicit" in r.notes.lower()) or ("unsupported" in r.notes.lower())

    def test_clob_widened_auto(self, mapper):
        r = mapper.map_column("CLOB")
        assert r.databricks_delta_type == "STRING"
        assert r.status == "AUTO" and r.fidelity == "WIDENED"

    def test_nclob_widened_auto(self, mapper):
        r = mapper.map_column("NCLOB")
        assert r.databricks_delta_type == "STRING"
        assert r.status == "AUTO" and r.fidelity == "WIDENED"

    def test_blob_widened_auto(self, mapper):
        r = mapper.map_column("BLOB")
        assert r.databricks_delta_type == "BINARY"
        assert r.status == "AUTO" and r.fidelity == "WIDENED"

    def test_number_still_precision_aware(self, mapper):
        assert mapper.map_column("NUMBER", precision=4, scale=0).databricks_delta_type == "SMALLINT"
        assert mapper.map_column("NUMBER", precision=19, scale=4).databricks_delta_type == "DECIMAL(19,4)"
        assert mapper.map_column("NUMBER", precision=40, scale=2).status == "BLOCKED"
        assert mapper.map_column("NUMBER", precision=None, scale=None).status == "AUTO"

    @pytest.mark.parametrize("variant", ["BINARY FLOAT", "binary_float", "BINARY_FLOAT"])
    def test_binary_float_name_variants(self, mapper, variant):
        assert mapper.map_column(variant).databricks_delta_type == "FLOAT"

    @pytest.mark.parametrize("variant", ["SDO_GEOMETRY", "SDO GEOMETRY", "sdo_geometry"])
    def test_sdo_geometry_name_variants_blocked(self, mapper, variant):
        assert mapper.map_column(variant).status == "BLOCKED"

    @pytest.mark.parametrize("variant", [
        "TIMESTAMP WITH TIME ZONE",
        "timestamp_with_time_zone",
        "TIMESTAMP(6) WITH TIME ZONE",
    ])
    def test_tstz_name_variants_review_lossy(self, mapper, variant):
        r = mapper.map_column(variant)
        assert r.status == "REVIEW" and r.fidelity == "LOSSY"


# ============================================================ strategy
class TestStrategy:
    def test_full_load_when_nothing(self):
        cols = [{"column_name": "c", "data_type": "VARCHAR2"}]
        s, wm, fam = strat.detect_strategy(cols, [])
        assert s == strat.FULL_LOAD and wm is None

    def test_primary_key_only(self):
        cols = [{"column_name": "id", "data_type": "VARCHAR2"}]
        s, wm, fam = strat.detect_strategy(cols, ["id"])
        assert s == strat.PRIMARY_KEY and wm is None

    def test_watermark_only(self):
        cols = [{"column_name": "updated_date", "data_type": "TIMESTAMP(6)"}]
        s, wm, fam = strat.detect_strategy(cols, [])
        assert s == strat.WATERMARK and wm == "updated_date" and fam == "TIMESTAMP"

    def test_hybrid(self):
        cols = [
            {"column_name": "id", "data_type": "NUMBER", "scale": 0},
            {"column_name": "updated_ts", "data_type": "TIMESTAMP(6)"},
        ]
        s, wm, fam = strat.detect_strategy(cols, ["id"])
        assert s == strat.HYBRID and wm == "updated_ts"

    def test_number_with_scale_not_watermark(self):
        # a money-like NUMBER(19,4) must NOT be picked as a watermark
        cols = [{"column_name": "amount", "data_type": "NUMBER", "scale": 4}]
        assert strat.pick_watermark_column(cols) is None

    def test_integer_number_is_not_watermark(self):
        # even an integer-like NUMBER (sequence/version) is never a watermark
        cols = [{"column_name": "row_version", "data_type": "NUMBER", "scale": 0}]
        assert strat.pick_watermark_column(cols) is None

    def test_approved_temporal_chosen_numeric_ignored(self):
        cols = [
            {"column_name": "seq", "data_type": "NUMBER", "scale": 0},
            {"column_name": "updated_date", "data_type": "TIMESTAMP(6)"},
        ]
        assert strat.pick_watermark_column(cols)["column_name"] == "updated_date"

    def test_is_valid_strategy(self):
        assert strat.is_valid_strategy("HYBRID")
        assert not strat.is_valid_strategy("NONSENSE")


# ============================================================ watermark type eligibility
class TestWatermarkTypeEligibility:
    @pytest.mark.parametrize("dtype", [
        "NUMBER", "DECIMAL", "INTEGER", "SMALLINT", "BIGINT", "FLOAT",
        "BINARY_FLOAT", "BINARY_DOUBLE",
    ])
    def test_numeric_never_watermark(self, dtype):
        # even an update-looking name never makes a numeric column eligible
        cols = [{"column_name": "updated_date", "data_type": dtype, "scale": 0}]
        assert strat.pick_watermark_column(cols) is None

    def test_varchar2_never_watermark(self):
        cols = [{"column_name": "updated_date", "data_type": "VARCHAR2"}]
        assert strat.pick_watermark_column(cols) is None

    def test_date_eligible_regardless_of_name(self):
        assert strat.pick_watermark_column(
            [{"column_name": "updated_date", "data_type": "DATE"}]) is not None
        assert strat.pick_watermark_column(
            [{"column_name": "some_date", "data_type": "DATE"}]) is not None

    def test_timestamp_eligible_regardless_of_name(self):
        assert strat.pick_watermark_column(
            [{"column_name": "updated_ts", "data_type": "TIMESTAMP"}]) is not None
        assert strat.pick_watermark_column(
            [{"column_name": "event_ts", "data_type": "TIMESTAMP"}]) is not None

    @pytest.mark.parametrize("dtype,fam", [
        ("TIMESTAMP(6)", "TIMESTAMP"),
        ("TIMESTAMP(9)", "TIMESTAMP"),
        ("TIMESTAMP WITH TIME ZONE", "TIMESTAMP WITH TIME ZONE"),
        ("TIMESTAMP(6) WITH TIME ZONE", "TIMESTAMP WITH TIME ZONE"),
        ("TIMESTAMP WITH LOCAL TIME ZONE", "TIMESTAMP WITH LOCAL TIME ZONE"),
    ])
    def test_temporal_normalization(self, dtype, fam):
        cols = [{"column_name": "any_name", "data_type": dtype}]
        chosen = strat.pick_watermark_column(cols)
        assert chosen is not None
        assert strat.normalize_watermark_type(chosen["data_type"]) == fam

    def test_interval_not_eligible(self):
        cols = [{"column_name": "updated_at", "data_type": "INTERVAL DAY TO SECOND"}]
        assert strat.pick_watermark_column(cols) is None


# ============================================================ watermark name semantics
class TestWatermarkNameSemantics:
    @pytest.mark.parametrize("name", [
        "UPDATED_DATE", "LAST_MODIFIED_DATE", "AUDIT_TS", "SOURCE_CHANGE_DTTM",
        "ETL_UPDATED_AT", "MODIFIED_AT", "CHANGE_TIMESTAMP", "REVISION_TS",
    ])
    def test_update_names_high_score(self, name):
        assert strat._semantic_score(name) == 0

    @pytest.mark.parametrize("name", [
        "CREATED_DATE", "INSERTED_AT", "LOAD_TIMESTAMP", "INGESTED_TS",
    ])
    def test_create_names_middle_score(self, name):
        assert strat._semantic_score(name) == 1

    @pytest.mark.parametrize("name", [
        "ORDER_DATE", "EVENT_TIMESTAMP", "SNAPSHOT_DATE", "BUSINESS_DATE",
        "CUSTOM_TEMPORAL_COLUMN",
    ])
    def test_other_temporal_fallback_score(self, name):
        assert strat._semantic_score(name) == 2

    @pytest.mark.parametrize("name", [
        "CREATED_DATE", "INSERTED_AT", "ORDER_DATE", "EVENT_DATE",
        "SNAPSHOT_DATE", "BUSINESS_DATE", "AUDIT_TS", "SOURCE_CHANGE_DTTM",
        "ETL_UPDATE_TIME", "CUSTOM_TEMPORAL_COLUMN",
    ])
    def test_temporal_names_remain_eligible(self, name):
        cols = [{"column_name": name, "data_type": "TIMESTAMP(6)"}]
        assert strat.pick_watermark_column(cols) is not None

    def test_name_never_makes_number_eligible(self):
        cols = [{"column_name": "UPDATED_TIMESTAMP", "data_type": "NUMBER", "scale": 0}]
        assert strat.pick_watermark_column(cols) is None


# ============================================================ watermark ranking
class TestWatermarkRanking:
    def test_tz_ranks_above_plain_timestamp(self):
        cols = [
            {"column_name": "plain_ts", "data_type": "TIMESTAMP(6)", "ordinal_position": 1},
            {"column_name": "tz_ts", "data_type": "TIMESTAMP(6) WITH TIME ZONE", "ordinal_position": 2},
        ]
        assert strat.pick_watermark_column(cols)["column_name"] == "tz_ts"

    def test_local_tz_ranks_above_plain_timestamp(self):
        cols = [
            {"column_name": "plain_ts", "data_type": "TIMESTAMP(6)", "ordinal_position": 1},
            {"column_name": "ltz_ts", "data_type": "TIMESTAMP(6) WITH LOCAL TIME ZONE", "ordinal_position": 2},
        ]
        assert strat.pick_watermark_column(cols)["column_name"] == "ltz_ts"

    def test_timestamp_ranks_above_date(self):
        cols = [
            {"column_name": "a_date", "data_type": "DATE", "ordinal_position": 1},
            {"column_name": "a_ts", "data_type": "TIMESTAMP(6)", "ordinal_position": 2},
        ]
        assert strat.pick_watermark_column(cols)["column_name"] == "a_ts"

    def test_greater_precision_wins_when_equal(self):
        cols = [
            {"column_name": "a_updated_at", "data_type": "TIMESTAMP(3)", "ordinal_position": 1},
            {"column_name": "b_updated_at", "data_type": "TIMESTAMP(9)", "ordinal_position": 2},
        ]
        assert strat.pick_watermark_column(cols)["column_name"] == "b_updated_at"

    def test_update_beats_created_datatype_equal(self):
        cols = [
            {"column_name": "created_ts", "data_type": "TIMESTAMP(6)", "ordinal_position": 1},
            {"column_name": "updated_ts", "data_type": "TIMESTAMP(6)", "ordinal_position": 2},
        ]
        assert strat.pick_watermark_column(cols)["column_name"] == "updated_ts"

    def test_created_beats_generic_datatype_equal(self):
        cols = [
            {"column_name": "order_ts", "data_type": "TIMESTAMP(6)", "ordinal_position": 1},
            {"column_name": "created_ts", "data_type": "TIMESTAMP(6)", "ordinal_position": 2},
        ]
        assert strat.pick_watermark_column(cols)["column_name"] == "created_ts"

    def test_lowest_ordinal_breaks_complete_tie(self):
        cols = [
            {"column_name": "a_updated_at", "data_type": "TIMESTAMP(6)", "ordinal_position": 5},
            {"column_name": "b_updated_at", "data_type": "TIMESTAMP(6)", "ordinal_position": 2},
        ]
        assert strat.pick_watermark_column(cols)["column_name"] == "b_updated_at"

    def test_deterministic_regardless_of_input_order(self):
        cols = [
            {"column_name": "created_date", "data_type": "TIMESTAMP(6)", "ordinal_position": 1},
            {"column_name": "updated_date", "data_type": "DATE", "ordinal_position": 2},
            {"column_name": "audit_ts", "data_type": "TIMESTAMP(9) WITH TIME ZONE", "ordinal_position": 3},
            {"column_name": "order_date", "data_type": "TIMESTAMP(6)", "ordinal_position": 4},
        ]
        a = strat.pick_watermark_column(cols)["column_name"]
        b = strat.pick_watermark_column(list(reversed(cols)))["column_name"]
        assert a == b == "audit_ts"


# ============================================================ strategy decisions
class TestStrategyDecisions:
    def test_pk_plus_updated_date_hybrid(self):
        cols = [
            {"column_name": "id", "data_type": "NUMBER", "scale": 0, "ordinal_position": 1},
            {"column_name": "updated_date", "data_type": "TIMESTAMP(6)", "ordinal_position": 2},
        ]
        s, wm, fam = strat.detect_strategy(cols, ["id"])
        assert s == strat.HYBRID and wm == "updated_date" and fam == "TIMESTAMP"

    def test_pk_plus_created_date_hybrid(self):
        cols = [
            {"column_name": "id", "data_type": "NUMBER", "scale": 0, "ordinal_position": 1},
            {"column_name": "created_date", "data_type": "TIMESTAMP(6)", "ordinal_position": 2},
        ]
        s, wm, fam = strat.detect_strategy(cols, ["id"])
        assert s == strat.HYBRID and wm == "created_date" and fam == "TIMESTAMP"

    def test_pk_plus_order_date_hybrid(self):
        cols = [
            {"column_name": "id", "data_type": "NUMBER", "scale": 0, "ordinal_position": 1},
            {"column_name": "order_date", "data_type": "DATE", "ordinal_position": 2},
        ]
        s, wm, fam = strat.detect_strategy(cols, ["id"])
        assert s == strat.HYBRID and wm == "order_date" and fam == "DATE"

    def test_no_pk_plus_created_date_watermark(self):
        cols = [{"column_name": "created_date", "data_type": "TIMESTAMP(6)"}]
        s, wm, fam = strat.detect_strategy(cols, [])
        assert s == strat.WATERMARK and wm == "created_date"

    def test_no_pk_plus_arbitrary_temporal_watermark(self):
        cols = [{"column_name": "custom_temporal_column", "data_type": "TIMESTAMP(6)"}]
        s, wm, fam = strat.detect_strategy(cols, [])
        assert s == strat.WATERMARK and wm == "custom_temporal_column"

    def test_pk_no_temporal_primary_key(self):
        cols = [{"column_name": "id", "data_type": "NUMBER", "scale": 0}]
        s, wm, fam = strat.detect_strategy(cols, ["id"])
        assert s == strat.PRIMARY_KEY and wm is None and fam is None

    def test_no_pk_no_temporal_full_load(self):
        cols = [{"column_name": "name", "data_type": "VARCHAR2"}]
        s, wm, fam = strat.detect_strategy(cols, [])
        assert s == strat.FULL_LOAD and wm is None and fam is None

    def test_numeric_pk_no_temporal_never_hybrid(self):
        cols = [{"column_name": "customer_id", "data_type": "NUMBER", "scale": 0}]
        s, wm, fam = strat.detect_strategy(cols, ["customer_id"])
        assert s == strat.PRIMARY_KEY and wm is None

    def test_numeric_quantity_no_pk_never_watermark(self):
        cols = [{"column_name": "stock_quantity", "data_type": "NUMBER", "scale": 0}]
        s, wm, fam = strat.detect_strategy(cols, [])
        assert s == strat.FULL_LOAD and wm is None


# ============================================================ configured / stale watermark
class TestConfiguredWatermark:
    def test_configured_audit_ts_accepted(self):
        cols = [{"column_name": "AUDIT_TS", "data_type": "TIMESTAMP(6)"}]
        chosen = strat.validate_configured_watermark(cols, "AUDIT_TS")
        assert chosen is not None and chosen["column_name"] == "AUDIT_TS"

    def test_configured_created_date_accepted(self):
        cols = [{"column_name": "CREATED_DATE", "data_type": "DATE"}]
        chosen = strat.validate_configured_watermark(cols, "CREATED_DATE")
        assert chosen is not None and chosen["column_name"] == "CREATED_DATE"

    def test_configured_custom_temporal_accepted(self):
        cols = [{"column_name": "CUSTOM_TEMPORAL_COLUMN", "data_type": "TIMESTAMP(6)"}]
        chosen = strat.validate_configured_watermark(cols, "custom_temporal_column")
        assert chosen is not None and chosen["column_name"] == "CUSTOM_TEMPORAL_COLUMN"

    def test_configured_customer_id_rejected(self):
        cols = [{"column_name": "customer_id", "data_type": "NUMBER", "scale": 0}]
        assert strat.validate_configured_watermark(cols, "CUSTOMER_ID") is None

    def test_configured_stock_quantity_rejected(self):
        cols = [{"column_name": "stock_quantity", "data_type": "NUMBER", "scale": 0}]
        assert strat.validate_configured_watermark(cols, "STOCK_QUANTITY") is None

    def test_invalid_configured_replaced_by_best_temporal(self):
        cols = [
            {"column_name": "customer_id", "data_type": "NUMBER", "scale": 0, "ordinal_position": 1},
            {"column_name": "created_date", "data_type": "TIMESTAMP(6)", "ordinal_position": 2},
        ]
        s, wm, fam = strat.detect_strategy(
            cols, ["customer_id"], configured_watermark="CUSTOMER_ID")
        assert s == strat.HYBRID and wm == "created_date"

    def test_invalid_configured_cleared_when_no_temporal(self):
        cols = [{"column_name": "customer_id", "data_type": "NUMBER", "scale": 0}]
        s, wm, fam = strat.detect_strategy(
            cols, ["customer_id"], configured_watermark="CUSTOMER_ID")
        assert s == strat.PRIMARY_KEY and wm is None and fam is None

    def test_missing_configured_column_rejected(self):
        cols = [{"column_name": "updated_date", "data_type": "TIMESTAMP(6)"}]
        assert strat.validate_configured_watermark(cols, "NONEXISTENT") is None


# ============================================================ real table examples
class TestRealTableExamples:
    def test_employees_hybrid_updated_date(self):
        cols = [
            {"column_name": "EMPLOYEE_ID", "data_type": "NUMBER", "scale": 0, "ordinal_position": 1},
            {"column_name": "NAME", "data_type": "VARCHAR2", "ordinal_position": 2},
            {"column_name": "UPDATED_DATE", "data_type": "TIMESTAMP(6)", "ordinal_position": 3},
        ]
        s, wm, fam = strat.detect_strategy(cols, ["EMPLOYEE_ID"])
        assert s == strat.HYBRID and wm == "UPDATED_DATE" and fam == "TIMESTAMP"

    def test_inventory_hybrid_last_modified_date(self):
        cols = [
            {"column_name": "ITEM_ID", "data_type": "NUMBER", "scale": 0, "ordinal_position": 1},
            {"column_name": "LAST_MODIFIED_DATE", "data_type": "TIMESTAMP(6)", "ordinal_position": 2},
        ]
        s, wm, fam = strat.detect_strategy(cols, ["ITEM_ID"])
        assert s == strat.HYBRID and wm == "LAST_MODIFIED_DATE" and fam == "TIMESTAMP"

    def test_orders_hybrid_prefers_update_over_order_date(self):
        cols = [
            {"column_name": "ORDER_ID", "data_type": "NUMBER", "scale": 0, "ordinal_position": 1},
            {"column_name": "ORDER_LINE_ID", "data_type": "NUMBER", "scale": 0, "ordinal_position": 2},
            {"column_name": "ORDER_DATE", "data_type": "TIMESTAMP(6)", "ordinal_position": 3},
            {"column_name": "UPDATED_DATE", "data_type": "TIMESTAMP(6)", "ordinal_position": 4},
        ]
        s, wm, fam = strat.detect_strategy(cols, ["ORDER_ID", "ORDER_LINE_ID"])
        assert s == strat.HYBRID and wm == "UPDATED_DATE" and fam == "TIMESTAMP"

    def test_customers_hybrid_created_date_only(self):
        # CUSTOMERS with only CREATED_DATE + PK -> HYBRID on CREATED_DATE.
        # A stale configured numeric CUSTOMER_ID is rejected first.
        cols = [
            {"column_name": "CUSTOMER_ID", "data_type": "NUMBER", "scale": 0, "ordinal_position": 1},
            {"column_name": "NAME", "data_type": "VARCHAR2", "ordinal_position": 2},
            {"column_name": "CREATED_DATE", "data_type": "TIMESTAMP(6)", "ordinal_position": 3},
        ]
        s, wm, fam = strat.detect_strategy(
            cols, ["CUSTOMER_ID"], configured_watermark="CUSTOMER_ID")
        assert s == strat.HYBRID and wm == "CREATED_DATE" and fam == "TIMESTAMP"

    def test_customers_primary_key_when_no_temporal(self):
        cols = [
            {"column_name": "CUSTOMER_ID", "data_type": "NUMBER", "scale": 0, "ordinal_position": 1},
            {"column_name": "NAME", "data_type": "VARCHAR2", "ordinal_position": 2},
        ]
        s, wm, fam = strat.detect_strategy(
            cols, ["CUSTOMER_ID"], configured_watermark="CUSTOMER_ID")
        assert s == strat.PRIMARY_KEY and wm is None and fam is None

    def test_product_snapshot_watermark_on_snapshot_date(self):
        # No PK; SNAPSHOT_DATE is temporal and thus eligible -> WATERMARK.
        # A stale configured numeric STOCK_QUANTITY is rejected first.
        cols = [
            {"column_name": "PRODUCT_ID", "data_type": "NUMBER", "scale": 0, "ordinal_position": 1},
            {"column_name": "STOCK_QUANTITY", "data_type": "NUMBER", "scale": 0, "ordinal_position": 2},
            {"column_name": "SNAPSHOT_DATE", "data_type": "DATE", "ordinal_position": 3},
        ]
        s, wm, fam = strat.detect_strategy(
            cols, [], configured_watermark="STOCK_QUANTITY")
        assert s == strat.WATERMARK and wm == "SNAPSHOT_DATE" and fam == "DATE"

    def test_product_snapshot_full_load_when_no_temporal(self):
        cols = [
            {"column_name": "PRODUCT_ID", "data_type": "NUMBER", "scale": 0, "ordinal_position": 1},
            {"column_name": "STOCK_QUANTITY", "data_type": "NUMBER", "scale": 0, "ordinal_position": 2},
        ]
        s, wm, fam = strat.detect_strategy(
            cols, [], configured_watermark="STOCK_QUANTITY")
        assert s == strat.FULL_LOAD and wm is None and fam is None


# ============================================================ run id
class TestRunId:
    def test_run_id_prefix_and_uniqueness(self):
        a = new_run_id("init")
        b = new_run_id("init")
        assert a.startswith("init_") and a != b


# ============================================================ NB11a temporal watermark guard
NB11A = os.path.abspath(os.path.join(HERE, "..", "notebooks", "NB11a_DeltaSyncPrep.py"))


def _nb11a_source():
    with open(NB11A, "r", encoding="utf-8") as fh:
        return fh.read()


class TestWatermarkTypeSupport:
    @pytest.mark.parametrize("dtype", [
        "DATE",
        "TIMESTAMP",
        "TIMESTAMP(6)",
        "TIMESTAMP(9)",
        "TIMESTAMP WITH TIME ZONE",
        "TIMESTAMP(6) WITH TIME ZONE",
        "TIMESTAMP WITH LOCAL TIME ZONE",
        "TIMESTAMP(6) WITH LOCAL TIME ZONE",
    ])
    def test_supported_temporal_types(self, dtype):
        assert strat.is_supported_watermark_type(dtype) is True

    @pytest.mark.parametrize("dtype", [
        "NUMBER", "BINARY_DOUBLE", "BINARY_FLOAT", "DECIMAL(38,0)",
        "VARCHAR2", "CHAR", "BOOLEAN",
        "INTERVAL DAY TO SECOND", "INTERVAL YEAR TO MONTH",
    ])
    def test_rejected_non_temporal_types(self, dtype):
        assert strat.is_supported_watermark_type(dtype) is False

    def test_empty_type_rejected(self):
        assert strat.is_supported_watermark_type("") is False

    def test_null_type_rejected(self):
        assert strat.is_supported_watermark_type(None) is False

    def test_leading_trailing_spaces_normalized(self):
        assert strat.normalize_watermark_type("  TIMESTAMP  ") == "TIMESTAMP"
        assert strat.is_supported_watermark_type("   DATE   ") is True

    def test_repeated_whitespace_normalized(self):
        assert strat.normalize_watermark_type(
            "TIMESTAMP    WITH   TIME  ZONE") == "TIMESTAMP WITH TIME ZONE"
        assert strat.is_supported_watermark_type(
            "TIMESTAMP   WITH   LOCAL   TIME   ZONE") is True

    def test_lowercase_temporal_normalized(self):
        assert strat.normalize_watermark_type(
            "timestamp(6) with time zone") == "TIMESTAMP WITH TIME ZONE"
        assert strat.is_supported_watermark_type("timestamp(9)") is True

    def test_precision_stripped(self):
        assert strat.normalize_watermark_type("TIMESTAMP(6)") == "TIMESTAMP"
        assert strat.normalize_watermark_type(
            "TIMESTAMP(6) WITH LOCAL TIME ZONE") == "TIMESTAMP WITH LOCAL TIME ZONE"

    def test_watermark_number_not_eligible(self):
        assert strat.is_supported_watermark_type("NUMBER") is False

    def test_hybrid_number_not_eligible(self):
        assert strat.is_supported_watermark_type("NUMBER") is False

    def test_watermark_timestamp_eligible(self):
        assert strat.is_supported_watermark_type("TIMESTAMP") is True

    def test_hybrid_timestamp_eligible(self):
        assert strat.is_supported_watermark_type("TIMESTAMP(6)") is True


class TestNB11aStaticGuards:
    def test_uses_shared_helper(self):
        src = _nb11a_source()
        assert "is_supported_watermark_type" in src
        assert "normalize_watermark_type" in src

    def test_defines_error_markers(self):
        src = _nb11a_source()
        assert "NON_TEMPORAL_WATERMARK" in src
        assert "DELTA_CONFIG_ERROR" in src

    def test_no_numeric_coercion_branch(self):
        # active numeric watermark conversion must be gone
        src = _nb11a_source()
        assert "BINARY_DOUBLE" not in src
        assert "BINARY_FLOAT" not in src
        assert "Decimal" not in src
        assert "InvalidOperation" not in src

    def test_coerce_rejects_non_temporal(self):
        assert "Unsupported non-temporal watermark type" in _nb11a_source()

    def test_validation_before_capture_and_query(self):
        src = _nb11a_source()
        guard = src.index("NON_TEMPORAL_WATERMARK")
        # The temporal guard must run before the upper watermark is captured,
        # before the incremental query is built, and before the row is queued.
        # The call site wraps its args onto the next line; the def does not.
        assert guard < src.index("capture_upper_watermark(\n")
        assert guard < src.index("incremental_extract_query")
        assert guard < src.index("queue.append")

    def test_primary_key_builds_full_extract_without_watermark(self):
        assert "full_extract_query" in _nb11a_source()


# ============================================================ delta-prep guard (helper level)
class TestDeltaPrepGuard:
    def test_watermark_with_number_rejected(self):
        assert strat.is_supported_watermark_type("NUMBER") is False

    def test_hybrid_with_number_rejected(self):
        assert strat.is_supported_watermark_type("NUMBER") is False

    def test_watermark_with_timestamp_accepted(self):
        assert strat.is_supported_watermark_type("TIMESTAMP") is True

    def test_hybrid_with_created_date_accepted(self):
        # CREATED_DATE stored as TIMESTAMP remains an accepted temporal watermark
        assert strat.is_supported_watermark_type("TIMESTAMP(6)") is True
        cols = [
            {"column_name": "id", "data_type": "NUMBER", "scale": 0, "ordinal_position": 1},
            {"column_name": "created_date", "data_type": "TIMESTAMP(6)", "ordinal_position": 2},
        ]
        s, wm, fam = strat.detect_strategy(cols, ["id"])
        assert s == strat.HYBRID and wm == "created_date"

    def test_primary_key_does_not_require_watermark(self):
        cols = [{"column_name": "id", "data_type": "NUMBER", "scale": 0}]
        s, wm, fam = strat.detect_strategy(cols, ["id"])
        assert s == strat.PRIMARY_KEY and wm is None and fam is None


# ============================================================ timezone handling
COMMON_NB = os.path.abspath(os.path.join(HERE, "..", "notebooks", "_common.py"))


class TestTimezoneHandling:
    def test_utc_naive_timestamp_serialized_as_utc(self):
        out = sqlb.canonical_watermark_string("2026-08-18 07:00:45.123456")
        assert out == "2026-08-18T07:00:45.123456Z"

    def test_date_serialized_as_utc(self):
        from datetime import date
        out = sqlb.canonical_watermark_string(date(2026, 8, 18))
        assert out.endswith("Z") and out.startswith("2026-08-18T00:00:00")

    def test_tz_value_normalized_to_utc(self):
        out = sqlb.canonical_watermark_string("2026-08-18T12:30:45.123456+05:30")
        assert out == "2026-08-18T07:00:45.123456Z"

    def test_timestamp_z_produces_tz_literal(self):
        lit = sqlb._format_watermark_literal(
            "2026-08-18T07:00:45.123456Z", "TIMESTAMP WITH TIME ZONE")
        assert "TO_TIMESTAMP_TZ(" in lit and "+00:00" in lit

    def test_positive_offset_literal(self):
        lit = sqlb._format_watermark_literal(
            "2026-08-18T12:30:45.123456+05:30", "TIMESTAMP WITH TIME ZONE")
        assert "TO_TIMESTAMP_TZ(" in lit and "+05:30" in lit

    def test_negative_offset_literal(self):
        lit = sqlb._format_watermark_literal(
            "2026-08-18T00:30:45.123456-07:00", "TIMESTAMP WITH TIME ZONE")
        assert "TO_TIMESTAMP_TZ(" in lit and "-07:00" in lit

    def test_local_tz_uses_tz_literal(self):
        lit = sqlb._format_watermark_literal(
            "2026-08-18T07:00:45.123456Z", "TIMESTAMP WITH LOCAL TIME ZONE")
        assert "TO_TIMESTAMP_TZ(" in lit

    def test_plain_timestamp_uses_utc_wallclock(self):
        lit = sqlb._format_watermark_literal(
            "2026-08-18T07:00:45.123456Z", "TIMESTAMP")
        assert "TO_TIMESTAMP(" in lit and "TO_TIMESTAMP_TZ(" not in lit

    def test_lower_and_upper_use_same_policy(self):
        q = sqlb.build_incremental_extract_query(
            "HR", "EMP", "UPDATED_AT", "TIMESTAMP WITH TIME ZONE",
            "2026-08-18T07:00:45.123456Z", "2026-08-19T07:00:45.123456Z")
        assert q.count("TO_TIMESTAMP_TZ(") == 2

    def test_spark_session_timezone_set_to_utc(self):
        with open(COMMON_NB, "r", encoding="utf-8") as fh:
            src = fh.read()
        assert "spark.sql.session.timeZone" in src
        assert '"UTC"' in src


# ============================================================ canonical checkpoint serialization
class TestCanonicalWatermarkString:
    def test_naive_datetime_to_utc_z(self):
        from datetime import datetime
        assert sqlb.canonical_watermark_string(
            datetime(2026, 8, 18, 7, 0, 45, 123456)) == "2026-08-18T07:00:45.123456Z"

    def test_aware_utc_datetime_to_z(self):
        from datetime import datetime, timezone
        assert sqlb.canonical_watermark_string(
            datetime(2026, 8, 18, 7, 0, 45, 123456, tzinfo=timezone.utc)) == \
            "2026-08-18T07:00:45.123456Z"

    def test_positive_offset_to_utc_z(self):
        assert sqlb.canonical_watermark_string(
            "2026-08-18T12:30:45.123456+05:30") == "2026-08-18T07:00:45.123456Z"

    def test_negative_offset_to_utc_z(self):
        assert sqlb.canonical_watermark_string(
            "2026-08-18T00:30:45.123456-07:00") == "2026-08-18T07:30:45.123456Z"

    def test_python_date_to_midnight_utc(self):
        from datetime import date
        assert sqlb.canonical_watermark_string(
            date(2026, 8, 18)) == "2026-08-18T00:00:00.000000Z"

    def test_legacy_space_string_parses(self):
        assert sqlb.canonical_watermark_string(
            "2026-08-18 07:00:45.123456") == "2026-08-18T07:00:45.123456Z"

    def test_iso_string_parses(self):
        assert sqlb.canonical_watermark_string(
            "2026-08-18T07:00:45.123456") == "2026-08-18T07:00:45.123456Z"

    def test_iso_z_parses(self):
        assert sqlb.canonical_watermark_string(
            "2026-08-18T07:00:45.123456Z") == "2026-08-18T07:00:45.123456Z"

    def test_iso_positive_offset_parses(self):
        assert sqlb.canonical_watermark_string(
            "2026-08-18T09:00:45.123456+02:00") == "2026-08-18T07:00:45.123456Z"

    def test_iso_negative_offset_parses(self):
        assert sqlb.canonical_watermark_string(
            "2026-08-18T05:00:45.123456-02:00") == "2026-08-18T07:00:45.123456Z"

    def test_microseconds_always_present(self):
        out = sqlb.canonical_watermark_string("2026-08-18 07:00:45")
        assert out == "2026-08-18T07:00:45.000000Z" and ".000000Z" in out

    def test_none_returns_none(self):
        assert sqlb.canonical_watermark_string(None) is None

    def test_strict_raises_on_malformed(self):
        with pytest.raises(ValueError):
            sqlb.canonical_watermark_string("not-a-timestamp", strict=True)

    def test_permissive_returns_malformed_unchanged(self):
        assert sqlb.canonical_watermark_string("not-a-timestamp") == "not-a-timestamp"


# ============================================================ checkpoint canonicalization (static)
NB10 = os.path.abspath(os.path.join(HERE, "..", "notebooks", "NB10_PostFullLoadState.py"))
NB11B = os.path.abspath(os.path.join(HERE, "..", "notebooks", "NB11b_DeltaSyncApply.py"))
STRATEGY_SRC = os.path.abspath(os.path.join(HERE, "..", "src", "strategy.py"))


def _read(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


class TestCheckpointCanonicalization:
    def test_nb10_uses_canonical_serializer(self):
        src = _read(NB10)
        assert "canonical_watermark_string(val" in src
        assert "new_wm = str(val)" not in src

    def test_nb11a_capture_uses_canonical(self):
        src = _nb11a_source()
        assert "canonical_watermark_string(raw" in src
        assert "None if raw is None else str(raw)" not in src

    def test_nb11b_commits_canonical_checkpoint(self):
        src = _read(NB11B)
        assert "canonical_watermark_string(" in src
        assert "last_watermark_value" in src

    def test_nb11b_checkpoint_before_queue_success(self):
        src = _read(NB11B)
        commit = src.index("repo.update_control(src_id, control_fields)")
        queue_done = src.index('_mark_queue("SUCCEEDED")')
        assert commit < queue_done


class TestNB11bTemporalGuard:
    def test_number_rejected_helper(self):
        assert strat.is_supported_watermark_type("NUMBER") is False

    def test_varchar2_rejected_helper(self):
        assert strat.is_supported_watermark_type("VARCHAR2") is False

    def test_timestamp_accepted_helper(self):
        assert strat.is_supported_watermark_type("TIMESTAMP") is True

    def test_date_accepted_helper(self):
        assert strat.is_supported_watermark_type("DATE") is True

    def test_nb11b_imports_shared_helpers(self):
        src = _read(NB11B)
        assert "is_supported_watermark_type" in src
        assert "normalize_watermark_type" in src

    def test_nb11b_has_temporal_guard(self):
        src = _read(NB11B)
        assert "Unsupported non-temporal watermark type" in src

    def test_nb11b_numeric_code_removed(self):
        src = _read(NB11B)
        assert "BINARY_DOUBLE" not in src
        assert "BINARY_FLOAT" not in src
        assert "Decimal" not in src
        assert "InvalidOperation" not in src
        assert "Invalid numeric watermark" not in src

    def test_nb11b_delta_literal_uses_canonical_cast(self):
        src = _read(NB11B)
        assert "CAST('" in src and "AS TIMESTAMP)" in src
        assert "canonical_watermark_string(value, strict=True)" in src

    def test_nb11b_preserves_delete_interval(self):
        src = _read(NB11B)
        assert "> {lower_lit}" in src and "<= {upper_lit}" in src

    def test_nb11b_guard_before_read(self):
        src = _read(NB11B)
        guard = src.index("Unsupported non-temporal watermark type")
        assert guard < src.index('q["source_query"]')


class TestStrategyDocumentation:
    def test_no_stale_name_eligibility_comment(self):
        src = _read(STRATEGY_SRC)
        assert "must also have approved change semantics" not in src
        assert "temporal type alone is still" not in src

    def test_ranking_only_policy_documented(self):
        src = _read(STRATEGY_SRC)
        assert "ranking only" in src

    def test_temporal_eligibility_unchanged(self):
        assert strat.is_supported_watermark_type("TIMESTAMP(6)") is True
        assert strat.is_supported_watermark_type("NUMBER") is False


# ============================================================ JDBC partition safety
NB09 = os.path.abspath(os.path.join(HERE, "..", "notebooks", "NB09_FullLoad.py"))

_LONG_MAX = 2 ** 63 - 1
_LONG_MIN = -(2 ** 63)


class TestNormalizeSignedLongBound:
    def test_integral_decimal_string_normalizes(self):
        assert part.normalize_signed_long_bound("1007.0000000000") == 1007
        assert part.normalize_signed_long_bound("2001.0000000000") == 2001
        assert part.normalize_signed_long_bound("1.0000000000") == 1

    def test_decimal_object_normalizes(self):
        from decimal import Decimal
        assert part.normalize_signed_long_bound(Decimal("2001.0000000000")) == 2001

    def test_plain_int_and_negative_and_zero(self):
        assert part.normalize_signed_long_bound(1007) == 1007
        assert part.normalize_signed_long_bound(0) == 0
        assert part.normalize_signed_long_bound("-5.0000000000") == -5

    def test_nonzero_fraction_rejected(self):
        assert part.normalize_signed_long_bound("1007.5") is None
        assert part.normalize_signed_long_bound("2.0000000001") is None

    def test_malformed_rejected(self):
        assert part.normalize_signed_long_bound("abc") is None
        assert part.normalize_signed_long_bound("") is None
        assert part.normalize_signed_long_bound(None) is None

    def test_nan_and_infinity_rejected(self):
        assert part.normalize_signed_long_bound("NaN") is None
        assert part.normalize_signed_long_bound("Infinity") is None
        assert part.normalize_signed_long_bound("-Infinity") is None

    def test_signed_64_bit_limits(self):
        assert part.normalize_signed_long_bound(str(_LONG_MAX)) == _LONG_MAX
        assert part.normalize_signed_long_bound(str(_LONG_MIN)) == _LONG_MIN
        assert part.normalize_signed_long_bound(str(_LONG_MAX + 1)) is None
        assert part.normalize_signed_long_bound(str(_LONG_MIN - 1)) is None


class TestResolvePartitioning:
    def test_unconstrained_number_disables(self):
        eff, lo, hi, reason = part.resolve_partitioning(
            "NUMBER", None, None, "DECIMAL(38,0)", "1.0000000000", "1007.0000000000", 8)
        assert eff is None and lo is None and hi is None and reason is not None

    def test_number_18_bigint_enables(self):
        eff, lo, hi, reason = part.resolve_partitioning(
            "NUMBER", 18, 0, "BIGINT", "1.0000000000", "1007.0000000000", 8)
        assert reason is None and lo == 1 and hi == 1007 and eff == 8

    def test_decimal_target_cannot_enable(self):
        eff, lo, hi, reason = part.resolve_partitioning(
            "NUMBER", 10, 0, "DECIMAL(38,0)", 1, 1007, 8)
        assert eff is None and reason is not None

    def test_precision_over_18_disables(self):
        eff, lo, hi, reason = part.resolve_partitioning(
            "NUMBER", 19, 0, "BIGINT", 1, 1007, 8)
        assert eff is None and reason is not None

    def test_nonzero_scale_disables(self):
        eff, lo, hi, reason = part.resolve_partitioning(
            "NUMBER", 10, 2, "BIGINT", 1, 1007, 8)
        assert eff is None and reason is not None

    def test_equal_bounds_disables(self):
        eff, lo, hi, reason = part.resolve_partitioning(
            "NUMBER", 10, 0, "INT", 5, 5, 8)
        assert eff is None and reason is not None

    def test_reversed_bounds_disables(self):
        eff, lo, hi, reason = part.resolve_partitioning(
            "NUMBER", 10, 0, "INT", 100, 5, 8)
        assert eff is None and reason is not None

    def test_num_partitions_le_1_disables(self):
        eff, lo, hi, reason = part.resolve_partitioning(
            "NUMBER", 10, 0, "INT", 1, 1007, 1)
        assert eff is None and reason is not None

    def test_effective_partitions_capped_by_range(self):
        # range 1..3 -> 3 distinct values -> effective capped at 3 even if 8 requested
        eff, lo, hi, reason = part.resolve_partitioning(
            "NUMBER", 3, 0, "SMALLINT", 1, 3, 8)
        assert reason is None and eff == 3 and lo == 1 and hi == 3

    def test_decimal_formatted_bounds_normalized(self):
        eff, lo, hi, reason = part.resolve_partitioning(
            "NUMBER", 10, 0, "INT", "1007.0000000000", "5000.0000000000", 8)
        assert reason is None and lo == 1007 and hi == 5000

    def test_is_integer_like_target_excludes_decimal(self):
        assert part.is_integer_like_target("BIGINT") is True
        assert part.is_integer_like_target("INT") is True
        assert part.is_integer_like_target("SMALLINT") is True
        assert part.is_integer_like_target("DECIMAL(38,0)") is False


class TestNB09PartitionWiring:
    def test_no_decimal_in_partition_eligibility(self):
        src = _read(NB09)
        # DECIMAL must not appear in an integer-like partition check anymore
        assert "startswith(\"DECIMAL\")" not in src
        assert "_integer_like" not in src

    def test_uses_resolve_partitioning(self):
        src = _read(NB09)
        # NB09 now routes partition planning through the source adapter.
        assert "resolve_partition_plan" in src

    def test_reads_source_inventory_metadata(self):
        src = _read(NB09)
        assert "source_inventory" in src
        assert "numeric_precision" in src and "numeric_scale" in src

    def test_disabled_message_present(self):
        src = _read(NB09)
        assert "JDBC partitioning disabled for" in src
        assert "correctness-safe unpartitioned read" in src

    def test_extract_query_not_cast(self):
        # the source SELECT must remain the plain full extract (no PK cast)
        src = _read(NB09)
        assert "full_extract_query(" in src and 'watermark_column=d.get("watermark_column")' in src
        assert "CAST(" not in src

    def test_dataframe_cached_once_and_unpersisted(self):
        src = _read(NB09)
        assert src.count(".cache()") == 2  # partitioned + unpartitioned branches
        assert "src_df.unpersist()" in src

    def test_composite_pk_disabled_in_notebook(self):
        src = _read(NB09)
        assert "len(pk) != 1" in src


# ============================================================ sql_builder temporal-only literals
SQL_BUILDER_SRC = os.path.abspath(os.path.join(HERE, "..", "src", "sql_builder.py"))


class TestSqlBuilderTemporalOnly:
    @pytest.mark.parametrize("family", [
        "DATE", "TIMESTAMP", "TIMESTAMP(6)",
        "TIMESTAMP WITH TIME ZONE", "TIMESTAMP(6) WITH TIME ZONE",
        "TIMESTAMP WITH LOCAL TIME ZONE",
    ])
    def test_temporal_families_produce_literal(self, family):
        lit = sqlb._format_watermark_literal("2026-08-18T07:00:45.123456Z", family)
        assert lit.startswith("TO_")

    def test_date_literal(self):
        assert "TO_DATE(" in sqlb._format_watermark_literal(
            "2026-08-18T00:00:00Z", "DATE")

    def test_timestamp_literal(self):
        lit = sqlb._format_watermark_literal("2026-08-18T07:00:45.123456Z", "TIMESTAMP")
        assert "TO_TIMESTAMP(" in lit and "TO_TIMESTAMP_TZ(" not in lit

    def test_tz_literal(self):
        assert "TO_TIMESTAMP_TZ(" in sqlb._format_watermark_literal(
            "2026-08-18T07:00:45.123456Z", "TIMESTAMP WITH TIME ZONE")

    def test_local_tz_literal(self):
        assert "TO_TIMESTAMP_TZ(" in sqlb._format_watermark_literal(
            "2026-08-18T07:00:45.123456Z", "TIMESTAMP WITH LOCAL TIME ZONE")

    def test_lower_and_upper_same_family(self):
        q = sqlb.build_incremental_extract_query(
            "HR", "EMP", "UPDATED_AT", "TIMESTAMP",
            "2026-08-18T07:00:45.123456Z", "2026-08-19T07:00:45.123456Z")
        assert q.count("TO_TIMESTAMP(") == 2

    def test_exact_interval_preserved(self):
        q = sqlb.build_incremental_extract_query(
            "HR", "EMP", "UPDATED_AT", "TIMESTAMP",
            "2026-08-18T07:00:45.123456Z", "2026-08-19T07:00:45.123456Z")
        assert '"UPDATED_AT" >' in q and '"UPDATED_AT" <=' in q

    @pytest.mark.parametrize("family", [
        "NUMBER", "BINARY_DOUBLE", "BINARY_FLOAT", "DECIMAL(38,0)",
        "INTEGER", "FLOAT", "VARCHAR2", "CHAR", "BOOLEAN",
        "INTERVAL DAY TO SECOND", "", None,
    ])
    def test_non_temporal_family_raises(self, family):
        with pytest.raises(ValueError):
            sqlb._format_watermark_literal("2026-08-18T07:00:45.123456Z", family)

    def test_null_bound_raises(self):
        with pytest.raises(ValueError):
            sqlb._format_watermark_literal(None, "TIMESTAMP")

    def test_build_incremental_rejects_numeric_family(self):
        with pytest.raises(ValueError):
            sqlb.build_incremental_extract_query(
                "HR", "EMP", "VERSION", "NUMBER",
                "2026-08-18T07:00:45.123456Z", "2026-08-19T07:00:45.123456Z")

    def test_numeric_code_absent_from_source(self):
        src = _read(SQL_BUILDER_SRC)
        assert "from decimal import" not in src
        assert "Invalid numeric watermark" not in src
        assert "BINARY_DOUBLE" not in src
        assert "BINARY_FLOAT" not in src
        assert "NUMBER" not in src


# ============================================================ NB11b staged failure handling
class TestNB11bStagedFailure:
    def test_operational_states_present(self):
        src = _read(NB11B)
        assert "DELTA_FAILED" in src
        assert "CHECKPOINT_COMMIT_FAILED" in src
        assert "QUEUE_FINALIZATION_FAILED" in src
        assert "FAILED_FINALIZATION" in src

    def test_stage_flags_present(self):
        src = _read(NB11B)
        assert "data_applied" in src
        assert "checkpoint_committed" in src

    def test_checkpoint_committed_before_queue_success(self):
        src = _read(NB11B)
        assert src.index("checkpoint_committed = True") < src.index('_mark_queue("SUCCEEDED")')

    def test_checkpoint_failure_continues_not_delta_failed(self):
        src = _read(NB11B)
        cf = src.index("CHECKPOINT_COMMIT_FAILED")
        assert "continue" in src[cf:cf + 900]

    def test_finalization_failure_continues(self):
        src = _read(NB11B)
        qf = src.index("QUEUE_FINALIZATION_FAILED")
        assert "continue" in src[qf:qf + 1500]

    def test_finalization_failure_preserves_watermark(self):
        # the finalization handler must not touch last_watermark_value
        src = _read(NB11B)
        qf = src.index("QUEUE_FINALIZATION_FAILED")
        assert "last_watermark_value" not in src[qf:qf + 700]

    def test_finalization_attempts_failed_finalization_mark(self):
        src = _read(NB11B)
        qf = src.index("QUEUE_FINALIZATION_FAILED")
        assert "FAILED_FINALIZATION" in src[qf:qf + 700]

    def test_uses_canonical_and_guard(self):
        src = _read(NB11B)
        assert "canonical_watermark_string" in src
        assert "is_supported_watermark_type" in src

    def test_primary_key_watermark_conditional(self):
        # only WATERMARK/HYBRID gets a temporal checkpoint
        src = _read(NB11B)
        assert 'if strategy in ("WATERMARK", "HYBRID"):' in src
        assert 'control_fields["last_watermark_value"]' in src

    def test_notebook_raises_on_any_failure(self):
        src = _read(NB11B)
        assert "if failed > 0:" in src


# ============================================================ FULL_LOAD delta refresh support
NB12 = os.path.abspath(os.path.join(HERE, "..", "notebooks", "NB12_ValidationAndReconciliation.py"))


class TestFullLoadDeltaSupport:
    # ---- NB11a queue behavior ----
    def test_nb11a_eligibility_includes_full_load(self):
        src = _read(NB11A)
        assert "'WATERMARK','PRIMARY_KEY','HYBRID','FULL_LOAD'" in src

    def test_nb11a_full_load_branch(self):
        src = _read(NB11A)
        assert 'if strategy == "FULL_LOAD":' in src

    def test_nb11a_full_load_builds_full_extract(self):
        src = _read(NB11A)
        fl = src.index('if strategy == "FULL_LOAD":')
        segment = src[fl:fl + 300]
        assert "full_extract_query(" in segment and "columns=approved_columns" in segment

    def test_nb11a_full_load_nulls_watermark_fields(self):
        src = _read(NB11A)
        assert "wm_col = wm_type = last_wm = upper_wm = None" in src

    def test_nb11a_full_load_queued_status(self):
        src = _read(NB11A)
        assert "DELTA_FULL_REFRESH_QUEUED" in src

    # ---- NB11b apply behavior ----
    def test_nb11b_full_load_branch_overwrite(self):
        src = _read(NB11B)
        fl = src.index('if strategy == "FULL_LOAD":')
        segment = src[fl:fl + 400]
        assert '.mode("overwrite")' in segment
        assert "saveAsTable(plain_target)" in segment
        assert 'op = "DELTA_FULL_REFRESH"' in segment

    def test_nb11b_full_load_not_append(self):
        src = _read(NB11B)
        fl = src.index('if strategy == "FULL_LOAD":')
        segment = src[fl:fl + 400]
        assert '.mode("append")' not in segment

    def test_nb11b_full_load_not_merge(self):
        src = _read(NB11B)
        fl = src.index('if strategy == "FULL_LOAD":')
        segment = src[fl:fl + 400]
        assert "build_merge_sql" not in segment

    def test_nb11b_full_load_count_must_match(self):
        src = _read(NB11B)
        assert 'strategy == "FULL_LOAD" and t_count != s_count' in src
        assert "FULL_LOAD refresh count mismatch" in src

    def test_nb11b_full_load_logs_operation(self):
        assert "DELTA_FULL_REFRESH" in _read(NB11B)

    def test_nb11b_full_load_success_status(self):
        assert "DELTA_FULL_REFRESH_SUCCEEDED" in _read(NB11B)

    def test_nb11b_full_load_no_watermark_update(self):
        # last_watermark_value is only ever set for WATERMARK/HYBRID
        src = _read(NB11B)
        assert 'if strategy in ("WATERMARK", "HYBRID"):' in src
        wm = src.index('control_fields["last_watermark_value"]')
        guard = src.rindex('if strategy in ("WATERMARK", "HYBRID"):', 0, wm)
        assert wm - guard < 200

    def test_nb11b_queue_success_after_control(self):
        src = _read(NB11B)
        assert src.index("checkpoint_committed = True") < src.index('_mark_queue("SUCCEEDED")')

    def test_nb11b_source_always_unpersisted(self):
        src = _read(NB11B)
        assert "src_df.unpersist()" in src
        assert "finally:" in src

    # ---- NB12 reconciliation behavior ----
    def test_nb12_delta_includes_full_refresh(self):
        src = _read(NB12)
        assert "'DELTA_MERGE','DELTA_APPEND','DELTA_FULL_REFRESH'" in src

    def test_nb12_full_refresh_row_count_check(self):
        src = _read(NB12)
        assert 'r["operation"] == "DELTA_FULL_REFRESH"' in src
        fr = src.index('r["operation"] == "DELTA_FULL_REFRESH"')
        segment = src[fr:fr + 600]
        assert "src_count != tgt_count" in segment
        assert '"FAIL"' in segment and '"PASS"' in segment

    def test_nb12_no_watermark_check_for_full_load(self):
        # the delta watermark boundary report is scoped to WATERMARK/HYBRID only,
        # so FULL_LOAD never gets a watermark reconciliation row
        src = _read(NB12)
        assert "q.load_strategy IN ('WATERMARK','HYBRID')" in src
