#!/usr/bin/env python3
"""Unit tests for the feedback analyzer."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from analyzer import (
    DataLoadError,
    DataValidationError,
    FeedbackAnalyzerError,
    delete_keys_from_nested_json,
    export_statistics,
    export_to_json,
    filter_by_date_range,
    generate_statistics,
    load_feedback_data,
    make_serializable,
    parse_args,
    safe_get,
)


class TestSafeGet:
    """Tests for safe_get function."""

    def test_simple_key(self):
        data = {"a": 1, "b": 2}
        assert safe_get(data, "a") == 1
        assert safe_get(data, "b") == 2

    def test_nested_keys(self):
        data = {"a": {"b": {"c": 3}}}
        assert safe_get(data, "a", "b", "c") == 3

    def test_missing_key_returns_default(self):
        data = {"a": 1}
        assert safe_get(data, "b") is None
        assert safe_get(data, "b", default="missing") == "missing"

    def test_missing_nested_key_returns_default(self):
        data = {"a": {"b": 1}}
        assert safe_get(data, "a", "c") is None
        assert safe_get(data, "a", "b", "c") is None

    def test_non_dict_returns_default(self):
        assert safe_get("string", "key") is None
        assert safe_get(123, "key") is None
        assert safe_get(None, "key") is None

    def test_empty_dict(self):
        assert safe_get({}, "a") is None


class TestDeleteKeysFromNestedJson:
    """Tests for delete_keys_from_nested_json function."""

    def test_delete_simple_key(self):
        data = {"a": 1, "b": 2, "c": 3}
        result = delete_keys_from_nested_json(data, "b")
        assert result == {"a": 1, "c": 3}

    def test_delete_nested_key(self):
        data = {"a": {"b": 1, "c": 2}, "d": 3}
        result = delete_keys_from_nested_json(data, "b")
        assert result == {"a": {"c": 2}, "d": 3}

    def test_delete_key_in_list(self):
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        result = delete_keys_from_nested_json(data, "b")
        assert result == [{"a": 1}, {"a": 3}]

    def test_delete_deeply_nested(self):
        data = {"level1": {"level2": {"level3": {"target": "delete", "keep": "value"}}}}
        result = delete_keys_from_nested_json(data, "target")
        assert result == {"level1": {"level2": {"level3": {"keep": "value"}}}}

    def test_preserves_primitives(self):
        assert delete_keys_from_nested_json("string", "key") == "string"
        assert delete_keys_from_nested_json(123, "key") == 123
        assert delete_keys_from_nested_json(None, "key") is None


class TestLoadFeedbackData:
    """Tests for load_feedback_data function."""

    def test_load_valid_json(self):
        data = [{"id": 1, "created_at": 1234567890}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            result = load_feedback_data(Path(f.name))
        assert result == data

    def test_file_not_found(self):
        with pytest.raises(DataLoadError, match="File not found"):
            load_feedback_data(Path("/nonexistent/file.json"))

    def test_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json{")
            f.flush()
            with pytest.raises(DataLoadError, match="Invalid JSON"):
                load_feedback_data(Path(f.name))

    def test_non_list_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"not": "a list"}, f)
            f.flush()
            with pytest.raises(DataValidationError, match="Expected list"):
                load_feedback_data(Path(f.name))


class TestFilterByDateRange:
    """Tests for filter_by_date_range function."""

    def test_filter_basic(self):
        # Timestamps for 2024-12-05 and 2024-12-15 (UTC)
        data = [
            {"id": 1, "created_at": 1733356800},  # 2024-12-05 00:00:00 UTC
            {"id": 2, "created_at": 1734220800},  # 2024-12-15 00:00:00 UTC
        ]
        result = filter_by_date_range(data, "2024-12-01", "2024-12-10", timezone="UTC")
        assert len(result) == 1
        assert result.iloc[0]["id"] == 1

    def test_empty_data(self):
        with pytest.raises(DataValidationError, match="No data to filter"):
            filter_by_date_range([], "2024-01-01", "2024-01-31")

    def test_missing_created_at(self):
        data = [{"id": 1}]
        with pytest.raises(DataValidationError, match="missing required 'created_at'"):
            filter_by_date_range(data, "2024-01-01", "2024-01-31")

    def test_inclusive_date_range(self):
        # Test that both start and end dates are inclusive
        data = [
            {"id": 1, "created_at": 1733011200},  # 2024-12-01 00:00:00 UTC
            {"id": 2, "created_at": 1733097599},  # 2024-12-01 23:59:59 UTC
            {"id": 3, "created_at": 1733097600},  # 2024-12-02 00:00:00 UTC
        ]
        result = filter_by_date_range(data, "2024-12-01", "2024-12-01", timezone="UTC")
        assert len(result) == 2

    def test_no_date_range_returns_all(self):
        # When start_date and end_date are None, return all data
        data = [
            {"id": 1, "created_at": 1733011200},  # 2024-12-01
            {"id": 2, "created_at": 1734220800},  # 2024-12-15
            {"id": 3, "created_at": 1735689600},  # 2025-01-01
        ]
        result = filter_by_date_range(data, None, None, timezone="UTC")
        assert len(result) == 3

    def test_only_start_date(self):
        # When only start_date is provided, filter from start to latest
        data = [
            {"id": 1, "created_at": 1733011200},  # 2024-12-01
            {"id": 2, "created_at": 1734220800},  # 2024-12-15
            {"id": 3, "created_at": 1735689600},  # 2025-01-01
        ]
        result = filter_by_date_range(data, "2024-12-10", None, timezone="UTC")
        assert len(result) == 2
        assert result.iloc[0]["id"] == 2
        assert result.iloc[1]["id"] == 3

    def test_only_end_date(self):
        # When only end_date is provided, filter from earliest to end
        data = [
            {"id": 1, "created_at": 1733011200},  # 2024-12-01
            {"id": 2, "created_at": 1734220800},  # 2024-12-15
            {"id": 3, "created_at": 1735689600},  # 2025-01-01
        ]
        result = filter_by_date_range(data, None, "2024-12-10", timezone="UTC")
        assert len(result) == 1
        assert result.iloc[0]["id"] == 1


class TestMakeSerializable:
    """Tests for make_serializable function."""

    def test_dict_with_non_string_keys(self):
        data = {1: "a", 2: "b"}
        result = make_serializable(data)
        assert result == {"1": "a", "2": "b"}

    def test_nested_structure(self):
        data = {"a": [1, 2, {"b": 3}]}
        result = make_serializable(data)
        assert result == {"a": [1, 2, {"b": 3}]}

    def test_float_rounding(self):
        data = {"rate": 0.123456789}
        result = make_serializable(data)
        assert result["rate"] == 0.1235

    def test_counter(self):
        from collections import Counter

        data = {"counts": Counter(["a", "a", "b"])}
        result = make_serializable(data)
        assert result == {"counts": {"a": 2, "b": 1}}


class TestParseArgs:
    """Tests for parse_args function."""

    def test_with_dates(self):
        args = parse_args(["input.json", "-s", "2025-01-01", "-e", "2025-01-31"])
        assert args.input == Path("input.json")
        assert args.start == "2025-01-01"
        assert args.end == "2025-01-31"

    def test_default_values(self):
        args = parse_args(["input.json"])
        assert args.input == Path("input.json")
        assert args.start is None
        assert args.end is None
        assert args.output_dir == Path(".")
        assert args.timezone == "Asia/Hong_Kong"
        assert args.no_export is False
        assert args.quiet is False
        assert args.verbose is False

    def test_custom_options(self):
        args = parse_args([
            "input.json",
            "-s", "2025-01-01",
            "-e", "2025-01-31",
            "-o", "/output",
            "--timezone", "UTC",
            "--no-export",
            "-q",
        ])
        assert args.output_dir == Path("/output")
        assert args.timezone == "UTC"
        assert args.no_export is True
        assert args.quiet is True

    def test_missing_input_file(self):
        with pytest.raises(SystemExit):
            parse_args([])  # Missing input file


class TestExportFunctions:
    """Tests for export functions."""

    def test_export_to_json(self):
        df = pd.DataFrame([
            {"id": 1, "created_at": pd.Timestamp("2025-01-01", tz="UTC")},
        ])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            export_to_json(df, Path(f.name))
            with open(f.name) as rf:
                result = json.load(rf)
        assert len(result) == 1
        assert result[0]["id"] == 1
        assert isinstance(result[0]["created_at"], int)

    def test_export_statistics(self):
        stats = {"overview": {"total": 100}, "rate": 0.5}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            export_statistics(stats, Path(f.name))
            with open(f.name) as rf:
                result = json.load(rf)
        assert result == {"overview": {"total": 100}, "rate": 0.5}


class TestGenerateStatistics:
    """Tests for generate_statistics function."""

    def test_basic_statistics(self):
        df = pd.DataFrame([
            {
                "created_at": pd.Timestamp("2025-01-01 10:00:00", tz="UTC"),
                "data": {"rating": 1, "reason": "helpful", "model_id": "gpt-4"},
            },
            {
                "created_at": pd.Timestamp("2025-01-01 14:00:00", tz="UTC"),
                "data": {"rating": -1, "reason": "incorrect", "model_id": "gpt-4"},
            },
        ])
        stats = generate_statistics(df)

        assert stats["overview"]["total_records"] == 2
        assert stats["rating_analysis"]["thumbs_up_rate"] == 0.5
        assert stats["reason_analysis"]["distribution"]["helpful"] == 1
        assert stats["model_analysis"]["distribution"]["gpt-4"] == 2

    def test_empty_rag_analysis(self):
        df = pd.DataFrame([
            {
                "created_at": pd.Timestamp("2025-01-01", tz="UTC"),
                "data": {"rating": 1},
            },
        ])
        stats = generate_statistics(df)
        # No snapshot column, so RAG analysis should be empty
        assert stats["rag_analysis"] == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
