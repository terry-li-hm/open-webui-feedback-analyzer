#!/usr/bin/env python3
"""
Open WebUI Feedback Analyzer

Analyze feedback data exported from Open WebUI, generating comprehensive
statistics on ratings, models, reasons, temporal patterns, and RAG usage.
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

# Constants
DEFAULT_TIMEZONE = "Asia/Hong_Kong"
RATING_THUMBS_UP = 1
RATING_THUMBS_DOWN = -1
TIMESTAMP_UNIT = "s"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class FeedbackAnalyzerError(Exception):
    """Base exception for feedback analyzer errors."""

    pass


class DataLoadError(FeedbackAnalyzerError):
    """Raised when data cannot be loaded."""

    pass


class DataValidationError(FeedbackAnalyzerError):
    """Raised when data validation fails."""

    pass


def safe_get(d: Any, *keys: str, default: Any = None) -> Any:
    """Safely navigate nested dictionaries.

    Args:
        d: Dictionary or nested structure to navigate
        *keys: Keys to traverse
        default: Default value if key not found

    Returns:
        Value at the nested key path, or default if not found
    """
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d


def delete_keys_from_nested_json(data: Any, key_to_delete: str) -> Any:
    """Recursively remove a key from nested JSON structures.

    Args:
        data: JSON data structure (dict, list, or primitive)
        key_to_delete: Key name to remove from all dicts

    Returns:
        Data structure with specified key removed
    """
    if isinstance(data, dict):
        return {
            k: delete_keys_from_nested_json(v, key_to_delete)
            for k, v in data.items()
            if k != key_to_delete
        }
    elif isinstance(data, list):
        return [delete_keys_from_nested_json(item, key_to_delete) for item in data]
    return data


def load_feedback_data(filepath: Path, encoding: str = "utf-8") -> list[dict]:
    """Load and parse feedback JSON file.

    Args:
        filepath: Path to the JSON file
        encoding: File encoding (default: utf-8)

    Returns:
        List of feedback records

    Raises:
        DataLoadError: If file cannot be read or parsed
    """
    try:
        with open(filepath, "r", encoding=encoding) as f:
            data = json.load(f)
    except FileNotFoundError:
        raise DataLoadError(f"File not found: {filepath}")
    except json.JSONDecodeError as e:
        raise DataLoadError(f"Invalid JSON in {filepath}: {e}")
    except PermissionError:
        raise DataLoadError(f"Permission denied reading {filepath}")
    except UnicodeDecodeError as e:
        raise DataLoadError(
            f"Encoding error in {filepath}: {e}. "
            f"Try specifying --encoding (e.g., --encoding gbk or --encoding utf-8-sig)"
        )

    if not isinstance(data, list):
        raise DataValidationError(f"Expected list of records, got {type(data).__name__}")

    return data


def filter_by_date_range(
    data: list[dict],
    start_date: str,
    end_date: str,
    timezone: str = DEFAULT_TIMEZONE,
) -> pd.DataFrame:
    """Filter feedback records by date range using root-level created_at.

    Args:
        data: List of feedback records
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        timezone: Timezone for date interpretation

    Returns:
        Filtered DataFrame with timezone-aware timestamps

    Raises:
        DataValidationError: If data is empty or missing required fields
    """
    if not data:
        raise DataValidationError("No data to filter")

    df = pd.DataFrame(data)

    if "created_at" not in df.columns:
        raise DataValidationError("Data missing required 'created_at' field")

    df["created_at"] = pd.to_datetime(
        df["created_at"], unit=TIMESTAMP_UNIT, utc=True
    ).dt.tz_convert(timezone)

    start = pd.Timestamp(start_date, tz=timezone)
    end = pd.Timestamp(end_date, tz=timezone) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    mask = df["created_at"].between(start, end)
    return df[mask].reset_index(drop=True)


def export_to_json(df: pd.DataFrame, filepath: Path) -> None:
    """Export DataFrame to JSON, converting timestamps back to Unix seconds.

    Args:
        df: DataFrame to export
        filepath: Output file path

    Raises:
        FeedbackAnalyzerError: If export fails
    """
    try:
        df = df.copy()
        df["created_at"] = df["created_at"].astype("int64") // 10**9
        if "updated_at" in df.columns:
            df["updated_at"] = pd.to_datetime(df["updated_at"], unit=TIMESTAMP_UNIT).astype("int64") // 10**9

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(df.to_dict("records"), f, indent=4, ensure_ascii=False)
    except (OSError, PermissionError) as e:
        raise FeedbackAnalyzerError(f"Failed to export data: {e}")


def _analyze_overview(df: pd.DataFrame) -> dict:
    """Generate overview statistics."""
    stats = {
        "total_records": len(df),
        "date_range": {
            "earliest": df["created_at"].min().isoformat(),
            "latest": df["created_at"].max().isoformat(),
        },
    }

    if "snapshot" in df.columns:
        archived_count = df["snapshot"].apply(
            lambda x: safe_get(x, "archived", default=False)
        ).sum()
        pinned_count = df["snapshot"].apply(
            lambda x: safe_get(x, "pinned", default=False)
        ).sum()
        stats["archived_chats"] = int(archived_count)
        stats["pinned_chats"] = int(pinned_count)

    return stats


def _analyze_ratings(df: pd.DataFrame) -> dict:
    """Generate rating statistics."""
    ratings = df["data"].apply(lambda x: safe_get(x, "rating"))
    stats = {
        "distribution": ratings.value_counts(dropna=False).to_dict(),
        "thumbs_up_rate": (
            (ratings == RATING_THUMBS_UP).sum() / len(ratings) if len(ratings) > 0 else 0
        ),
    }

    # Detailed rating (1-10 scale from details.rating)
    if "details" in df.columns:
        detailed_ratings = df["details"].apply(lambda x: safe_get(x, "rating"))
        valid_detailed = detailed_ratings.dropna()
        if len(valid_detailed) > 0:
            stats["detailed_distribution"] = (
                detailed_ratings.value_counts(dropna=False).sort_index().to_dict()
            )
            stats["detailed_average"] = valid_detailed.mean()

    return stats


def _analyze_reasons(df: pd.DataFrame) -> dict:
    """Generate reason statistics."""
    ratings = df["data"].apply(lambda x: safe_get(x, "rating"))
    reasons = df["data"].apply(lambda x: safe_get(x, "reason"))

    stats = {
        "distribution": reasons.value_counts(dropna=False).to_dict(),
    }

    # Cross-analysis: rating by reason
    rating_reason = pd.DataFrame({"rating": ratings, "reason": reasons})
    stats["thumbs_up_rate_by_reason"] = (
        rating_reason.groupby("reason")["rating"]
        .apply(lambda x: (x == RATING_THUMBS_UP).sum() / len(x) if len(x) > 0 else 0)
        .to_dict()
    )

    return stats


def _analyze_models(df: pd.DataFrame) -> dict:
    """Generate model statistics."""
    ratings = df["data"].apply(lambda x: safe_get(x, "rating"))
    models = df["data"].apply(lambda x: safe_get(x, "model_id"))

    stats = {
        "distribution": models.value_counts(dropna=False).to_dict(),
    }

    # Base model analysis
    if "base_models" in df.columns:
        base_models = df["base_models"].apply(
            lambda x: list(x.values())[0] if isinstance(x, dict) and x else None
        )
        stats["base_model_distribution"] = base_models.value_counts(dropna=False).to_dict()

    # Cross-analysis: rating by model
    rating_model = pd.DataFrame({"rating": ratings, "model": models})
    stats["thumbs_up_rate_by_model"] = (
        rating_model.groupby("model")["rating"]
        .apply(lambda x: (x == RATING_THUMBS_UP).sum() / len(x) if len(x) > 0 else 0)
        .to_dict()
    )

    return stats


def _analyze_temporal(df: pd.DataFrame) -> dict:
    """Generate temporal statistics."""
    return {
        "by_hour": df["created_at"].dt.hour.value_counts().sort_index().to_dict(),
        "by_day_of_week": df["created_at"].dt.day_name().value_counts().to_dict(),
        "by_date": df["created_at"].dt.date.astype(str).value_counts().sort_index().to_dict(),
    }


def _analyze_rag(df: pd.DataFrame, show_progress: bool = False) -> dict:
    """Generate RAG (Retrieval Augmented Generation) statistics."""
    if "snapshot" not in df.columns:
        return {}

    source_counts: list[int] = []
    query_counts: list[int] = []
    action_counter: Counter = Counter()

    iterator = df.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=len(df), desc="Analyzing RAG data")

    for _, row in iterator:
        snapshot = row.get("snapshot", {})
        history = safe_get(snapshot, "history", default={})
        messages = history.get("messages", [])

        # Handle both list and dict formats for messages
        if isinstance(messages, dict):
            messages = list(messages.values())

        for msg in messages:
            if msg.get("role") == "assistant":
                status_history = msg.get("statusHistory", [])
                for status in status_history:
                    action = status.get("action")
                    if action:
                        action_counter[action] += 1
                    if action == "sources_retrieved":
                        source_counts.append(status.get("count", 0))
                    if action == "queries_generated":
                        queries = status.get("queries", [])
                        query_counts.append(len(queries))

    stats: dict[str, Any] = {"action_distribution": dict(action_counter)}

    if source_counts:
        stats["sources_retrieved"] = {
            "total": sum(source_counts),
            "average": sum(source_counts) / len(source_counts),
            "distribution": dict(Counter(source_counts)),
        }

    if query_counts:
        stats["queries_generated"] = {
            "average_per_response": sum(query_counts) / len(query_counts),
            "distribution": dict(Counter(query_counts)),
        }

    return stats


def generate_statistics(df: pd.DataFrame, show_progress: bool = False) -> dict:
    """Generate comprehensive statistics from feedback data.

    Args:
        df: DataFrame containing feedback data
        show_progress: Whether to show progress bar for long operations

    Returns:
        Dictionary containing all statistics
    """
    logger.info("Generating statistics...")

    return {
        "overview": _analyze_overview(df),
        "rating_analysis": _analyze_ratings(df),
        "reason_analysis": _analyze_reasons(df),
        "model_analysis": _analyze_models(df),
        "temporal_analysis": _analyze_temporal(df),
        "rag_analysis": _analyze_rag(df, show_progress=show_progress),
    }


def print_statistics(stats: dict) -> None:
    """Print statistics in a readable format."""
    print("\n" + "=" * 60)
    print("FEEDBACK DATA STATISTICS")
    print("=" * 60)

    overview = stats["overview"]
    print(f"\nOVERVIEW")
    print(f"   Total records: {overview['total_records']}")
    print(f"   Date range: {overview['date_range']['earliest']} to {overview['date_range']['latest']}")
    if "archived_chats" in overview:
        print(f"   Archived chats: {overview['archived_chats']}")
        print(f"   Pinned chats: {overview['pinned_chats']}")

    rating = stats["rating_analysis"]
    print(f"\nRATING ANALYSIS")
    print(f"   Distribution: {rating['distribution']}")
    print(f"   Thumbs up rate: {rating['thumbs_up_rate']:.1%}")
    if "detailed_average" in rating:
        print(f"   Detailed rating average: {rating['detailed_average']:.1f}/10")

    reason = stats["reason_analysis"]
    print(f"\nREASON ANALYSIS")
    for r, count in reason["distribution"].items():
        rate = reason["thumbs_up_rate_by_reason"].get(r, 0)
        print(f"   {r}: {count} ({rate:.1%} positive)")

    model = stats["model_analysis"]
    print(f"\nMODEL ANALYSIS")
    for m, count in model["distribution"].items():
        rate = model["thumbs_up_rate_by_model"].get(m, 0)
        print(f"   {m}: {count} ({rate:.1%} positive)")

    rag = stats["rag_analysis"]
    print(f"\nRAG ANALYSIS")
    if rag.get("action_distribution"):
        print(f"   Actions: {rag['action_distribution']}")
    if rag.get("sources_retrieved"):
        print(f"   Avg sources retrieved: {rag['sources_retrieved']['average']:.1f}")
    if rag.get("queries_generated"):
        print(f"   Avg queries generated: {rag['queries_generated']['average_per_response']:.1f}")

    temporal = stats["temporal_analysis"]
    print(f"\nTEMPORAL ANALYSIS")
    print(f"   By date: {temporal['by_date']}")


def make_serializable(obj: Any) -> Any:
    """Convert objects to JSON-serializable format.

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(i) for i in obj]
    elif isinstance(obj, Counter):
        return dict(obj)
    elif isinstance(obj, float):
        return round(obj, 4)
    return obj


def export_statistics(stats: dict, filepath: Path) -> None:
    """Export statistics to JSON file.

    Args:
        stats: Statistics dictionary
        filepath: Output file path

    Raises:
        FeedbackAnalyzerError: If export fails
    """
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(make_serializable(stats), f, indent=4, ensure_ascii=False)
    except (OSError, PermissionError) as e:
        raise FeedbackAnalyzerError(f"Failed to export statistics: {e}")


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        args: Command line arguments (defaults to sys.argv)

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Analyze feedback data exported from Open WebUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s feedback.json --start 2025-12-01 --end 2025-12-09
  %(prog)s feedback.json -s 2025-12-01 -e 2025-12-09 --timezone UTC
  %(prog)s feedback.json -s 2025-12-01 -e 2025-12-09 -o ./output
        """,
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Path to feedback JSON file",
    )
    parser.add_argument(
        "-s", "--start",
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "-e", "--end",
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("."),
        help="Output directory (default: current directory)",
    )
    parser.add_argument(
        "--timezone",
        default=DEFAULT_TIMEZONE,
        help=f"Timezone for date interpretation (default: {DEFAULT_TIMEZONE})",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Input file encoding (default: utf-8). Try 'gbk' for Chinese or 'utf-8-sig' for BOM",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip exporting filtered data and statistics",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bar for long operations",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output (only show errors)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    """Main entry point.

    Args:
        args: Command line arguments (for testing)

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parsed = parse_args(args)

    # Configure logging level
    if parsed.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif parsed.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Load data
        logger.info(f"Loading data from {parsed.input} (encoding: {parsed.encoding})...")
        data = load_feedback_data(parsed.input, encoding=parsed.encoding)
        logger.info(f"Loaded {len(data)} total records")

        # Remove sources key
        logger.debug("Removing 'sources' from nested structures...")
        filtered_data = delete_keys_from_nested_json(data, "sources")

        # Filter by date
        logger.info(f"Filtering to date range {parsed.start} to {parsed.end}...")
        df = filter_by_date_range(
            filtered_data,
            parsed.start,
            parsed.end,
            timezone=parsed.timezone,
        )
        logger.info(f"Filtered to {len(df)} records")

        if len(df) == 0:
            logger.warning("No records found in the specified date range")
            return 0

        # Generate statistics
        stats = generate_statistics(df, show_progress=parsed.progress)

        if not parsed.quiet:
            print_statistics(stats)

        # Export if requested
        if not parsed.no_export:
            parsed.output_dir.mkdir(parents=True, exist_ok=True)

            data_file = parsed.output_dir / f"{parsed.start}-{parsed.end}-data.json"
            logger.info(f"Exporting data to {data_file}...")
            export_to_json(df, data_file)

            stats_file = parsed.output_dir / f"{parsed.start}-{parsed.end}-statistics.json"
            logger.info(f"Exporting statistics to {stats_file}...")
            export_statistics(stats, stats_file)

        logger.info("Done!")
        return 0

    except FeedbackAnalyzerError as e:
        logger.error(str(e))
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
