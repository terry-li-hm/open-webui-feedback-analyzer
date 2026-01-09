#!/usr/bin/env python3
"""
Open WebUI Feedback Analyzer

Analyze feedback data exported from Open WebUI, generating comprehensive
statistics on ratings, models, reasons, temporal patterns, and RAG usage.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

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


def load_feedback_data(filepath: Path) -> list[dict]:
    """Load and parse feedback JSON file.

    Args:
        filepath: Path to the JSON file

    Returns:
        List of feedback records

    Raises:
        DataLoadError: If file cannot be read or parsed
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise DataLoadError(f"File not found: {filepath}")
    except json.JSONDecodeError as e:
        raise DataLoadError(f"Invalid JSON in {filepath}: {e}")
    except PermissionError:
        raise DataLoadError(f"Permission denied reading {filepath}")

    if not isinstance(data, list):
        raise DataValidationError(f"Expected list of records, got {type(data).__name__}")

    return data


def filter_by_date_range(
    data: list[dict],
    start_date: str | None = None,
    end_date: str | None = None,
    timezone: str = DEFAULT_TIMEZONE,
) -> pd.DataFrame:
    """Filter feedback records by date range using root-level created_at.

    Args:
        data: List of feedback records
        start_date: Start date (YYYY-MM-DD), or None for no lower bound
        end_date: End date (YYYY-MM-DD), or None for no upper bound
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

    # If no dates specified, return all data
    if start_date is None and end_date is None:
        return df.reset_index(drop=True)

    # Apply date filters
    mask = pd.Series([True] * len(df))
    if start_date is not None:
        start = pd.Timestamp(start_date, tz=timezone)
        mask &= df["created_at"] >= start
    if end_date is not None:
        end = pd.Timestamp(end_date, tz=timezone) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        mask &= df["created_at"] <= end

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
    # Weekly stats: group by year-week
    weekly = df["created_at"].dt.to_period("W").astype(str).value_counts().sort_index().to_dict()

    return {
        "by_hour": df["created_at"].dt.hour.value_counts().sort_index().to_dict(),
        "by_day_of_week": df["created_at"].dt.day_name().value_counts().to_dict(),
        "by_week": weekly,
        "by_date": df["created_at"].dt.date.astype(str).value_counts().sort_index().to_dict(),
    }


def _analyze_rag(df: pd.DataFrame) -> dict:
    """Generate RAG (Retrieval Augmented Generation) statistics."""
    if "snapshot" not in df.columns:
        return {}

    source_counts: list[int] = []
    query_counts: list[int] = []
    action_counter: Counter = Counter()

    for _, row in df.iterrows():
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


def _analyze_comments(df: pd.DataFrame) -> dict:
    """Analyze user comments, especially on negative feedback."""
    comments = df["data"].apply(lambda x: safe_get(x, "comment"))
    ratings = df["data"].apply(lambda x: safe_get(x, "rating"))

    # Filter non-empty comments
    has_comment = comments.apply(lambda x: bool(x and str(x).strip()))
    comments_df = pd.DataFrame({
        "comment": comments,
        "rating": ratings,
        "has_comment": has_comment
    })

    stats: dict[str, Any] = {
        "total_with_comments": int(has_comment.sum()),
        "comment_rate": has_comment.sum() / len(df) if len(df) > 0 else 0,
    }

    # Comments on negative feedback (most useful for improvement)
    negative_with_comments = comments_df[
        (comments_df["rating"] == RATING_THUMBS_DOWN) & (comments_df["has_comment"])
    ]
    if len(negative_with_comments) > 0:
        stats["negative_feedback_comments"] = negative_with_comments["comment"].tolist()

    return stats


def _analyze_feedback_position(df: pd.DataFrame) -> dict:
    """Analyze where in conversations feedback typically occurs."""
    if "meta" not in df.columns:
        return {}

    positions = df["meta"].apply(lambda x: safe_get(x, "message_index"))
    valid_positions = positions.dropna()

    if len(valid_positions) == 0:
        return {}

    ratings = df["data"].apply(lambda x: safe_get(x, "rating"))
    position_df = pd.DataFrame({"position": positions, "rating": ratings}).dropna()

    stats = {
        "average_position": valid_positions.mean(),
        "position_distribution": valid_positions.value_counts().sort_index().to_dict(),
    }

    # Average position by rating
    if len(position_df) > 0:
        avg_by_rating = position_df.groupby("rating")["position"].mean().to_dict()
        stats["average_position_by_rating"] = {
            "thumbs_up": avg_by_rating.get(RATING_THUMBS_UP, 0),
            "thumbs_down": avg_by_rating.get(RATING_THUMBS_DOWN, 0),
        }

    return stats


def _analyze_detailed_ratings(df: pd.DataFrame) -> dict:
    """Analyze detailed 1-10 ratings distribution."""
    if "details" not in df.columns:
        return {}

    detailed = df["details"].apply(lambda x: safe_get(x, "rating"))
    valid = detailed.dropna()

    if len(valid) == 0:
        return {}

    return {
        "count": len(valid),
        "average": valid.mean(),
        "median": valid.median(),
        "distribution": valid.value_counts().sort_index().to_dict(),
    }


def _analyze_tags(df: pd.DataFrame) -> dict:
    """Analyze tag usage in feedback."""
    tags_series = df["data"].apply(lambda x: safe_get(x, "tags", default=[]))

    all_tags: list[str] = []
    for tags in tags_series:
        if isinstance(tags, list):
            all_tags.extend(tags)

    if not all_tags:
        return {}

    return {
        "total_tags": len(all_tags),
        "unique_tags": len(set(all_tags)),
        "distribution": dict(Counter(all_tags)),
    }


def _analyze_files(df: pd.DataFrame) -> dict:
    """Analyze file attachment correlation with ratings."""
    if "snapshot" not in df.columns:
        return {}

    def has_files(snapshot):
        files = safe_get(snapshot, "files", default=[])
        return bool(files and len(files) > 0)

    file_attached = df["snapshot"].apply(has_files)
    ratings = df["data"].apply(lambda x: safe_get(x, "rating"))

    stats = {
        "conversations_with_files": int(file_attached.sum()),
        "file_attachment_rate": file_attached.sum() / len(df) if len(df) > 0 else 0,
    }

    # Rating comparison: with files vs without
    with_files = ratings[file_attached]
    without_files = ratings[~file_attached]

    if len(with_files) > 0:
        stats["thumbs_up_rate_with_files"] = (with_files == RATING_THUMBS_UP).sum() / len(with_files)
    if len(without_files) > 0:
        stats["thumbs_up_rate_without_files"] = (without_files == RATING_THUMBS_UP).sum() / len(without_files)

    return stats


def generate_statistics(df: pd.DataFrame) -> dict:
    """Generate comprehensive statistics from feedback data.

    Args:
        df: DataFrame containing feedback data

    Returns:
        Dictionary containing all statistics
    """
    logger.info("Generating statistics...")

    return {
        "overview": _analyze_overview(df),
        "rating_analysis": _analyze_ratings(df),
        "detailed_ratings": _analyze_detailed_ratings(df),
        "reason_analysis": _analyze_reasons(df),
        "model_analysis": _analyze_models(df),
        "temporal_analysis": _analyze_temporal(df),
        "rag_analysis": _analyze_rag(df),
        "comment_analysis": _analyze_comments(df),
        "feedback_position": _analyze_feedback_position(df),
        "tag_analysis": _analyze_tags(df),
        "file_analysis": _analyze_files(df),
    }


def _format_bar(value: float, max_value: float, width: int = 10) -> str:
    """Create a visual bar representation."""
    if max_value == 0:
        return " " * width
    filled = int((value / max_value) * width)
    return "#" * filled + "-" * (width - filled)


def _format_date_short(iso_date: str, include_year: bool = True) -> str:
    """Convert ISO date to short format (YYYY-MM-DD or MM-DD)."""
    try:
        # Handle both ISO format and date strings
        if "T" in iso_date:
            date_part = iso_date.split("T")[0]
        else:
            date_part = iso_date
        parts = date_part.split("-")
        if include_year:
            return f"{parts[0]}-{parts[1]}-{parts[2]}"
        return f"{parts[1]}-{parts[2]}"
    except (IndexError, AttributeError):
        return str(iso_date)[:10]


def print_statistics(stats: dict) -> None:
    """Print statistics in a readable format."""
    width = 60

    print("\n" + "=" * width)
    print("FEEDBACK DATA STATISTICS".center(width))
    print("=" * width)

    # Overview
    overview = stats["overview"]
    start_date = _format_date_short(overview["date_range"]["earliest"])
    end_date = _format_date_short(overview["date_range"]["latest"])

    print(f"\nOVERVIEW")
    print(f"  {'Total records:':<20} {overview['total_records']:>6}")
    print(f"  {'Date range:':<20} {start_date} to {end_date}")
    if overview.get("archived_chats", 0) > 0:
        print(f"  {'Archived chats:':<20} {overview['archived_chats']:>6}")
    if overview.get("pinned_chats", 0) > 0:
        print(f"  {'Pinned chats:':<20} {overview['pinned_chats']:>6}")

    # Rating Analysis
    rating = stats["rating_analysis"]
    dist = rating["distribution"]
    total = sum(v for v in dist.values() if v is not None)
    thumbs_up = dist.get(1, 0) or 0
    thumbs_down = dist.get(-1, 0) or 0

    print(f"\nRATING ANALYSIS")
    if total > 0:
        up_pct = thumbs_up / total
        down_pct = thumbs_down / total
        print(f"  {'Thumbs Up:':<20} {thumbs_up:>6}  ({up_pct:>5.1%})  {_format_bar(thumbs_up, total)}")
        print(f"  {'Thumbs Down:':<20} {thumbs_down:>6}  ({down_pct:>5.1%})  {_format_bar(thumbs_down, total)}")
    if "detailed_average" in rating:
        avg = rating["detailed_average"]
        print(f"  {'Detailed avg:':<20} {avg:>5.1f}/10  {_format_bar(avg, 10)}")

    # Reason Analysis
    reason = stats["reason_analysis"]
    reason_dist = reason["distribution"]
    if reason_dist:
        print(f"\nREASON ANALYSIS")
        max_count = max((v for v in reason_dist.values() if v), default=1)
        sorted_reasons = sorted(reason_dist.items(), key=lambda x: x[1] or 0, reverse=True)
        for r, count in sorted_reasons:
            if r is None or r == "":
                display_r = "(no reason)"
            else:
                display_r = r
            count = count or 0
            rate = reason["thumbs_up_rate_by_reason"].get(r, 0)
            label = str(display_r)[:25]
            print(f"  {label:<27} {count:>5}  ({rate:>5.1%} +)  {_format_bar(count, max_count)}")

    # Model Analysis (only show if multiple models)
    model = stats["model_analysis"]
    model_dist = model["distribution"]
    if model_dist and len(model_dist) > 1:
        print(f"\nMODEL ANALYSIS")
        max_count = max((v for v in model_dist.values() if v), default=1)
        sorted_models = sorted(model_dist.items(), key=lambda x: x[1] or 0, reverse=True)
        for m, count in sorted_models:
            if m is None or m == "":
                display_m = "(unknown)"
            else:
                display_m = m
            count = count or 0
            rate = model["thumbs_up_rate_by_model"].get(m, 0)
            label = str(display_m)[:35]
            print(f"  {label:<37} {count:>5}  ({rate:>5.1%} +)  {_format_bar(count, max_count)}")

    # RAG Analysis
    rag = stats["rag_analysis"]
    if rag.get("action_distribution") or rag.get("sources_retrieved") or rag.get("queries_generated"):
        print(f"\nRAG ANALYSIS")
        if rag.get("sources_retrieved"):
            src = rag["sources_retrieved"]
            print(f"  {'Sources retrieved:':<20} {src['total']:>6} total, {src['average']:.1f} avg")
        if rag.get("queries_generated"):
            qry = rag["queries_generated"]
            print(f"  {'Queries generated:':<20} {qry['average_per_response']:.1f} avg per response")
        if rag.get("action_distribution"):
            print(f"  {'Actions:':<20}")
            actions = rag["action_distribution"]
            max_count = max(actions.values(), default=1)
            sorted_actions = sorted(actions.items(), key=lambda x: x[1], reverse=True)
            for action, count in sorted_actions:
                label = str(action)[:18]
                print(f"    {label:<18} {count:>5}  {_format_bar(count, max_count)}")

    # Temporal Analysis
    temporal = stats["temporal_analysis"]

    # Weekly
    by_week = temporal.get("by_week", {})
    if by_week:
        print(f"\nWEEKLY ACTIVITY")
        max_count = max(by_week.values(), default=1)
        for week, count in sorted(by_week.items()):
            print(f"  {week:<15} {count:>5}  {_format_bar(count, max_count)}")

    # Daily
    by_date = temporal.get("by_date", {})
    if by_date:
        print(f"\nDAILY ACTIVITY")
        max_count = max(by_date.values(), default=1)
        for date, count in sorted(by_date.items()):
            short_date = _format_date_short(date)
            print(f"  {short_date:<15} {count:>5}  {_format_bar(count, max_count)}")

    # Day of week
    by_dow = temporal.get("by_day_of_week", {})
    if by_dow:
        print(f"\nBY DAY OF WEEK")
        max_count = max(by_dow.values(), default=1)
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        for day in day_order:
            if day in by_dow:
                count = by_dow[day]
                print(f"  {day:<12} {count:>5}  {_format_bar(count, max_count)}")

    # Detailed Ratings (1-10)
    detailed = stats.get("detailed_ratings", {})
    if detailed.get("distribution"):
        print(f"\nDETAILED RATINGS (1-10)")
        print(f"  {'Average:':<20} {detailed['average']:>5.1f}")
        print(f"  {'Median:':<20} {detailed['median']:>5.0f}")
        dist = detailed["distribution"]
        max_count = max(dist.values(), default=1)
        for rating in sorted(dist.keys()):
            count = dist[rating]
            print(f"  {int(rating):<20} {count:>5}  {_format_bar(count, max_count)}")

    # Feedback Position (only show if there's variance)
    position = stats.get("feedback_position", {})
    pos_dist = position.get("position_distribution", {})
    if position.get("average_position") is not None and len(pos_dist) > 1:
        print(f"\nFEEDBACK POSITION")
        print(f"  {'Avg message index:':<20} {position['average_position']:>5.1f}")
        if "average_position_by_rating" in position:
            pos_by_rating = position["average_position_by_rating"]
            print(f"  {'Avg for thumbs up:':<20} {pos_by_rating.get('thumbs_up', 0):>5.1f}")
            print(f"  {'Avg for thumbs down:':<20} {pos_by_rating.get('thumbs_down', 0):>5.1f}")

    # File Analysis
    files = stats.get("file_analysis", {})
    if files.get("conversations_with_files", 0) > 0 or files.get("file_attachment_rate", 0) > 0:
        print(f"\nFILE ATTACHMENTS")
        print(f"  {'With files:':<20} {files.get('conversations_with_files', 0):>5}  ({files.get('file_attachment_rate', 0):>5.1%})")
        if "thumbs_up_rate_with_files" in files:
            print(f"  {'+ rate (with files):':<20} {files['thumbs_up_rate_with_files']:>5.1%}")
        if "thumbs_up_rate_without_files" in files:
            print(f"  {'+ rate (no files):':<20} {files['thumbs_up_rate_without_files']:>5.1%}")

    # Comment Analysis
    comments = stats.get("comment_analysis", {})
    if comments.get("total_with_comments", 0) > 0:
        print(f"\nCOMMENT ANALYSIS")
        print(f"  {'With comments:':<20} {comments['total_with_comments']:>5}  ({comments.get('comment_rate', 0):>5.1%})")
        neg_comments = comments.get("negative_feedback_comments", [])
        if neg_comments:
            print(f"  Negative feedback comments ({len(neg_comments)}):")
            for c in neg_comments[:50]:  # Show up to 50
                comment_preview = str(c)[:150] + "..." if len(str(c)) > 150 else str(c)
                print(f"    - {comment_preview}")
            if len(neg_comments) > 50:
                print(f"    ... and {len(neg_comments) - 50} more")

    # Tag Analysis
    tags = stats.get("tag_analysis", {})
    if tags.get("total_tags", 0) > 0:
        print(f"\nTAG ANALYSIS")
        print(f"  {'Total tags:':<20} {tags['total_tags']:>5}")
        print(f"  {'Unique tags:':<20} {tags['unique_tags']:>5}")
        if tags.get("distribution"):
            dist = tags["distribution"]
            max_count = max(dist.values(), default=1)
            sorted_tags = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:10]
            for tag, count in sorted_tags:
                label = str(tag)[:18]
                print(f"  {label:<20} {count:>5}  {_format_bar(count, max_count)}")

    print("\n" + "=" * width)


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
  %(prog)s feedback.json                                    # All data
  %(prog)s feedback.json -s 2025-07-01                      # From date to now
  %(prog)s feedback.json -s 2025-12-01 -e 2025-12-09        # Date range
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
        default=None,
        help="Start date (YYYY-MM-DD). If omitted, includes all past data",
    )
    parser.add_argument(
        "-e", "--end",
        default=None,
        help="End date (YYYY-MM-DD). If omitted, includes up to latest data",
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
        "--no-export",
        action="store_true",
        help="Skip exporting filtered data and statistics",
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
        logger.info(f"Loading data from {parsed.input}...")
        data = load_feedback_data(parsed.input)
        logger.info(f"Loaded {len(data)} total records")

        # Remove sources key
        logger.debug("Removing 'sources' from nested structures...")
        filtered_data = delete_keys_from_nested_json(data, "sources")

        # Filter by date
        if parsed.start or parsed.end:
            date_desc = f"{parsed.start or 'beginning'} to {parsed.end or 'latest'}"
        else:
            date_desc = "all data"
        logger.info(f"Filtering to {date_desc}...")

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
        stats = generate_statistics(df)

        if not parsed.quiet:
            print_statistics(stats)

        # Export if requested
        if not parsed.no_export:
            parsed.output_dir.mkdir(parents=True, exist_ok=True)

            # Use actual date range from data for filename if not specified
            actual_start = parsed.start or df["created_at"].min().strftime("%Y-%m-%d")
            actual_end = parsed.end or df["created_at"].max().strftime("%Y-%m-%d")

            data_file = parsed.output_dir / f"{actual_start}-{actual_end}-data.json"
            logger.info(f"Exporting data to {data_file}...")
            export_to_json(df, data_file)

            stats_file = parsed.output_dir / f"{actual_start}-{actual_end}-statistics.json"
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
