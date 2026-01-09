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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Constants
DEFAULT_TIMEZONE = "Asia/Hong_Kong"
RATING_THUMBS_UP = 1
RATING_THUMBS_DOWN = -1
TIMESTAMP_UNIT = "s"

# Display limits
DAILY_DISPLAY_LIMIT = 14        # Show last N days
WEEKLY_DISPLAY_LIMIT = 8        # Show last N weeks
COMMENT_DISPLAY_LIMIT = 50      # Show up to N comments
COMMENT_TRUNCATE_LENGTH = 150   # Truncate comments at N chars
REASON_LABEL_WIDTH = 25         # Width for reason labels
MODEL_LABEL_WIDTH = 35          # Width for model labels
REPORT_WIDTH = 60               # Width of report separator lines
BAR_WIDTH = 10                  # Width of horizontal bars
DOW_BAR_HEIGHT = 5              # Height of day-of-week vertical bars

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
        # Convert timezone-aware datetime to Unix timestamp (seconds)
        df["created_at"] = df["created_at"].apply(lambda x: int(x.timestamp()))
        if "updated_at" in df.columns:
            df["updated_at"] = pd.to_datetime(df["updated_at"], unit=TIMESTAMP_UNIT)
            df["updated_at"] = df["updated_at"].apply(lambda x: int(x.timestamp()) if pd.notna(x) else None)

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
    # Monthly stats: group by year-month
    df_copy = df.copy()
    df_copy["month_period"] = df_copy["created_at"].dt.to_period("M").astype(str)
    monthly = df_copy["month_period"].value_counts().sort_index().to_dict()

    # Monthly accuracy
    ratings = df_copy["data"].apply(lambda x: safe_get(x, "rating"))
    df_copy["rating"] = ratings
    monthly_accuracy = df_copy.groupby("month_period").apply(
        lambda x: (x["rating"] == RATING_THUMBS_UP).sum() / len(x) if len(x) > 0 else 0
    ).sort_index().to_dict()

    # Weekly stats: group by year-week
    weekly = df["created_at"].dt.to_period("W").astype(str).value_counts().sort_index().to_dict()

    # Daily stats
    daily = df["created_at"].dt.date.astype(str).value_counts().sort_index().to_dict()

    return {
        "by_hour": df["created_at"].dt.hour.value_counts().sort_index().to_dict(),
        "by_day_of_week": df["created_at"].dt.day_name().value_counts().to_dict(),
        "by_month": monthly,
        "by_month_accuracy": monthly_accuracy,
        "by_week": weekly,
        "by_date": daily,
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


def _analyze_trends(df: pd.DataFrame) -> dict:
    """Analyze week-on-week and month-on-month trends for volume and accuracy."""
    if len(df) == 0:
        return {}

    # Add period columns
    df = df.copy()
    df["week"] = df["created_at"].dt.to_period("W")
    df["month"] = df["created_at"].dt.to_period("M")

    ratings = df["data"].apply(lambda x: safe_get(x, "rating"))

    stats = {}

    # Week-on-Week analysis
    weeks = sorted(df["week"].unique())
    if len(weeks) >= 2:
        current_week = weeks[-1]
        prev_week = weeks[-2]

        current_week_df = df[df["week"] == current_week]
        prev_week_df = df[df["week"] == prev_week]

        current_week_ratings = current_week_df["data"].apply(lambda x: safe_get(x, "rating"))
        prev_week_ratings = prev_week_df["data"].apply(lambda x: safe_get(x, "rating"))

        # Volume
        current_week_vol = len(current_week_df)
        prev_week_vol = len(prev_week_df)
        wow_vol_change = (current_week_vol - prev_week_vol) / prev_week_vol if prev_week_vol > 0 else 0

        # Accuracy (thumbs up rate)
        current_week_acc = (current_week_ratings == RATING_THUMBS_UP).sum() / len(current_week_ratings) if len(current_week_ratings) > 0 else 0
        prev_week_acc = (prev_week_ratings == RATING_THUMBS_UP).sum() / len(prev_week_ratings) if len(prev_week_ratings) > 0 else 0
        wow_acc_change = current_week_acc - prev_week_acc  # Percentage point change

        stats["week_on_week"] = {
            "current_week": str(current_week),
            "previous_week": str(prev_week),
            "current_volume": current_week_vol,
            "previous_volume": prev_week_vol,
            "volume_change": wow_vol_change,
            "current_accuracy": current_week_acc,
            "previous_accuracy": prev_week_acc,
            "accuracy_change": wow_acc_change,
        }

    # Month-on-Month analysis
    months = sorted(df["month"].unique())
    if len(months) >= 2:
        current_month = months[-1]
        prev_month = months[-2]

        current_month_df = df[df["month"] == current_month]
        prev_month_df = df[df["month"] == prev_month]

        current_month_ratings = current_month_df["data"].apply(lambda x: safe_get(x, "rating"))
        prev_month_ratings = prev_month_df["data"].apply(lambda x: safe_get(x, "rating"))

        # Volume
        current_month_vol = len(current_month_df)
        prev_month_vol = len(prev_month_df)
        mom_vol_change = (current_month_vol - prev_month_vol) / prev_month_vol if prev_month_vol > 0 else 0

        # Accuracy (thumbs up rate)
        current_month_acc = (current_month_ratings == RATING_THUMBS_UP).sum() / len(current_month_ratings) if len(current_month_ratings) > 0 else 0
        prev_month_acc = (prev_month_ratings == RATING_THUMBS_UP).sum() / len(prev_month_ratings) if len(prev_month_ratings) > 0 else 0
        mom_acc_change = current_month_acc - prev_month_acc  # Percentage point change

        stats["month_on_month"] = {
            "current_month": str(current_month),
            "previous_month": str(prev_month),
            "current_volume": current_month_vol,
            "previous_volume": prev_month_vol,
            "volume_change": mom_vol_change,
            "current_accuracy": current_month_acc,
            "previous_accuracy": prev_month_acc,
            "accuracy_change": mom_acc_change,
        }

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
        "trend_analysis": _analyze_trends(df),
        "rag_analysis": _analyze_rag(df),
        "comment_analysis": _analyze_comments(df),
        "feedback_position": _analyze_feedback_position(df),
        "tag_analysis": _analyze_tags(df),
        "file_analysis": _analyze_files(df),
    }


def _format_bar(value: float, max_value: float, width: int = BAR_WIDTH) -> str:
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
    width = REPORT_WIDTH

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

    # Trend Analysis (WoW/MoM)
    trends = stats.get("trend_analysis", {})
    if trends:
        print(f"\nTREND ANALYSIS")

        # Week-on-Week
        wow = trends.get("week_on_week", {})
        if wow:
            vol_change = wow["volume_change"]
            acc_change = wow["accuracy_change"]
            vol_arrow = "↑" if vol_change > 0 else "↓" if vol_change < 0 else "→"
            acc_arrow = "↑" if acc_change > 0 else "↓" if acc_change < 0 else "→"
            print(f"  Week-on-Week ({wow['previous_week']} → {wow['current_week']})")
            print(f"    {'Volume:':<14} {wow['current_volume']:>5} ({vol_arrow}{abs(vol_change):>5.1%} vs {wow['previous_volume']})")
            print(f"    {'Accuracy:':<14} {wow['current_accuracy']:>5.1%} ({acc_arrow}{abs(acc_change)*100:>4.1f}pp vs {wow['previous_accuracy']:.1%})")

        # Month-on-Month
        mom = trends.get("month_on_month", {})
        if mom:
            vol_change = mom["volume_change"]
            acc_change = mom["accuracy_change"]
            vol_arrow = "↑" if vol_change > 0 else "↓" if vol_change < 0 else "→"
            acc_arrow = "↑" if acc_change > 0 else "↓" if acc_change < 0 else "→"
            print(f"  Month-on-Month ({mom['previous_month']} → {mom['current_month']})")
            print(f"    {'Volume:':<14} {mom['current_volume']:>5} ({vol_arrow}{abs(vol_change):>5.1%} vs {mom['previous_volume']})")
            print(f"    {'Accuracy:':<14} {mom['current_accuracy']:>5.1%} ({acc_arrow}{abs(acc_change)*100:>4.1f}pp vs {mom['previous_accuracy']:.1%})")

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
            label = str(display_r)[:REASON_LABEL_WIDTH]
            print(f"  {label:<{REASON_LABEL_WIDTH + 2}} {count:>5}  ({rate:>5.1%} +)  {_format_bar(count, max_count)}")

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
            label = str(display_m)[:MODEL_LABEL_WIDTH]
            print(f"  {label:<{MODEL_LABEL_WIDTH + 2}} {count:>5}  ({rate:>5.1%} +)  {_format_bar(count, max_count)}")

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

    # Monthly trend with change indicators
    by_month = temporal.get("by_month", {})
    by_month_acc = temporal.get("by_month_accuracy", {})
    if by_month and len(by_month) > 1:
        print(f"\nMONTHLY TREND")
        months = sorted(by_month.keys())[-12:]  # Last 12 months

        # Header
        print(f"  {'Month':<10} {'Volume':>7} {'Chg':>7}   {'Accuracy':<12} {'Chg':>6}")
        print(f"  {'-'*10} {'-'*7} {'-'*7}   {'-'*12} {'-'*6}")

        prev_vol = None
        prev_acc = None
        for m in months:
            vol = by_month[m]
            acc = by_month_acc.get(m, 0)

            # Volume change
            if prev_vol is None:
                vol_chg = "     -"
            else:
                pct_chg = (vol - prev_vol) / prev_vol if prev_vol > 0 else 0
                arrow = "↑" if pct_chg > 0 else "↓" if pct_chg < 0 else "→"
                vol_chg = f"{arrow}{abs(pct_chg):>5.0%}"

            # Accuracy change (percentage points)
            if prev_acc is None:
                acc_chg = "    -"
            else:
                pp_chg = (acc - prev_acc) * 100  # Convert to percentage points
                arrow = "↑" if pp_chg > 0 else "↓" if pp_chg < 0 else "→"
                acc_chg = f"{arrow}{abs(pp_chg):>4.1f}pp"

            # Accuracy bar (10 chars wide, showing 0-100%)
            bar_filled = int(acc * 10)
            acc_bar = "█" * bar_filled + "░" * (10 - bar_filled)

            print(f"  {m:<10} {vol:>7} {vol_chg}   {acc_bar} {acc:>5.1%} {acc_chg}")

            prev_vol = vol
            prev_acc = acc

    # Weekly (last N weeks)
    by_week = temporal.get("by_week", {})
    if by_week:
        sorted_weeks = sorted(by_week.items())
        recent_weeks = sorted_weeks[-WEEKLY_DISPLAY_LIMIT:]
        if recent_weeks:
            total_weeks = len(sorted_weeks)
            header = f"WEEKLY ACTIVITY (last {len(recent_weeks)}"
            if total_weeks > WEEKLY_DISPLAY_LIMIT:
                header += f" of {total_weeks}"
            header += " weeks)"
            print(f"\n{header}")
            max_count = max(c for _, c in recent_weeks)
            for week, count in recent_weeks:
                print(f"  {week:<15} {count:>5}  {_format_bar(count, max_count)}")

    # Daily (last N days)
    by_date = temporal.get("by_date", {})
    if by_date:
        sorted_dates = sorted(by_date.items())
        recent_dates = sorted_dates[-DAILY_DISPLAY_LIMIT:]
        if recent_dates:
            total_days = len(sorted_dates)
            header = f"DAILY ACTIVITY (last {len(recent_dates)}"
            if total_days > DAILY_DISPLAY_LIMIT:
                header += f" of {total_days}"
            header += " days)"
            print(f"\n{header}")
            max_count = max(c for _, c in recent_dates)
            for date, count in recent_dates:
                short_date = _format_date_short(date)
                print(f"  {short_date:<15} {count:>5}  {_format_bar(count, max_count)}")

    # Day of week (horizontal display)
    by_dow = temporal.get("by_day_of_week", {})
    if by_dow:
        print(f"\nBY DAY OF WEEK")
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_abbrev = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        counts = [by_dow.get(day, 0) for day in day_order]
        max_count = max(counts) if counts else 1

        # Header row (day abbreviations)
        print("  " + "".join(f"{d:>6}" for d in day_abbrev))
        # Count row
        print("  " + "".join(f"{c:>6}" for c in counts))
        # Bar row
        for row in range(DOW_BAR_HEIGHT, 0, -1):
            threshold = (row / DOW_BAR_HEIGHT) * max_count
            bars = "".join(
                f"{'  ##  ' if c >= threshold else '      '}"
                for c in counts
            )
            print("  " + bars)

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
            for c in neg_comments[:COMMENT_DISPLAY_LIMIT]:
                # Replace newlines with visible indicator for compact display
                comment_text = str(c).replace("\r\n", " ↵ ").replace("\n", " ↵ ").replace("\r", " ↵ ")
                comment_preview = comment_text[:COMMENT_TRUNCATE_LENGTH] + "..." if len(comment_text) > COMMENT_TRUNCATE_LENGTH else comment_text
                print(f"    - {comment_preview}")
            if len(neg_comments) > COMMENT_DISPLAY_LIMIT:
                print(f"    ... and {len(neg_comments) - COMMENT_DISPLAY_LIMIT} more")

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


def generate_html_report(stats: dict, df: pd.DataFrame, filepath: Path) -> None:
    """Generate interactive HTML report with charts for stakeholders.

    Args:
        stats: Statistics dictionary from generate_statistics
        df: DataFrame with the filtered feedback data
        filepath: Output HTML file path
    """
    # Color scheme
    colors = {
        "primary": "#3498db",
        "success": "#27ae60",
        "danger": "#e74c3c",
        "warning": "#f39c12",
        "info": "#17a2b8",
        "light": "#f8f9fa",
        "dark": "#343a40",
    }

    # Extract data
    overview = stats.get("overview", {})
    rating = stats.get("rating_analysis", {})
    trends = stats.get("trend_analysis", {})
    temporal = stats.get("temporal_analysis", {})
    reason = stats.get("reason_analysis", {})
    model = stats.get("model_analysis", {})

    # Calculate KPIs
    total_records = overview.get("total_records", 0)
    dist = rating.get("distribution", {})
    thumbs_up = dist.get(1, 0) or 0
    thumbs_down = dist.get(-1, 0) or 0
    accuracy = thumbs_up / (thumbs_up + thumbs_down) if (thumbs_up + thumbs_down) > 0 else 0

    # WoW/MoM changes
    wow = trends.get("week_on_week", {})
    mom = trends.get("month_on_month", {})

    # Create figures list
    figures_html = []

    # --- Figure 1: Volume and Accuracy Over Time (Daily) ---
    by_date = temporal.get("by_date", {})
    if by_date:
        dates = list(by_date.keys())[-30:]  # Last 30 days
        volumes = [by_date[d] for d in dates]

        # Calculate daily accuracy
        df_copy = df.copy()
        df_copy["date_str"] = df_copy["created_at"].dt.date.astype(str)
        daily_acc = df_copy.groupby("date_str").apply(
            lambda x: (x["data"].apply(lambda d: safe_get(d, "rating")) == RATING_THUMBS_UP).sum() / len(x) if len(x) > 0 else 0
        ).to_dict()
        accuracies = [daily_acc.get(d, 0) * 100 for d in dates]

        fig1 = make_subplots(specs=[[{"secondary_y": True}]])
        fig1.add_trace(
            go.Bar(name="Volume", x=dates, y=volumes, marker_color=colors["primary"], opacity=0.7),
            secondary_y=False,
        )
        fig1.add_trace(
            go.Scatter(name="Accuracy %", x=dates, y=accuracies, mode="lines+markers",
                      line=dict(color=colors["success"], width=3), marker=dict(size=6)),
            secondary_y=True,
        )
        fig1.update_layout(
            title="Daily Volume & Accuracy Trend (Last 30 Days)",
            xaxis_title="Date",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            template="plotly_white",
        )
        fig1.update_yaxes(title_text="Feedback Volume", secondary_y=False)
        fig1.update_yaxes(title_text="Accuracy (%)", secondary_y=True, range=[0, 100])
        figures_html.append(fig1.to_html(full_html=False, include_plotlyjs=False))

    # --- Figure 2: Weekly Trend ---
    by_week = temporal.get("by_week", {})
    if by_week and len(by_week) > 1:
        weeks = list(by_week.keys())[-12:]  # Last 12 weeks
        week_volumes = [by_week[w] for w in weeks]

        # Calculate weekly accuracy
        df_copy = df.copy()
        df_copy["week_str"] = df_copy["created_at"].dt.to_period("W").astype(str)
        weekly_acc = df_copy.groupby("week_str").apply(
            lambda x: (x["data"].apply(lambda d: safe_get(d, "rating")) == RATING_THUMBS_UP).sum() / len(x) if len(x) > 0 else 0
        ).to_dict()
        week_accuracies = [weekly_acc.get(w, 0) * 100 for w in weeks]

        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(
            go.Bar(name="Volume", x=weeks, y=week_volumes, marker_color=colors["info"], opacity=0.7),
            secondary_y=False,
        )
        fig2.add_trace(
            go.Scatter(name="Accuracy %", x=weeks, y=week_accuracies, mode="lines+markers",
                      line=dict(color=colors["success"], width=3), marker=dict(size=8)),
            secondary_y=True,
        )
        fig2.update_layout(
            title="Weekly Volume & Accuracy Trend",
            xaxis_title="Week",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            template="plotly_white",
        )
        fig2.update_yaxes(title_text="Feedback Volume", secondary_y=False)
        fig2.update_yaxes(title_text="Accuracy (%)", secondary_y=True, range=[0, 100])
        figures_html.append(fig2.to_html(full_html=False, include_plotlyjs=False))

    # --- Figure 2b: Monthly Trend ---
    by_month = temporal.get("by_month", {})
    if by_month and len(by_month) > 1:
        months = list(by_month.keys())[-12:]  # Last 12 months
        month_volumes = [by_month[m] for m in months]

        # Calculate monthly accuracy
        df_copy = df.copy()
        df_copy["month_str"] = df_copy["created_at"].dt.to_period("M").astype(str)
        monthly_acc = df_copy.groupby("month_str").apply(
            lambda x: (x["data"].apply(lambda d: safe_get(d, "rating")) == RATING_THUMBS_UP).sum() / len(x) if len(x) > 0 else 0
        ).to_dict()
        month_accuracies = [monthly_acc.get(m, 0) * 100 for m in months]

        fig2b = make_subplots(specs=[[{"secondary_y": True}]])
        fig2b.add_trace(
            go.Bar(name="Volume", x=months, y=month_volumes, marker_color=colors["warning"], opacity=0.7),
            secondary_y=False,
        )
        fig2b.add_trace(
            go.Scatter(name="Accuracy %", x=months, y=month_accuracies, mode="lines+markers",
                      line=dict(color=colors["success"], width=3), marker=dict(size=8)),
            secondary_y=True,
        )
        fig2b.update_layout(
            title="Monthly Volume & Accuracy Trend",
            xaxis_title="Month",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            template="plotly_white",
        )
        fig2b.update_yaxes(title_text="Feedback Volume", secondary_y=False)
        fig2b.update_yaxes(title_text="Accuracy (%)", secondary_y=True, range=[0, 100])
        figures_html.append(fig2b.to_html(full_html=False, include_plotlyjs=False))

    # --- Figure 3: Rating Distribution (Pie) ---
    fig3 = go.Figure(data=[go.Pie(
        labels=["Thumbs Up (Accurate)", "Thumbs Down (Inaccurate)"],
        values=[thumbs_up, thumbs_down],
        marker_colors=[colors["success"], colors["danger"]],
        hole=0.4,
        textinfo="label+percent+value",
        textposition="outside",
    )])
    fig3.update_layout(
        title="Overall Rating Distribution",
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    figures_html.append(fig3.to_html(full_html=False, include_plotlyjs=False))

    # --- Figure 4: Reason Analysis ---
    reason_dist = reason.get("distribution", {})
    if reason_dist:
        reasons = list(reason_dist.keys())
        reason_counts = list(reason_dist.values())
        reason_labels = [r if r else "(no reason)" for r in reasons]

        # Get positive rate by reason
        pos_by_reason = reason.get("positive_rate_by_reason", {})

        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=reason_counts,
            y=reason_labels,
            orientation="h",
            marker_color=colors["primary"],
            text=[f"{c} ({pos_by_reason.get(r, 0):.0%} positive)" for r, c in zip(reasons, reason_counts)],
            textposition="auto",
        ))
        fig4.update_layout(
            title="Feedback by Reason",
            xaxis_title="Count",
            yaxis_title="Reason",
            template="plotly_white",
            height=max(300, len(reasons) * 40),
        )
        figures_html.append(fig4.to_html(full_html=False, include_plotlyjs=False))

    # --- Figure 5: Day of Week Distribution ---
    by_dow = temporal.get("by_day_of_week", {})
    if by_dow:
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_counts = [by_dow.get(day, 0) for day in day_order]

        fig5 = go.Figure(data=[go.Bar(
            x=day_order,
            y=dow_counts,
            marker_color=[colors["primary"]] * 5 + [colors["warning"]] * 2,  # Weekdays vs weekend
            text=dow_counts,
            textposition="auto",
        )])
        fig5.update_layout(
            title="Feedback by Day of Week",
            xaxis_title="Day",
            yaxis_title="Count",
            template="plotly_white",
        )
        figures_html.append(fig5.to_html(full_html=False, include_plotlyjs=False))

    # --- Figure 6: Hourly Distribution ---
    by_hour = temporal.get("by_hour", {})
    if by_hour:
        hours = list(range(24))
        hour_counts = [by_hour.get(h, 0) for h in hours]
        hour_labels = [f"{h:02d}:00" for h in hours]

        fig6 = go.Figure(data=[go.Bar(
            x=hour_labels,
            y=hour_counts,
            marker_color=colors["info"],
        )])
        fig6.update_layout(
            title="Feedback by Hour of Day",
            xaxis_title="Hour",
            yaxis_title="Count",
            template="plotly_white",
        )
        figures_html.append(fig6.to_html(full_html=False, include_plotlyjs=False))

    # --- Figure 7: Model Analysis (if multiple models) ---
    model_dist = model.get("distribution", {})
    if model_dist and len(model_dist) > 1:
        models = list(model_dist.keys())
        model_counts = list(model_dist.values())
        pos_by_model = model.get("thumbs_up_rate_by_model", {})

        fig7 = go.Figure()
        fig7.add_trace(go.Bar(
            x=model_counts,
            y=models,
            orientation="h",
            marker_color=colors["primary"],
            text=[f"{c} ({pos_by_model.get(m, 0):.0%} accurate)" for m, c in zip(models, model_counts)],
            textposition="auto",
        ))
        fig7.update_layout(
            title="Feedback by Model",
            xaxis_title="Count",
            yaxis_title="Model",
            template="plotly_white",
            height=max(300, len(models) * 50),
        )
        figures_html.append(fig7.to_html(full_html=False, include_plotlyjs=False))

    # Build KPI cards HTML
    def format_change(value: float, is_pct_points: bool = False) -> str:
        if value > 0:
            arrow = "↑"
            color = colors["success"]
        elif value < 0:
            arrow = "↓"
            color = colors["danger"]
        else:
            arrow = "→"
            color = colors["dark"]
        if is_pct_points:
            return f'<span style="color: {color}">{arrow} {abs(value)*100:.1f}pp</span>'
        return f'<span style="color: {color}">{arrow} {abs(value):.1%}</span>'

    # KPI HTML
    wow_vol_html = format_change(wow.get("volume_change", 0)) if wow else "N/A"
    wow_acc_html = format_change(wow.get("accuracy_change", 0), is_pct_points=True) if wow else "N/A"
    mom_vol_html = format_change(mom.get("volume_change", 0)) if mom else "N/A"
    mom_acc_html = format_change(mom.get("accuracy_change", 0), is_pct_points=True) if mom else "N/A"

    date_range = overview.get("date_range", {})
    start_date = date_range.get("earliest", "")[:10]
    end_date = date_range.get("latest", "")[:10]

    # Build complete HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Feedback Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background-color: #f5f7fa;
            color: #333;
            line-height: 1.6;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        .header {{
            background: linear-gradient(135deg, {colors["primary"]}, {colors["info"]});
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .header h1 {{ font-size: 2em; margin-bottom: 10px; }}
        .header p {{ opacity: 0.9; }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .kpi-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .kpi-card .value {{
            font-size: 2.5em;
            font-weight: bold;
            color: {colors["primary"]};
        }}
        .kpi-card .label {{ color: #666; font-size: 0.9em; margin-top: 5px; }}
        .kpi-card .change {{ font-size: 0.85em; margin-top: 8px; }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .chart-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.85em;
        }}
        @media (max-width: 768px) {{
            .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .chart-row {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Feedback Analysis Report</h1>
            <p>Period: {start_date} to {end_date}</p>
        </div>

        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="value">{total_records:,}</div>
                <div class="label">Total Feedback</div>
                <div class="change">WoW: {wow_vol_html} | MoM: {mom_vol_html}</div>
            </div>
            <div class="kpi-card">
                <div class="value" style="color: {colors["success"]}">{accuracy:.1%}</div>
                <div class="label">Accuracy Rate</div>
                <div class="change">WoW: {wow_acc_html} | MoM: {mom_acc_html}</div>
            </div>
            <div class="kpi-card">
                <div class="value" style="color: {colors["success"]}">{thumbs_up:,}</div>
                <div class="label">Thumbs Up (Accurate)</div>
            </div>
            <div class="kpi-card">
                <div class="value" style="color: {colors["danger"]}">{thumbs_down:,}</div>
                <div class="label">Thumbs Down (Inaccurate)</div>
            </div>
        </div>

        {"".join(f'<div class="chart-container">{fig}</div>' for fig in figures_html)}

        <div class="footer">
            Generated by Open WebUI Feedback Analyzer<br>
            Report generated on {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
        </div>
    </div>
</body>
</html>"""

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"HTML report generated: {filepath}")
    except (OSError, PermissionError) as e:
        raise FeedbackAnalyzerError(f"Failed to generate HTML report: {e}")


def generate_chart_image(stats: dict, filepath: Path) -> None:
    """Generate static PNG chart for email/presentations.

    Args:
        stats: Statistics dictionary from generate_statistics
        filepath: Output PNG file path
    """
    temporal = stats.get("temporal_analysis", {})
    by_month = temporal.get("by_month", {})
    by_month_acc = temporal.get("by_month_accuracy", {})

    if not by_month or len(by_month) < 2:
        logger.warning("Not enough monthly data to generate chart")
        return

    # Prepare data
    months = sorted(by_month.keys())
    volumes = [by_month[m] for m in months]
    accuracies = [by_month_acc.get(m, 0) * 100 for m in months]

    # Month name mapping
    month_names = {
        '01': 'Jan', '02': 'Feb', '03': 'Mar', '04': 'Apr',
        '05': 'May', '06': 'Jun', '07': 'Jul', '08': 'Aug',
        '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'
    }

    # Colors
    vol_color = '#4A90D9'  # Professional blue
    acc_color = '#2ECC71'  # Clean green
    bg_color = '#FAFBFC'
    grid_color = '#E8ECF0'
    text_color = '#2C3E50'

    # Create figure
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 11
    fig, ax1 = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('white')
    ax1.set_facecolor(bg_color)

    # X-axis positions
    x = list(range(len(months)))

    # Volume bars
    bar_width = 0.65
    bars = ax1.bar(x, volumes, color=vol_color, alpha=0.85, width=bar_width,
                   edgecolor='white', linewidth=1.5, label='Feedback Volume', zorder=3)

    # Style left y-axis (hide labels since values are on bars)
    ax1.set_ylabel('')
    ax1.tick_params(axis='y', left=False, labelleft=False)
    ax1.set_ylim(0, max(volumes) * 1.25)

    # X-axis labels (Month Year format)
    x_labels = [f"{month_names.get(m[-2:], m[-2:])}\n{m[:4]}" for m in months]
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, fontsize=10, color=text_color)
    ax1.tick_params(axis='x', length=0, pad=8)

    # Volume labels on bars
    for bar, vol in zip(bars, volumes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(volumes)*0.02,
                 f'{vol:,}', ha='center', va='bottom', fontsize=11, fontweight='600',
                 color=vol_color)

    # Accuracy line (right y-axis)
    ax2 = ax1.twinx()
    line = ax2.plot(x, accuracies, color=acc_color, marker='o', linewidth=3.5,
                    markersize=12, label='Accuracy Rate', markerfacecolor='white',
                    markeredgewidth=3, markeredgecolor=acc_color, zorder=5)

    # Style right y-axis (hide labels since values are on line)
    ax2.set_ylabel('')
    ax2.tick_params(axis='y', right=False, labelright=False)
    ax2.set_ylim(0, 105)

    # Accuracy labels
    for i, acc in enumerate(accuracies):
        y_offset = 4 if i == 0 or accuracies[i] >= accuracies[i-1] else -8
        va = 'bottom' if y_offset > 0 else 'top'
        ax2.text(i, acc + y_offset, f'{acc:.1f}%', ha='center', va=va,
                 fontsize=11, fontweight='bold', color=acc_color,
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8))

    # Grid
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(True, color=grid_color, linestyle='-', linewidth=1, alpha=0.7)
    ax1.xaxis.grid(False)

    # Remove spines
    for spine in ['top', 'right', 'left', 'bottom']:
        ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)

    # Title
    date_range = stats.get("overview", {}).get("date_range", {})
    start_date = date_range.get("earliest", "")[:10]
    end_date = date_range.get("latest", "")[:10]
    fig.suptitle('Chatbot Performance Report', fontsize=18, fontweight='bold',
                 color=text_color, y=0.98)
    ax1.set_title(f'Monthly Volume & Accuracy  |  {start_date} to {end_date}',
                  fontsize=12, color='#666', pad=20, style='italic')

    # Legend at bottom
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center',
                        bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=11,
                        frameon=True, fancybox=True, shadow=False,
                        edgecolor=grid_color)

    # Summary box (top left to avoid overlapping with data on right side)
    total_vol = sum(volumes)

    summary_text = f'Total Feedback: {total_vol:,}'

    props = dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=grid_color,
                 linewidth=1.5, alpha=0.95)
    ax1.text(0.02, 0.97, summary_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='left', bbox=props,
             color=text_color, fontweight='500')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.88)

    try:
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white',
                    edgecolor='none', pad_inches=0.3)
        plt.close()
        logger.info(f"Chart image generated: {filepath}")
    except (OSError, PermissionError) as e:
        plt.close()
        raise FeedbackAnalyzerError(f"Failed to generate chart image: {e}")


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
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate interactive HTML report for stakeholders",
    )
    parser.add_argument(
        "--chart",
        action="store_true",
        help="Generate static PNG chart for emails/presentations",
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

            # Generate HTML report if requested
            if parsed.html:
                html_file = parsed.output_dir / f"{actual_start}-{actual_end}-report.html"
                logger.info(f"Generating HTML report to {html_file}...")
                generate_html_report(stats, df, html_file)

            # Generate static chart image if requested
            if parsed.chart:
                chart_file = parsed.output_dir / f"{actual_start}-{actual_end}-chart.png"
                logger.info(f"Generating chart image to {chart_file}...")
                generate_chart_image(stats, chart_file)

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
