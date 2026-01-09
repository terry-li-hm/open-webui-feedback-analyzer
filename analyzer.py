import json
import pandas as pd
from collections import Counter

TIMEZONE = 'Asia/Hong_Kong'

def delete_keys_from_nested_json(data, key_to_delete):
    """Recursively remove a key from nested JSON structures."""
    if isinstance(data, dict):
        return {k: delete_keys_from_nested_json(v, key_to_delete)
                for k, v in data.items() if k != key_to_delete}
    elif isinstance(data, list):
        return [delete_keys_from_nested_json(item, key_to_delete) for item in data]
    return data


def load_feedback_data(filepath):
    """Load and parse feedback JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_by_date_range(data, start_date, end_date):
    """Filter feedback records by date range using root-level created_at."""
    df = pd.DataFrame(data)
    df['created_at'] = pd.to_datetime(df['created_at'], unit='s', utc=True).dt.tz_convert(TIMEZONE)

    start = pd.Timestamp(start_date, tz=TIMEZONE)
    end = pd.Timestamp(end_date, tz=TIMEZONE) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    mask = df['created_at'].between(start, end)
    return df[mask].reset_index(drop=True)


def export_to_json(df, filename):
    """Export DataFrame to JSON, converting timestamps back to Unix seconds."""
    df = df.copy()
    df['created_at'] = df['created_at'].astype('int64') // 10**9
    if 'updated_at' in df.columns:
        df['updated_at'] = pd.to_datetime(df['updated_at'], unit='s').astype('int64') // 10**9

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(df.to_dict('records'), f, indent=4, ensure_ascii=False)


def safe_get(d, *keys, default=None):
    """Safely navigate nested dictionaries."""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d


def generate_statistics(df):
    """Generate comprehensive statistics from feedback data."""
    stats = {
        'overview': {},
        'rating_analysis': {},
        'model_analysis': {},
        'reason_analysis': {},
        'temporal_analysis': {},
        'rag_analysis': {}
    }

    # Overview
    stats['overview']['total_records'] = len(df)
    stats['overview']['date_range'] = {
        'earliest': df['created_at'].min().isoformat(),
        'latest': df['created_at'].max().isoformat()
    }

    # Rating analysis (from data.rating: 1 = thumbs up, -1 = thumbs down)
    ratings = df['data'].apply(lambda x: safe_get(x, 'rating'))
    stats['rating_analysis']['distribution'] = ratings.value_counts(dropna=False).to_dict()
    stats['rating_analysis']['thumbs_up_rate'] = (ratings == 1).sum() / len(ratings) if len(ratings) > 0 else 0

    # Detailed rating (1-10 scale from details.rating)
    if 'details' in df.columns:
        detailed_ratings = df['details'].apply(lambda x: safe_get(x, 'rating'))
        valid_detailed = detailed_ratings.dropna()
        if len(valid_detailed) > 0:
            stats['rating_analysis']['detailed_distribution'] = detailed_ratings.value_counts(dropna=False).sort_index().to_dict()
            stats['rating_analysis']['detailed_average'] = valid_detailed.mean()

    # Reason analysis
    reasons = df['data'].apply(lambda x: safe_get(x, 'reason'))
    stats['reason_analysis']['distribution'] = reasons.value_counts(dropna=False).to_dict()

    # Cross-analysis: rating by reason
    rating_reason = pd.DataFrame({'rating': ratings, 'reason': reasons})
    stats['reason_analysis']['thumbs_up_rate_by_reason'] = (
        rating_reason.groupby('reason')['rating']
        .apply(lambda x: (x == 1).sum() / len(x) if len(x) > 0 else 0)
        .to_dict()
    )

    # Model analysis (from data.model_id)
    models = df['data'].apply(lambda x: safe_get(x, 'model_id'))
    stats['model_analysis']['distribution'] = models.value_counts(dropna=False).to_dict()

    # Base model analysis
    if 'base_models' in df.columns:
        base_models = df['base_models'].apply(
            lambda x: list(x.values())[0] if isinstance(x, dict) and x else None
        )
        stats['model_analysis']['base_model_distribution'] = base_models.value_counts(dropna=False).to_dict()

    # Cross-analysis: rating by model
    rating_model = pd.DataFrame({'rating': ratings, 'model': models})
    stats['model_analysis']['thumbs_up_rate_by_model'] = (
        rating_model.groupby('model')['rating']
        .apply(lambda x: (x == 1).sum() / len(x) if len(x) > 0 else 0)
        .to_dict()
    )

    # Temporal analysis
    stats['temporal_analysis']['by_hour'] = df['created_at'].dt.hour.value_counts().sort_index().to_dict()
    stats['temporal_analysis']['by_day_of_week'] = (
        df['created_at'].dt.day_name().value_counts().to_dict()
    )
    stats['temporal_analysis']['by_date'] = (
        df['created_at'].dt.date.astype(str).value_counts().sort_index().to_dict()
    )

    # RAG analysis (from snapshot.history.messages statusHistory)
    if 'snapshot' in df.columns:
        source_counts = []
        query_counts = []
        action_counter = Counter()

        for _, row in df.iterrows():
            snapshot = row.get('snapshot', {})
            history = safe_get(snapshot, 'history', default={})
            messages = history.get('messages', [])

            # Handle both list and dict formats for messages
            if isinstance(messages, dict):
                messages = list(messages.values())

            for msg in messages:
                if msg.get('role') == 'assistant':
                    status_history = msg.get('statusHistory', [])
                    for status in status_history:
                        action = status.get('action')
                        if action:
                            action_counter[action] += 1
                        if action == 'sources_retrieved':
                            source_counts.append(status.get('count', 0))
                        if action == 'queries_generated':
                            queries = status.get('queries', [])
                            query_counts.append(len(queries))

        stats['rag_analysis']['action_distribution'] = dict(action_counter)
        if source_counts:
            stats['rag_analysis']['sources_retrieved'] = {
                'total': sum(source_counts),
                'average': sum(source_counts) / len(source_counts),
                'distribution': Counter(source_counts)
            }
        if query_counts:
            stats['rag_analysis']['queries_generated'] = {
                'average_per_response': sum(query_counts) / len(query_counts),
                'distribution': Counter(query_counts)
            }

    # Chat metadata analysis
    if 'snapshot' in df.columns:
        archived_count = df['snapshot'].apply(lambda x: safe_get(x, 'archived', default=False)).sum()
        pinned_count = df['snapshot'].apply(lambda x: safe_get(x, 'pinned', default=False)).sum()
        stats['overview']['archived_chats'] = int(archived_count)
        stats['overview']['pinned_chats'] = int(pinned_count)

    return stats


def print_statistics(stats):
    """Print statistics in a readable format."""
    print("\n" + "="*60)
    print("FEEDBACK DATA STATISTICS")
    print("="*60)

    print(f"\n📊 OVERVIEW")
    print(f"   Total records: {stats['overview']['total_records']}")
    print(f"   Date range: {stats['overview']['date_range']['earliest']} to {stats['overview']['date_range']['latest']}")
    if 'archived_chats' in stats['overview']:
        print(f"   Archived chats: {stats['overview']['archived_chats']}")
        print(f"   Pinned chats: {stats['overview']['pinned_chats']}")

    print(f"\n👍 RATING ANALYSIS")
    print(f"   Distribution: {stats['rating_analysis']['distribution']}")
    print(f"   Thumbs up rate: {stats['rating_analysis']['thumbs_up_rate']:.1%}")
    if 'detailed_average' in stats['rating_analysis']:
        print(f"   Detailed rating average: {stats['rating_analysis']['detailed_average']:.1f}/10")

    print(f"\n💬 REASON ANALYSIS")
    for reason, count in stats['reason_analysis']['distribution'].items():
        rate = stats['reason_analysis']['thumbs_up_rate_by_reason'].get(reason, 0)
        print(f"   {reason}: {count} ({rate:.1%} positive)")

    print(f"\n🤖 MODEL ANALYSIS")
    for model, count in stats['model_analysis']['distribution'].items():
        rate = stats['model_analysis']['thumbs_up_rate_by_model'].get(model, 0)
        print(f"   {model}: {count} ({rate:.1%} positive)")

    print(f"\n🔍 RAG ANALYSIS")
    if stats['rag_analysis'].get('action_distribution'):
        print(f"   Actions: {stats['rag_analysis']['action_distribution']}")
    if stats['rag_analysis'].get('sources_retrieved'):
        print(f"   Avg sources retrieved: {stats['rag_analysis']['sources_retrieved']['average']:.1f}")
    if stats['rag_analysis'].get('queries_generated'):
        print(f"   Avg queries generated: {stats['rag_analysis']['queries_generated']['average_per_response']:.1f}")

    print(f"\n📅 TEMPORAL ANALYSIS")
    print(f"   By date: {stats['temporal_analysis']['by_date']}")


def main():
    # Configuration
    input_file = "feedback-history-export-1765275639575.json"
    start_date = '2025-12-01'
    end_date = '2025-12-09'

    # Load data
    print(f"Loading data from {input_file}...")
    data = load_feedback_data(input_file)
    print(f"Loaded {len(data)} total records")

    # Remove sources key
    print("Removing 'sources' from nested structures...")
    filtered_data = delete_keys_from_nested_json(data, "sources")

    # Filter by date
    print(f"Filtering to date range {start_date} to {end_date}...")
    df = filter_by_date_range(filtered_data, start_date, end_date)
    print(f"Filtered to {len(df)} records")

    # Generate statistics
    print("Generating statistics...")
    stats = generate_statistics(df)
    print_statistics(stats)

    # Export filtered data
    output_file = f"{start_date}-{end_date}-data.json"
    print(f"\nExporting to {output_file}...")
    export_to_json(df, output_file)

    # Export statistics
    stats_file = f"{start_date}-{end_date}-statistics.json"
    print(f"Exporting statistics to {stats_file}...")

    # Convert any non-serializable types in stats
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(i) for i in obj]
        elif isinstance(obj, Counter):
            return dict(obj)
        elif isinstance(obj, float):
            return round(obj, 4)
        return obj

    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(make_serializable(stats), f, indent=4, ensure_ascii=False)

    print("Done!")


if __name__ == "__main__":
    main()
