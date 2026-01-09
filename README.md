# Open WebUI Feedback Analyzer

Analyze feedback data exported from Open WebUI, generating comprehensive statistics on ratings, models, reasons, temporal patterns, and RAG usage.

## Requirements

- Python 3.7+
- pandas

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python analyzer.py <input_file> -s <start_date> -e <end_date> [options]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `input_file` | Path to feedback JSON file exported from Open WebUI |
| `-s, --start` | Start date (YYYY-MM-DD) |
| `-e, --end` | End date (YYYY-MM-DD) |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-o, --output-dir` | `.` | Output directory for exported files |
| `--timezone` | `Asia/Hong_Kong` | Timezone for date interpretation |
| `--no-export` | - | Skip exporting filtered data and statistics |
| `-q, --quiet` | - | Suppress output (only show errors) |
| `-v, --verbose` | - | Enable verbose logging |

### Examples

```bash
# Basic usage
python analyzer.py feedback.json -s 2025-12-01 -e 2025-12-09

# With custom timezone and output directory
python analyzer.py feedback.json -s 2025-12-01 -e 2025-12-09 --timezone UTC -o ./output
```

## Output

The analyzer generates two output files:

1. `{start_date}-{end_date}-data.json` - Filtered feedback records
2. `{start_date}-{end_date}-statistics.json` - Comprehensive statistics including:
   - Overview (total records, date range, archived/pinned counts)
   - Rating analysis (thumbs up/down distribution, detailed ratings)
   - Reason analysis (distribution and positive rate by reason)
   - Model analysis (distribution and positive rate by model)
   - Temporal analysis (by hour, day of week, date)
   - RAG analysis (actions, sources retrieved, queries generated)

## Running Tests

```bash
pip install -r requirements-dev.txt
pytest test_analyzer.py -v
```
