# Open WebUI Feedback Analyzer

Analyze feedback data exported from Open WebUI, generating comprehensive statistics on ratings, models, reasons, temporal patterns, and RAG usage.

## Requirements

- Python 3.7+
- pandas
- plotly (for HTML reports)
- matplotlib (for static chart images)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python analyzer.py <input_file> [options]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `input_file` | (required) | Path to feedback JSON file exported from Open WebUI |
| `-s, --start` | (all) | Start date (YYYY-MM-DD). If omitted, includes all past data |
| `-e, --end` | (all) | End date (YYYY-MM-DD). If omitted, includes up to latest data |
| `-o, --output-dir` | `.` | Output directory for exported files |
| `--timezone` | `Asia/Hong_Kong` | Timezone for date interpretation |
| `--no-export` | - | Skip exporting filtered data and statistics |
| `--html` | - | Generate interactive HTML report for stakeholders |
| `--chart` | - | Generate static PNG chart for emails/presentations |
| `-q, --quiet` | - | Suppress output (only show errors) |
| `-v, --verbose` | - | Enable verbose logging |

### Examples

```bash
# Analyze all data
python analyzer.py feedback.json

# From a specific date to now
python analyzer.py feedback.json -s 2025-07-01

# Specific date range
python analyzer.py feedback.json -s 2025-12-01 -e 2025-12-09

# With custom timezone and output directory
python analyzer.py feedback.json -s 2025-12-01 -e 2025-12-09 --timezone UTC -o ./output

# Generate HTML report for stakeholders
python analyzer.py feedback.json --html

# Generate static PNG chart for email
python analyzer.py feedback.json --chart
```

## Output

The analyzer generates the following output files:

1. `{start_date}-{end_date}-data.json` - Filtered feedback records
2. `{start_date}-{end_date}-statistics.json` - Comprehensive statistics including:
   - **Overview** - Total records, date range, archived/pinned counts
   - **Rating analysis** - Thumbs up/down distribution and rates
   - **Detailed ratings** - 1-10 scale distribution, average, median
   - **Reason analysis** - Distribution and positive rate by reason
   - **Model analysis** - Distribution and positive rate by model
   - **Temporal analysis** - By hour, day of week, date
   - **RAG analysis** - Actions, sources retrieved, queries generated
   - **Comment analysis** - Comments on negative feedback
   - **Feedback position** - Where in conversations feedback occurs
   - **Tag analysis** - Tag usage distribution
   - **File analysis** - Correlation between file attachments and ratings
   - **Trend analysis** - Week-on-week and month-on-month comparisons

3. `{start_date}-{end_date}-chart.png` (with `--chart` flag) - Static chart image featuring:
   - Monthly volume bars with values
   - Accuracy trend line with percentage labels
   - Summary box with total feedback and accuracy change
   - Perfect for embedding in emails or presentations

4. `{start_date}-{end_date}-report.html` (with `--html` flag) - Interactive HTML report featuring:
   - KPI cards with WoW/MoM trends
   - Daily and weekly volume/accuracy charts
   - Rating distribution pie chart
   - Reason analysis bar chart
   - Day of week and hourly distribution
   - Model comparison (if multiple models)

## Running Tests

```bash
pip install -r requirements-dev.txt
pytest test_analyzer.py -v
```

## Schema Documentation

See [SCHEMA.md](SCHEMA.md) for detailed documentation of the feedback data structure exported by Open WebUI.

## License

MIT License - see [LICENSE](LICENSE) for details.
