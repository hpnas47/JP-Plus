"""HTML report generation."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from jinja2 import Template

from config.settings import get_settings

logger = logging.getLogger(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CFB Power Ratings - {{ year }} Week {{ week }}</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        header {
            background: #1a365d;
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        header h1 {
            font-size: 24px;
            margin-bottom: 5px;
        }
        header .meta {
            font-size: 14px;
            opacity: 0.9;
        }
        .card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .card h2 {
            font-size: 18px;
            margin-bottom: 15px;
            color: #1a365d;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 10px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }
        th, td {
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }
        th {
            background: #f7fafc;
            font-weight: 600;
            color: #4a5568;
            cursor: pointer;
        }
        th:hover {
            background: #edf2f7;
        }
        tr:hover {
            background: #f7fafc;
        }
        .spread {
            font-weight: 600;
        }
        .spread.home { color: #2b6cb0; }
        .spread.away { color: #c53030; }
        .value-play {
            background: #c6f6d5 !important;
        }
        .edge {
            font-weight: 600;
        }
        .edge.positive { color: #22543d; }
        .edge.negative { color: #742a2a; }
        .confidence {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 500;
        }
        .confidence.High { background: #c6f6d5; color: #22543d; }
        .confidence.Medium { background: #fefcbf; color: #744210; }
        .confidence.Low { background: #fed7d7; color: #742a2a; }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .stat-box {
            background: #f7fafc;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }
        .stat-box .number {
            font-size: 28px;
            font-weight: 700;
            color: #1a365d;
        }
        .stat-box .label {
            font-size: 12px;
            color: #718096;
            text-transform: uppercase;
        }
        .top-teams {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
        }
        .top-team {
            display: flex;
            align-items: center;
            padding: 8px;
            background: #f7fafc;
            border-radius: 4px;
        }
        .top-team .rank {
            font-weight: 700;
            color: #1a365d;
            margin-right: 10px;
            min-width: 25px;
        }
        .top-team .name {
            flex: 1;
        }
        .top-team .rating {
            font-weight: 500;
            color: #4a5568;
        }
        footer {
            text-align: center;
            padding: 20px;
            color: #718096;
            font-size: 12px;
        }
        @media (max-width: 768px) {
            table {
                font-size: 12px;
            }
            th, td {
                padding: 8px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>CFB Power Ratings Model</h1>
            <div class="meta">{{ year }} Season - Week {{ week }} | Generated: {{ generated }}</div>
        </header>

        <div class="stats-grid">
            <div class="stat-box">
                <div class="number">{{ total_games }}</div>
                <div class="label">Games This Week</div>
            </div>
            <div class="stat-box">
                <div class="number">{{ value_plays }}</div>
                <div class="label">Value Plays</div>
            </div>
            <div class="stat-box">
                <div class="number">{{ teams_rated }}</div>
                <div class="label">Teams Rated</div>
            </div>
            <div class="stat-box">
                <div class="number">{{ avg_edge }}</div>
                <div class="label">Avg Edge (pts)</div>
            </div>
        </div>

        {% if value_plays_data %}
        <div class="card">
            <h2>Value Plays (Edge >= {{ value_threshold }} pts)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Play</th>
                        <th>Side</th>
                        <th>Model</th>
                        <th>Vegas</th>
                        <th>Edge</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
                    {% for play in value_plays_data %}
                    <tr class="value-play">
                        <td>{{ play.away_team }} @ {{ play.home_team }}</td>
                        <td><strong>{{ play.team }}</strong> ({{ play.side }})</td>
                        <td>{{ play.model_spread }}</td>
                        <td>{{ play.vegas_spread }}</td>
                        <td class="edge positive">+{{ play.edge }}</td>
                        <td><span class="confidence {{ play.confidence }}">{{ play.confidence }}</span></td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}

        <div class="card">
            <h2>All Predictions</h2>
            <table>
                <thead>
                    <tr>
                        <th>Away</th>
                        <th>Home</th>
                        <th>Model Spread</th>
                        <th>Vegas</th>
                        <th>Edge</th>
                        <th>Win Prob</th>
                    </tr>
                </thead>
                <tbody>
                    {% for game in predictions %}
                    <tr {% if game.is_value %}class="value-play"{% endif %}>
                        <td>{{ game.away_team }}</td>
                        <td>{{ game.home_team }}</td>
                        <td class="spread {% if game.spread > 0 %}home{% else %}away{% endif %}">
                            {{ game.spread }}
                        </td>
                        <td>{{ game.vegas_spread if game.vegas_spread else 'N/A' }}</td>
                        <td class="edge {% if game.edge and game.edge > 0 %}positive{% elif game.edge %}negative{% endif %}">
                            {% if game.edge %}{{ game.edge }}{% else %}--{% endif %}
                        </td>
                        <td>{{ game.home_win_prob }}%</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <div class="card">
            <h2>Top 25 Power Ratings</h2>
            <div class="top-teams">
                {% for team in top_25 %}
                <div class="top-team">
                    <span class="rank">#{{ loop.index }}</span>
                    <span class="name">{{ team.team }}</span>
                    <span class="rating">{{ team.overall }}</span>
                </div>
                {% endfor %}
            </div>
        </div>

        <footer>
            CFB Power Ratings Model | Data from collegefootballdata.com
        </footer>
    </div>

    <script>
        // Simple table sorting
        document.querySelectorAll('th').forEach(th => {
            th.addEventListener('click', () => {
                const table = th.closest('table');
                const tbody = table.querySelector('tbody');
                const rows = Array.from(tbody.querySelectorAll('tr'));
                const index = Array.from(th.parentNode.children).indexOf(th);
                const isNumeric = !isNaN(parseFloat(rows[0]?.children[index]?.textContent));

                rows.sort((a, b) => {
                    const aVal = a.children[index]?.textContent || '';
                    const bVal = b.children[index]?.textContent || '';

                    if (isNumeric) {
                        return parseFloat(bVal) - parseFloat(aVal);
                    }
                    return aVal.localeCompare(bVal);
                });

                rows.forEach(row => tbody.appendChild(row));
            });
        });
    </script>
</body>
</html>
"""


class HTMLReporter:
    """Generate HTML reports for quick viewing."""

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize HTML reporter.

        Args:
            output_dir: Output directory for HTML files
        """
        settings = get_settings()
        self.output_dir = output_dir or settings.outputs_dir
        self.template = Template(HTML_TEMPLATE)

    def generate(
        self,
        predictions_df: pd.DataFrame,
        value_plays_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
        year: int,
        week: int,
        filename: Optional[str] = None,
    ) -> Path:
        """Generate HTML report.

        Args:
            predictions_df: Full predictions DataFrame
            value_plays_df: Value plays DataFrame
            ratings_df: Team ratings DataFrame
            year: Season year
            week: Week number
            filename: Custom filename (optional)

        Returns:
            Path to created file
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            filename = f"cfb_predictions_{year}_week{week}_{timestamp}.html"

        filepath = self.output_dir / filename

        # Prepare predictions data
        predictions_data = []
        for _, row in predictions_df.iterrows():
            predictions_data.append({
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "spread": row.get("model_spread", row.get("spread", 0)),
                "vegas_spread": row.get("vegas_spread"),
                "edge": round(row["edge"], 1) if pd.notna(row.get("edge")) else None,
                "home_win_prob": int(row.get("home_win_prob", 0.5) * 100),
                "is_value": row.get("is_value", False),
            })

        # Prepare value plays data
        value_plays_data = []
        for _, row in value_plays_df.iterrows():
            value_plays_data.append({
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "team": row.get("team", ""),
                "side": row.get("side", ""),
                "model_spread": row.get("model_spread", 0),
                "vegas_spread": row.get("vegas_spread", 0),
                "edge": round(row.get("edge", 0), 1),
                "confidence": row.get("confidence", "Medium"),
            })

        # Prepare top 25 ratings
        top_25 = []
        for _, row in ratings_df.head(25).iterrows():
            top_25.append({
                "team": row["team"],
                "overall": round(row["overall"], 2),
            })

        # Calculate stats
        edges = predictions_df["edge"].dropna()
        avg_edge = round(edges.abs().mean(), 1) if len(edges) > 0 else 0

        settings = get_settings()

        # Render template
        html = self.template.render(
            year=year,
            week=week,
            generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
            total_games=len(predictions_df),
            value_plays=len(value_plays_df),
            teams_rated=len(ratings_df),
            avg_edge=avg_edge,
            value_threshold=settings.value_threshold,
            predictions=predictions_data,
            value_plays_data=value_plays_data,
            top_25=top_25,
        )

        # Write file
        with open(filepath, "w") as f:
            f.write(html)

        logger.info(f"Generated HTML report at {filepath}")
        return filepath
