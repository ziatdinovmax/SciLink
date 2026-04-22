"""
Generate a monthly SciLink newsletter from git history and GitHub PRs.

Collects commits and merged PRs from the last N days, sends them to an LLM
to identify 2-3 key features, and produces a scientist-friendly newsletter
in Markdown and two HTML variants (full + Outlook-safe).

Requirements:
    - Git repo with remote at github.com
    - LLM API key (GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY)
    - Optional: GITHUB_TOKEN for private repos or higher rate limits
    - pygments (for Python syntax highlighting in HTML)

Usage:
    python generate_newsletter.py                              # last 30 days
    python generate_newsletter.py --days 14                    # last 2 weeks
    python generate_newsletter.py --model claude-sonnet-4-6    # specific model
    python generate_newsletter.py --output newsletters/        # custom output dir
    python generate_newsletter.py --focus "skills, knowledge"  # prioritize topics
    python generate_newsletter.py --context features.md        # ground-truth API/CLI/UI info
    python generate_newsletter.py --template-only              # blank template, no LLM
    python generate_newsletter.py --web-url https://...        # link for Outlook fallback
    python generate_newsletter.py --from-json newsletters/newsletter_2026-03.json --web-url https://...

    Two-step workflow (recommended):
        1. python generate_newsletter.py           # generates all files + JSON
        2. Review the output, prepare figures, upload HTML to shared drive
        3. python generate_newsletter.py --from-json newsletters/newsletter_2026-03.json \
               --figures fig1.png fig2.png --web-url https://...

Outputs:
    newsletter_YYYY-MM.md              Markdown version
    newsletter_YYYY-MM.html            Full HTML (browser/Gmail) — collapsible code examples
    newsletter_YYYY-MM_outlook.html    Outlook-safe HTML — code examples replaced with link
    newsletter_YYYY-MM.json            Raw structured JSON for programmatic use

    Each feature includes three usage examples: UI (always visible with
    screenshot placeholder), Python API, and CLI (collapsible in full HTML,
    linked out in Outlook version).

Newsletter structure:
    - 2-3 key features with explanations and code examples
    - New Use Cases section (placeholder for papers/applications)
    - Upcoming Events section (placeholder)

Post-generation editing:
    The output files are meant to be edited before sending. Common edits:

    UI screenshots:
        Preferred: use --from-json with --figures to re-render with images:
            python generate_newsletter.py --from-json newsletters/newsletter_2026-03.json \
                --figures feature1.png feature2.png
        Use 'none' to skip a position: --figures fig1.png none fig3.png
        Images are embedded as base64 data URIs (self-contained, works in email).

        Manual alternative: replace the placeholder <div> with an <img> tag:
            <img src="https://example.com/figure.png" alt="description"
                 width="100%" style="margin-top: 12px; border-radius: 6px;">

    New Use Cases:
        Search for <!-- REPEAT THIS BLOCK FOR EACH USE CASE --> and fill in
        the paper title, summary, figure, and description. Duplicate the block
        between the REPEAT/END comments for multiple entries.

    Upcoming Events:
        Search for "Upcoming events" and replace the TBA placeholder.

    Feature text / headlines:
        Edit directly in the HTML or Markdown. The JSON file can be edited
        and re-rendered if needed.

    Outlook distribution:
        Use newsletter_YYYY-MM_outlook.html for Outlook (Mac or Windows).
        Open in a browser, Cmd+A / Ctrl+A to select, copy, paste into Outlook
        compose. Pass --web-url to set where the "View code examples" link
        points (e.g., a hosted copy of the full HTML).
"""

import argparse
import base64
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter


# ── Git history collection ───────────────────────────────────────────────

def get_git_log(days: int, repo_dir: str = ".") -> list[dict]:
    """Get commits from the last N days with messages and short stats."""
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    result = subprocess.run(
        [
            "git", "log", f"--since={since}", "--pretty=format:%H|%ad|%s",
            "--date=short", "--stat",
        ],
        capture_output=True, text=True, cwd=repo_dir,
    )
    if result.returncode != 0:
        print(f"Warning: git log failed: {result.stderr}")
        return []

    commits = []
    current = None
    for line in result.stdout.strip().split("\n"):
        if "|" in line and len(line.split("|")) >= 3:
            parts = line.split("|", 2)
            if len(parts[0]) == 40:  # SHA
                if current:
                    commits.append(current)
                current = {
                    "sha": parts[0],
                    "date": parts[1],
                    "message": parts[2],
                    "files_changed": "",
                }
                continue
        if current and line.strip():
            current["files_changed"] += line.strip() + "\n"

    if current:
        commits.append(current)

    return commits


def get_git_diff_summary(days: int, repo_dir: str = ".") -> str:
    """Get a high-level diffstat for the period."""
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    result = subprocess.run(
        ["git", "log", f"--since={since}", "--pretty=format:", "--shortstat"],
        capture_output=True, text=True, cwd=repo_dir,
    )
    return result.stdout.strip()


# ── GitHub PR collection ─────────────────────────────────────────────────

def get_github_prs(
    owner_repo: str, days: int, token: str = None,
) -> list[dict]:
    """Fetch merged PRs from the last N days via GitHub REST API."""
    since = (datetime.now() - timedelta(days=days)).isoformat()
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    prs = []
    page = 1
    while True:
        url = (
            f"https://api.github.com/repos/{owner_repo}/pulls"
            f"?state=closed&sort=updated&direction=desc&per_page=30&page={page}"
        )
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"Warning: GitHub API request failed: {e}")
            break

        batch = resp.json()
        if not batch:
            break

        for pr in batch:
            if not pr.get("merged_at"):
                continue
            if pr["merged_at"] < since:
                return prs  # Past our window, stop
            prs.append({
                "number": pr["number"],
                "title": pr["title"],
                "body": (pr.get("body") or "")[:2000],
                "merged_at": pr["merged_at"][:10],
                "labels": [l["name"] for l in pr.get("labels", [])],
                "url": pr["html_url"],
            })

        page += 1
        if page > 5:  # Safety cap
            break

    return prs


# ── LLM newsletter generation ───────────────────────────────────────────

NEWSLETTER_PROMPT = """You are writing a monthly newsletter for SciLink, an LLM-powered multi-agent system for automating experimental data analysis in scientific research. Your audience is domain scientists (physicists, chemists, materials scientists) who use SciLink but are not software developers.

**Period:** {period}

**Merged Pull Requests:**
{pr_text}

**Git Commits (summaries):**
{commit_text}

**Instructions:**

1. Identify the 2-3 most impactful features or improvements from this period. Focus on what changes the user experience or analytical capabilities — skip internal refactors, CI fixes, or minor code cleanup.

2. For each feature, write:
   - A clear, non-jargon headline
   - A 2-3 sentence explanation of what it does and why it matters for a working scientist
   - For each feature, show how to use it via all three interfaces when applicable:
     (a) **Python API** — a short code snippet (3-8 lines) using SciLink's programmatic API (e.g., `agent.analyze()`, `synthesize_knowledge()`, `register_skill()`)
     (b) **CLI** — the equivalent `scilink analyze` or `scilink plan` command/chat interaction
     (c) **UI** — a brief description of how to do it in the Streamlit web UI (e.g., "Upload files in the sidebar, then click Analyze")
   Use realistic scientific scenarios (Raman, XPS, TEM, DSC, etc.).

{focus_section}{context_section}
3. Keep the tone professional but approachable. No marketing fluff. Scientists appreciate precision.

4. IMPORTANT: If a "Ground truth" section is provided above, use ONLY the API signatures, CLI commands, and UI descriptions from that section for your code examples. Do NOT invent API calls, class names, CLI flags, or UI workflows that are not described there. If the ground truth does not cover a particular interface (API/CLI/UI) for a feature, write "N/A" for that example rather than guessing.

You MUST output valid JSON with this structure:
{{
    "edition_title": "Short catchy title for this edition",
    "features": [
        {{
            "headline": "Feature headline",
            "explanation": "2-3 sentence explanation",
            "example_api": "Short Python API code snippet (3-8 lines)",
            "example_cli": "CLI command or chat interaction (1-3 lines)",
            "example_ui": "Brief description of how to do it in the Streamlit UI (1-2 sentences)"
        }}
    ]
}}

Output ONLY the JSON object.
"""


def generate_newsletter_content(
    commits: list[dict],
    prs: list[dict],
    days: int,
    model: Any,
    focus: str = None,
    context: str = None,
) -> dict:
    """Call LLM to generate newsletter content from raw changes."""
    end_date = datetime.now().strftime("%B %d, %Y")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%B %d, %Y")
    period = f"{start_date} — {end_date}"

    if prs:
        pr_lines = []
        for pr in prs:
            pr_lines.append(
                f"### PR #{pr['number']}: {pr['title']} (merged {pr['merged_at']})\n"
                f"{pr['body']}\n"
            )
        pr_text = "\n".join(pr_lines)
    else:
        pr_text = "(No merged PRs found for this period)"

    if commits:
        commit_lines = [
            f"- [{c['date']}] {c['message']}" for c in commits[:50]
        ]
        commit_text = "\n".join(commit_lines)
    else:
        commit_text = "(No commits found for this period)"

    if focus:
        focus_section = (
            f"**Editorial focus (prioritize these topics):** {focus}\n"
        )
    else:
        focus_section = ""

    if context:
        context_section = (
            f"\n**Ground truth (use these details for code examples and UI descriptions — do not invent alternatives):**\n{context}\n"
        )
    else:
        context_section = ""

    prompt = NEWSLETTER_PROMPT.format(
        period=period,
        pr_text=pr_text,
        commit_text=commit_text,
        focus_section=focus_section,
        context_section=context_section,
    )

    response = model.generate_content(
        contents=[prompt],
        generation_config=None,
        safety_settings=None,
    )
    response_text = response.text if hasattr(response, "text") else str(response)

    json_match = re.search(r"\{[\s\S]*\}", response_text)
    if not json_match:
        raise RuntimeError(
            f"LLM did not return valid JSON. Response:\n{response_text[:500]}"
        )

    return json.loads(json_match.group())


# ── Rendering ────────────────────────────────────────────────────────────

def _figure_tag(figures: list[str], index: int) -> str | None:
    """Return an <img> tag for the figure at the given index, or None."""
    if not figures or index >= len(figures) or not figures[index]:
        return None
    path = Path(figures[index])
    if not path.exists():
        print(f"Warning: figure not found: {path}")
        return None
    suffix = path.suffix.lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "gif": "image/gif", "svg": "image/svg+xml", "webp": "image/webp"}.get(suffix, f"image/{suffix}")
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:{mime};base64,{b64}" alt="Feature {index + 1}" width="760" style="max-width: 100%; margin-top: 12px; border-radius: 6px; border: 1px solid #e5e7eb; padding: 8px; background-color: #ffffff;">'


_SCREENSHOT_PLACEHOLDER_HTML = """<div style="background-color: #ede9fe; border: 1px dashed #a78bfa; border-radius: 6px;
                            padding: 24px; text-align: center;">
                    <p style="color: #6b7280; font-size: 13px; margin: 0;">[ UI screenshot ]</p>
                </div>"""


def render_markdown(content: dict, period_str: str, figures: list[str] = None) -> str:
    """Render newsletter content as Markdown."""
    lines = [
        f"# SciLink Newsletter — {content['edition_title']}",
        f"*{period_str}*\n",
    ]

    for i, feature in enumerate(content["features"], 1):
        lines.append(f"## {i}. {feature['headline']}\n")
        lines.append(f"{feature['explanation']}\n")
        if feature.get('example_ui'):
            lines.append(f"**UI:** {feature['example_ui']}\n")
        fig_path = figures[i - 1] if figures and i - 1 < len(figures) and figures[i - 1] else None
        if fig_path:
            lines.append(f"![Feature {i}]({fig_path})\n")
        else:
            lines.append("*[ UI screenshot ]*\n")
        lines.append(f"**Python API:**\n```python\n{feature['example_api']}\n```\n")
        lines.append(f"**CLI:**\n```\n{feature['example_cli']}\n```\n")

    lines.append("## New Use Cases\n")
    for uc in content.get("use_cases", []):
        lines.append(f"**{uc['title']}**\n")
        summary = uc.get('summary', '')
        if uc.get('link'):
            summary += f" — [{uc.get('link_text', 'Link')}]({uc['link']})"
        lines.append(f"*{summary}*\n")
        if uc.get('figure'):
            lines.append(f"![Use case]({uc['figure']})\n")
        lines.append(f"{uc.get('description', '')}\n")
    if not content.get("use_cases"):
        lines.append("*[TBA]*\n")

    lines.append("## Upcoming events\n")
    events = content.get("upcoming_events", "*[TBA]*")
    lines.append(f"{events}\n")

    return "\n".join(lines)


_LOGO_PATH = Path(__file__).resolve().parent / "scilink" / "ui" / "assets" / "scilink_logo_v3_light.svg"


def _logo_data_uri() -> str:
    """Return a base64 data URI for the SciLink logo, or empty string if missing."""
    if not _LOGO_PATH.exists():
        return ""
    b64 = base64.b64encode(_LOGO_PATH.read_bytes()).decode()
    return f"data:image/svg+xml;base64,{b64}"


def _highlight_python(code: str) -> str:
    """Syntax-highlight Python code as inline-styled HTML (email-safe)."""
    formatter = HtmlFormatter(noclasses=True, nowrap=True, style="monokai")
    return highlight(code, PythonLexer(), formatter)


def _render_code_section(feature: dict, variant: str, web_url: str = None) -> str:
    """Render the Python API & CLI section based on variant."""
    if variant == "full":
        return f"""
                <div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid #e5e7eb;">
                    <details style="margin: 0;">
                        <summary style="color: #6d28d9; font-size: 13px; cursor: pointer;
                                        list-style: none; user-select: none;">
                            <span style="color: #7c3aed;">&#9654;</span> Python API &amp; CLI examples
                        </summary>
                        <div style="margin-top: 8px;">
                            <strong style="color: #6d28d9; font-size: 12px; letter-spacing: 1px;">PYTHON API</strong>
                            <pre style="background-color: #1e1e2e; padding: 14px;
                                        border-radius: 6px; margin: 8px 0 12px 0; font-size: 13px;
                                        line-height: 1.5; overflow-x: auto; border: 1px solid #e5e7eb;">{_highlight_python(feature['example_api'])}</pre>
                            <strong style="color: #2563eb; font-size: 12px; letter-spacing: 1px;">CLI</strong>
                            <pre style="background-color: #1e1e2e; padding: 14px; color: #e2e8f0;
                                        border-radius: 6px; margin: 8px 0 0 0; font-size: 13px;
                                        line-height: 1.5; overflow-x: auto; border: 1px solid #e5e7eb;">{feature['example_cli']}</pre>
                        </div>
                    </details>
                </div>"""
    else:  # outlook — no code examples
        return ""


def render_html(content: dict, period_str: str, web_url: str = None, variant: str = "full", figures: list[str] = None) -> str:
    """Render newsletter content as styled HTML for email.

    variant: "full" (browser/Gmail, with <details>) or "outlook" (no <details>, link only).
    """
    features_html = ""
    for i, feature in enumerate(content["features"], 1):
        code_section = _render_code_section(feature, variant, web_url)
        fig = _figure_tag(figures, i - 1)
        screenshot_block = fig if fig else _SCREENSHOT_PLACEHOLDER_HTML
        features_html += f"""
        <div style="margin-bottom: 36px;">
            <h2 style="color: #1e1b4b; font-size: 20px; margin-bottom: 8px;
                       border-bottom: 2px solid #7c3aed; padding-bottom: 6px; display: inline-block;">
                {i}. {feature['headline']}
            </h2>
            <p style="color: #374151; font-size: 15px; line-height: 1.7;">
                {feature['explanation']}
            </p>
            {screenshot_block}
            {code_section}
        </div>"""

    # Use cases from JSON
    use_cases = content.get("use_cases", [])
    if use_cases:
        uc_blocks = ""
        for uc in use_cases:
            summary = uc.get('summary', '')
            if uc.get('link'):
                summary += f' &mdash; <a href="{uc["link"]}" style="color: #7c3aed;">{uc.get("link_text", "Link")}</a>'
            fig_html = ""
            if uc.get("figure"):
                fig_path = Path(uc["figure"])
                if fig_path.exists():
                    suffix = fig_path.suffix.lower().lstrip(".")
                    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(suffix, f"image/{suffix}")
                    b64 = base64.b64encode(fig_path.read_bytes()).decode()
                    fig_html = f'<div style="text-align: center;"><img src="data:{mime};base64,{b64}" alt="Use case" width="570" style="max-width: 75%; border-radius: 6px; border: 1px solid #e5e7eb; padding: 8px; background-color: #ffffff;"></div>'
            uc_blocks += f"""
            <div style="margin-top: 16px;">
                <p style="color: #6d28d9; font-size: 15px; font-weight: 600; margin: 0;">
                    {uc['title']}
                </p>
                <p style="color: #6b7280; font-size: 14px; margin: 4px 0 10px 0;">
                    {summary}
                </p>
                {fig_html}
                <p style="color: #374151; font-size: 14px; line-height: 1.6; margin-top: 10px;">
                    {uc.get('description', '')}
                </p>
            </div>"""
        use_cases_html = f"""
        <div style="margin-top: 36px; padding-top: 20px; border-top: 1px solid #e5e7eb;">
            <h2 style="color: #1e1b4b; font-size: 20px; margin-bottom: 8px;
                       border-bottom: 2px solid #7c3aed; padding-bottom: 6px; display: inline-block;">New Use Cases</h2>
            {uc_blocks}
        </div>"""
    else:
        use_cases_html = """
        <div style="margin-top: 36px; padding-top: 20px; border-top: 1px solid #e5e7eb;">
            <h2 style="color: #1e1b4b; font-size: 20px; margin-bottom: 8px;
                       border-bottom: 2px solid #7c3aed; padding-bottom: 6px; display: inline-block;">New Use Cases</h2>
            <p style="color: #9ca3af; font-size: 14px; font-style: italic;">TBA</p>
        </div>"""

    # Events from JSON
    events_text = content.get("upcoming_events", "TBA")
    if events_text == "TBA":
        events_p = '<p style="color: #9ca3af; font-size: 14px; font-style: italic;">TBA</p>'
    else:
        events_p = f'<p style="color: #374151; font-size: 14px;">{events_text}</p>'
    also_html = f"""
        <div style="margin-top: 36px; padding-top: 20px; border-top: 1px solid #e5e7eb;">
            <h2 style="color: #1e1b4b; font-size: 20px; margin-bottom: 8px;
                       border-bottom: 2px solid #7c3aed; padding-bottom: 6px; display: inline-block;">Upcoming Events</h2>
            {events_p}
        </div>"""

    return f"""<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="color-scheme" content="light">
    <meta name="supported-color-schemes" content="light">
</head>
<body style="margin: 0; padding: 0; background-color: #f3f4f6;
       font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
    <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%"
           style="background-color: #f3f4f6;" bgcolor="#f3f4f6">
        <tr>
            <td align="center" style="padding: 20px 0;">
                <table role="presentation" cellpadding="0" cellspacing="0" border="0"
                       width="800" style="max-width: 800px; background-color: #ffffff;
                       border: 1px solid #e5e7eb;" bgcolor="#ffffff">
                    <tr>
                        <td style="padding: 40px;">
        <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%" style="margin-bottom: 36px;">
            {"" if not _logo_data_uri() else f'''<tr><td align="center" style="padding-bottom: 16px;">
                <img src="{_logo_data_uri()}" alt="SciLink" width="100" height="100">
            </td></tr>'''}
            <tr><td align="center">
                <h1 style="color: #1e1b4b; font-size: 28px; margin: 0 0 4px 0; letter-spacing: -0.5px;">
                    SciLink Newsletter
                </h1>
            </td></tr>
            <tr><td align="center">
                <p style="color: #7c3aed; font-size: 19px; font-weight: 700; margin: 4px 0 0 0;">
                    {content['edition_title']}
                </p>
            </td></tr>
            <tr><td align="center">
                <p style="color: #9ca3af; font-size: 13px; margin-top: 10px; letter-spacing: 0.5px;">
                    {period_str}
                </p>
            </td></tr>
            <tr><td align="center" style="padding-top: 16px;">
                <table role="presentation" cellpadding="0" cellspacing="0" border="0">
                    <tr><td width="60" height="2" style="background-color: #7c3aed; font-size: 0; line-height: 0;" bgcolor="#7c3aed">&#8202;</td></tr>
                </table>
            </td></tr>
        </table>

        {features_html}
        {use_cases_html}
        {also_html}

        <table role="presentation" cellpadding="0" cellspacing="0" border="0"
               style="margin: 40px auto 0 auto;" align="center">
            <tr><td width="30" height="2" style="background-color: #7c3aed; font-size: 0; line-height: 0;" bgcolor="#7c3aed">&#8202;</td></tr>
        </table>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>"""


def _blank_template() -> dict:
    """Return a placeholder-only newsletter structure."""
    feature = {
        "headline": "[ Feature headline ]",
        "explanation": "[ 2-3 sentence explanation ]",
        "example_api": "# [ Python API example ]",
        "example_cli": "# [ CLI example ]",
        "example_ui": "[ UI description ]",
    }
    return {
        "edition_title": "[ Edition title ]",
        "features": [feature, feature, feature],
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate a monthly SciLink newsletter"
    )
    parser.add_argument(
        "--days", type=int, default=30,
        help="Look back this many days (default: 30)",
    )
    parser.add_argument(
        "--model", default="claude-sonnet-4-6",
        help="LLM model name (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--repo", default=".",
        help="Path to the git repo (default: current directory)",
    )
    parser.add_argument(
        "--github-repo", default="ziatdinovmax/SciLink",
        help="GitHub owner/repo for PR fetching",
    )
    parser.add_argument(
        "--output", default="./newsletters",
        help="Output directory (default: ./newsletters)",
    )
    parser.add_argument(
        "--github-token", default=None,
        help="GitHub token (or set GITHUB_TOKEN env var)",
    )
    parser.add_argument(
        "--focus", default=None,
        help="Guide what to prioritize (e.g., 'curve fitting improvements, new skill system')",
    )
    parser.add_argument(
        "--context", default=None, metavar="PATH",
        help="Path to a file with ground-truth usage info (API signatures, CLI syntax, UI flow). "
             "Injected into the LLM prompt so examples stay accurate.",
    )
    parser.add_argument(
        "--template-only", action="store_true",
        help="Generate a blank template with placeholders only (no LLM call)",
    )
    parser.add_argument(
        "--from-json", default=None, metavar="PATH",
        help="Re-render from a previously saved JSON file (no LLM call). "
             "Useful for adding --web-url after uploading the full HTML.",
    )
    parser.add_argument(
        "--figures", nargs="+", default=None, metavar="PATH",
        help="Image files for feature screenshots, one per feature in order. "
             "Use 'none' to skip a position (e.g., --figures fig1.png none fig3.png).",
    )
    parser.add_argument(
        "--web-url", default=None,
        help="URL where the full newsletter will be hosted (used as fallback link in Outlook)",
    )
    args = parser.parse_args()

    # Generate
    if args.from_json:
        content = json.loads(Path(args.from_json).read_text())
        print(f"Loaded newsletter content from {args.from_json}")
    elif args.template_only:
        content = _blank_template()
    else:
        import os
        github_token = args.github_token or os.environ.get("GITHUB_TOKEN")

        print(f"Collecting changes from the last {args.days} days...")

        commits = get_git_log(args.days, args.repo)
        print(f"  Found {len(commits)} commits")

        prs = get_github_prs(args.github_repo, args.days, github_token)
        print(f"  Found {len(prs)} merged PRs")

        if not commits and not prs:
            print("No changes found for this period. Nothing to report.")
            sys.exit(0)
        # Initialize LLM
        from scilink.auth import get_api_key, get_api_key_for_model
        from scilink.wrappers.litellm_wrapper import LiteLLMGenerativeModel

        api_key = get_api_key_for_model(args.model)
        if not api_key:
            for service in ("google", "openai", "anthropic"):
                api_key = get_api_key(service)
                if api_key:
                    break
        if not api_key:
            print("No API key found. Set GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY.")
            sys.exit(1)

        model = LiteLLMGenerativeModel(model=args.model, api_key=api_key)
        print(f"Generating newsletter with {args.model}...")
        context_text = Path(args.context).read_text() if args.context else None
        content = generate_newsletter_content(commits, prs, args.days, model, args.focus, context_text)

    # Normalize figures: "none" → empty string (skip that position)
    figures = None
    if args.figures:
        figures = [f if f.lower() != "none" else "" for f in args.figures]

    # Render
    end_date = datetime.now().strftime("%B %d, %Y")
    start_date = (datetime.now() - timedelta(days=args.days)).strftime("%B %d, %Y")
    period_str = f"{start_date} — {end_date}"

    md = render_markdown(content, period_str, figures=figures)
    html_full = render_html(content, period_str, web_url=args.web_url, variant="full", figures=figures)
    html_outlook = render_html(content, period_str, web_url=args.web_url, variant="outlook", figures=figures)

    # Save
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_tag = datetime.now().strftime("%Y-%m")

    md_path = out_dir / f"newsletter_{date_tag}.md"
    html_path = out_dir / f"newsletter_{date_tag}.html"
    outlook_path = out_dir / f"newsletter_{date_tag}_outlook.html"
    json_path = out_dir / f"newsletter_{date_tag}.json"

    md_path.write_text(md)
    html_path.write_text(html_full)
    outlook_path.write_text(html_outlook)
    json_path.write_text(json.dumps(content, indent=2))

    print(f"\nNewsletter generated:")
    print(f"  Markdown:     {md_path}")
    print(f"  HTML (full):  {html_path}")
    print(f"  HTML (email): {outlook_path}")
    print(f"  Raw JSON:     {json_path}")
    print(f"\nTitle: {content['edition_title']}")
    print(f"Features: {len(content['features'])}")
    print(f"\nUse the full HTML for browser/Gmail. Use the email HTML for Outlook.")


if __name__ == "__main__":
    main()
