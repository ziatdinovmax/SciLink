"""House prose-style rule appended to every orchestrator system prompt.

One sentence, principle-form (see CLAUDE.md on prompt patches). The
document-authoring prompts (white paper / technical document) carry a
fuller style block of their own; this constant extends the two banned
rhetorical moves to everything else the agents write — chat replies,
summaries, interpretations, reports, plans.
"""

PROSE_STYLE_RULE = (
    "**Writing style (all prose you produce):** never use the phrase "
    '"load-bearing", and never the "not merely X, but Y" construction — '
    "state the point directly."
)
