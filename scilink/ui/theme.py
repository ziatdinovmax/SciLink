"""Material Design theme — dark and light variants."""

import streamlit as st

_MATERIAL_CSS_DARK = """
<style>
/* ── Sidebar ────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background-color: #1E2530;
    border-right: 1px solid #3A4556;
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] > div:first-child > div:first-child {
    padding-top: 0 !important;
}
/* Tighten sidebar vertical spacing */
section[data-testid="stSidebar"] .block-container {
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding-top: 0.5rem !important;
}
/* Keep collapse button (<<) click target aligned */
[data-testid="stSidebarCollapseButton"] {
    z-index: 999;
    pointer-events: auto !important;
}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div {
    gap: 0.4rem;
}
section[data-testid="stSidebar"] h1 {
    font-size: 1.4em !important;
    margin: 0 0 0.25rem 0 !important;
    padding: 0 !important;
}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    font-size: 0.95em !important;
    margin: 0.25rem 0 0.15rem 0 !important;
    padding: 0 !important;
}
/* Pull the logo up toward the top of the sidebar */
section[data-testid="stSidebar"] [data-testid="stMarkdown"]:has(.logo-glow-sm) {
    margin-top: -2rem !important;
    padding-top: 0 !important;
}

/* ── Sidebar section dividers ───────────────────────── */
section[data-testid="stSidebar"] hr {
    border-color: #3A4556;
    margin: 0.4rem 0;
}

/* ── Sidebar metric styling ─────────────────────────── */
section[data-testid="stSidebar"] [data-testid="stMetric"] {
    background-color: #2A3340;
    border: 1px solid #3A4556;
    border-radius: 6px;
    padding: 0.5rem 0.75rem;
}
section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
    color: #9E9E9E;
    font-size: 0.75em;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: #E0E0E0;
    font-size: 0.9em;
}

/* ── Disabled sidebar widgets ──────────────────────── */
section[data-testid="stSidebar"] .stSelectbox:has([aria-disabled="true"]),
section[data-testid="stSidebar"] .stSelectbox:has([disabled]),
section[data-testid="stSidebar"] .stTextInput:has([aria-disabled="true"]),
section[data-testid="stSidebar"] .stTextInput:has([disabled]),
section[data-testid="stSidebar"] .stTextArea:has([aria-disabled="true"]),
section[data-testid="stSidebar"] .stTextArea:has([disabled]),
section[data-testid="stSidebar"] .stCheckbox:has([aria-disabled="true"]),
section[data-testid="stSidebar"] .stCheckbox:has([disabled]) {
    opacity: 0.40 !important;
}
section[data-testid="stSidebar"] .stButton > button:disabled {
    opacity: 0.35 !important;
    background-color: #BDBDBD !important;
    color: #757575 !important;
}
section[data-testid="stSidebar"] .stButton > button:disabled * {
    color: #757575 !important;
}

/* ── Buttons ────────────────────────────────────────── */
.stButton > button {
    border: none;
    border-radius: 4px;
    font-weight: 500;
    letter-spacing: 0.3px;
    transition: background-color 0.2s, box-shadow 0.2s;
    white-space: nowrap;
}
/* Primary buttons — purple (default for all) */
button[kind="primary"],
.stButton > button[kind="primary"] {
    background-color: #6200EE !important;
    color: #FFFFFF !important;
    font-weight: 600;
}
/* Uppercase only for action buttons inside the chat tabs */
.stTabs .stButton > button[kind="primary"] {
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
button[kind="primary"]:hover,
.stButton > button[kind="primary"]:hover {
    background-color: #7C4DFF !important;
    box-shadow: 0 2px 8px rgba(98, 0, 238, 0.35);
}
/* Mode selector — pull buttons up */
div:has(> [data-testid="stMarkdown"] .mode-selector-anchor) {
    margin-bottom: -2rem !important;
}
div:has(> [data-testid="stMarkdown"] .mode-selector-anchor) + div {
    margin-top: -2rem !important;
}
/* Sidebar buttons — original purple */
section[data-testid="stSidebar"] .stButton > button {
    background-color: #6200EE;
    color: #FFFFFF;
    font-size: 0.85em;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    padding: 0.4rem 0.6rem;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #7C4DFF;
    box-shadow: 0 2px 8px rgba(98, 0, 238, 0.35);
}
/* Theme toggle — pull up to header level and neutral gray */
div:has(> [data-testid="stMarkdown"] .theme-toggle-anchor) {
    margin-top: -2.5rem !important;
    margin-bottom: -1rem !important;
}
div:has(> [data-testid="stMarkdown"] .theme-toggle-anchor) + div {
    margin-top: -1rem !important;
    margin-bottom: 0 !important;
}
div:has(> [data-testid="stMarkdown"] .theme-toggle-anchor) + div button {
    background-color: #3A4556 !important;
    color: #B0BEC5 !important;
    border: 1px solid #4A5568 !important;
    font-size: 1em !important;
    padding: 0.2rem 0.5rem !important;
    box-shadow: none !important;
}
div:has(> [data-testid="stMarkdown"] .theme-toggle-anchor) + div button:hover {
    background-color: #4A5568 !important;
    color: #E0E0E0 !important;
    box-shadow: none !important;
}
/* File explorer tree buttons — soft gray, blue on selection */
[data-testid="stExpander"] .stButton > button {
    background-color: #2A3340;
    color: #B0BEC5;
    border: 1px solid #3A4556;
    text-transform: none;
    font-weight: 400;
    letter-spacing: 0;
    font-size: 0.85em;
    padding: 0.25rem 0.5rem;
}
[data-testid="stExpander"] .stButton > button:hover {
    background-color: #344155;
    border-color: #5B8DEF;
    color: #E0E0E0;
    box-shadow: none;
}
[data-testid="stExpander"] .stButton > button[kind="primary"] {
    background-color: #1A3A5C;
    border-color: #5B8DEF;
    color: #82B1FF;
}

/* ── Success-style button (used via st.markdown class) ─ */
.success-btn > button {
    background-color: #03DAC6 !important;
    color: #121212 !important;
    font-weight: 600;
}
.success-btn > button:hover {
    background-color: #04F1DB !important;
    box-shadow: 0 2px 8px rgba(3, 218, 198, 0.35) !important;
}

/* ── Chat messages ──────────────────────────────────── */
.stChatMessage {
    border-radius: 8px;
    border: 1px solid #3A4556;
}

/* ── Tabs ───────────────────────────────────────────── */
.stTabs [data-baseweb="tab"] {
    color: #B0B0B0;
    font-size: 1.05rem !important;
}
.stTabs [data-baseweb="tab"] * {
    font-size: 1.05rem !important;
}
.stTabs [aria-selected="true"] {
    color: #82B1FF;
    border-bottom-color: #82B1FF;
}

/* ── Expanders ──────────────────────────────────────── */
.streamlit-expanderHeader {
    color: #B0B0B0;
    font-size: 0.85em;
}

/* ── Text inputs & text areas ───────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background-color: #1E2530;
    color: #E0E0E0;
    border: 1px solid #4A5568;
    border-radius: 4px;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #BB86FC;
    box-shadow: 0 0 4px rgba(187, 134, 252, 0.3);
}

/* ── Select boxes ───────────────────────────────────── */
.stSelectbox > div > div {
    background-color: #1E2530;
    border: 1px solid #4A5568;
    border-radius: 4px;
}

/* ── Upload hero box ───────────────────────────────── */
.upload-hero-box {
    border: 2px dashed #4A5568;
    border-radius: 10px;
    padding: 32px 16px;
    text-align: center;
    margin-bottom: 16px;
    background: #1E2530;
}
.upload-hero-title {
    color: #82B1FF;
    font-size: 1.1em;
    margin: 0 0 4px 0;
}
.upload-hero-subtitle {
    color: #6B7A8C;
    font-size: 0.85em;
    margin: 0;
}

/* ── File uploader ──────────────────────────────────── */
.stFileUploader > div {
    border: 1px dashed #4A5568;
    border-radius: 4px;
}

/* ── Code blocks ────────────────────────────────────── */
.stCodeBlock {
    border: 1px solid #3A4556;
    border-radius: 4px;
}

/* ── Scrollbar (webkit) ─────────────────────────────── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #252D38;
}
::-webkit-scrollbar-thumb {
    background: #4A5568;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #82B1FF;
}

/* ── Headings ───────────────────────────────────────── */
h1 {
    color: #82B1FF !important;
}
h2, h3 {
    color: #E0E0E0 !important;
}

/* ── Agent working spinner ──────────────────────────── */
@keyframes scilink-pulse {
    0%, 100% { opacity: 0.4; transform: scale(1); }
    50%      { opacity: 1;   transform: scale(1.05); }
}
.agent-spinner-container {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    background: linear-gradient(135deg, #1E2530 0%, #252D38 100%);
    border: 1px solid #3A4556;
    border-left: 3px solid #4FC3F7;
    border-radius: 6px;
}
.agent-spinner-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #4FC3F7;
    animation: scilink-pulse 1.4s ease-in-out infinite;
    font-size: 0;
    line-height: 0;
    overflow: hidden;
}
.agent-spinner-dot:nth-child(2) { animation-delay: 0.2s; }
.agent-spinner-dot:nth-child(3) { animation-delay: 0.4s; }
.agent-spinner-heart {
    font-size: 1.1em;
    animation: scilink-pulse 1.4s ease-in-out infinite;
}
.agent-spinner-heart:nth-child(2) { animation-delay: 0.2s; }
.agent-spinner-heart:nth-child(3) { animation-delay: 0.4s; }
.agent-spinner-label {
    color: #E0E0E0;
    font-size: 0.9em;
    font-weight: 500;
}

/* ── Stop button (square icon beside spinner) ──────── */
/* Push the button wrapper down to align with the spinner bar */
[data-testid="stHorizontalBlock"]:has(.agent-spinner-container)
    > [data-testid="stColumn"]:last-child .stButton {
    padding-top: 10px;
}
/* ── Verbose output toggle ─────────────────────────── */
[data-testid="stToggle"] label span {
    color: #B0BEC5 !important;
    font-size: 0.9em;
}
[data-testid="stToggle"] [role="checkbox"] {
    background-color: #4A5568 !important;
}
[data-testid="stToggle"] [role="checkbox"][aria-checked="true"] {
    background-color: #4FC3F7 !important;
}
.stTabs .stButton > button[kind="secondary"] {
    width: 100% !important;
    height: 58px !important;
    min-height: 58px !important;
    padding: 0 !important;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1em;
    border-radius: 6px;
    background-color: #546E7A !important;
    color: #E0E0E0 !important;
    border: 1px solid #607D8B !important;
    line-height: 1;
    text-transform: none;
}
.stTabs .stButton > button[kind="secondary"]:hover {
    background-color: #D32F2F !important;
    border-color: #D32F2F !important;
    color: #FFFFFF !important;
}

/* ── Live log viewer ────────────────────────────────── */
.live-log-viewer {
    height: 280px;
    overflow-y: auto;
    margin: 0;
    background: #1E2530;
    padding: 8px;
    border-radius: 4px;
    border: 1px solid #3A4556;
    font-family: monospace;
    font-size: 13px;
    white-space: pre-wrap;
    color: #e0e0e0;
}

/* ── Hide Streamlit chrome (deploy, menu, stop) ────── */
.stDeployButton,
[data-testid="stAppDeployButton"],
#MainMenu,
[data-testid="stMainMenu"],
header [data-testid="stStatusWidget"] {
    display: none !important;
}


/* ── Floating background emojis ───────────────────── */
.floating-emojis {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
    animation: emojis-fade-in 2s ease-out forwards;
}
@keyframes emojis-fade-in {
    from { opacity: 0; }
    to   { opacity: 1; }
}
.floating-emojis span {
    position: absolute;
    display: block;
    font-size: var(--emoji-size, 28px);
    opacity: 0;
    animation: emoji-float var(--duration, 18s) var(--delay, 0s) ease-in-out infinite;
}
.floating-emojis span.rocket {
    animation: rocket-float var(--duration, 18s) var(--delay, 0s) linear infinite;
}
@keyframes emoji-float {
    0% {
        opacity: 0;
        transform: translateY(100vh) rotate(0deg);
    }
    3% {
        opacity: var(--peak-opacity, 0.12);
    }
    93% {
        opacity: var(--peak-opacity, 0.12);
    }
    100% {
        opacity: 0;
        transform: translateY(-10vh) rotate(var(--rotation, 360deg));
    }
}
@keyframes ufo-zip {
    0% {
        opacity: 0;
        transform: translateX(-10vw) translateY(0);
    }
    2% {
        opacity: var(--peak-opacity, 0.18);
    }
    50% {
        transform: translateX(50vw) translateY(-30px);
    }
    98% {
        opacity: var(--peak-opacity, 0.18);
    }
    100% {
        opacity: 0;
        transform: translateX(110vw) translateY(0);
    }
}
.floating-emojis span.ufo {
    bottom: auto;
    animation: ufo-zip var(--duration, 40s) var(--delay, 20s) linear infinite;
}
@keyframes rocket-float {
    0% {
        opacity: 0;
        transform: translateY(100vh) rotate(-45deg);
    }
    3% {
        opacity: var(--peak-opacity, 0.12);
    }
    93% {
        opacity: var(--peak-opacity, 0.12);
    }
    100% {
        opacity: 0;
        transform: translateY(-10vh) rotate(-45deg);
    }
}
</style>
"""

_MATERIAL_CSS_LIGHT = """
<style>
/* ── Override Streamlit base colors for light mode ─── */
.stApp, [data-testid="stAppViewContainer"] {
    background-color: #FAFAFA !important;
    color: #212121 !important;
}
[data-testid="stHeader"] {
    background-color: #FAFAFA !important;
}
/* Force all text/labels dark in light mode */
.stApp label, .stApp p, .stApp span, .stApp div,
.stApp [data-testid="stMarkdown"],
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] [data-testid="stMarkdown"] {
    color: #212121;
}

/* ── Sidebar ────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background-color: #F0F0F0;
    border-right: 1px solid #E0E0E0;
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] > div:first-child > div:first-child {
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding-top: 0.5rem !important;
}
/* Keep collapse button (<<) click target aligned in light mode */
[data-testid="stSidebarCollapseButton"] {
    z-index: 999;
    pointer-events: auto !important;
}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div {
    gap: 0.4rem;
}
section[data-testid="stSidebar"] h1 {
    font-size: 1.4em !important;
    margin: 0 0 0.25rem 0 !important;
    padding: 0 !important;
}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    font-size: 0.95em !important;
    margin: 0.25rem 0 0.15rem 0 !important;
    padding: 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stMarkdown"]:has(.logo-glow-sm) {
    margin-top: -2rem !important;
    padding-top: 0 !important;
}

/* ── Sidebar section dividers ───────────────────────── */
section[data-testid="stSidebar"] hr {
    border-color: #E0E0E0;
    margin: 0.4rem 0;
}

/* ── Sidebar metric styling ─────────────────────────── */
section[data-testid="stSidebar"] [data-testid="stMetric"] {
    background-color: #EEEEEE;
    border: 1px solid #E0E0E0;
    border-radius: 6px;
    padding: 0.5rem 0.75rem;
}
section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
    color: #212121;
    font-size: 0.75em;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: #212121;
    font-size: 0.9em;
}

/* ── Disabled sidebar widgets ──────────────────────── */
section[data-testid="stSidebar"] .stSelectbox:has([aria-disabled="true"]),
section[data-testid="stSidebar"] .stSelectbox:has([disabled]),
section[data-testid="stSidebar"] .stTextInput:has([aria-disabled="true"]),
section[data-testid="stSidebar"] .stTextInput:has([disabled]),
section[data-testid="stSidebar"] .stTextArea:has([aria-disabled="true"]),
section[data-testid="stSidebar"] .stTextArea:has([disabled]),
section[data-testid="stSidebar"] .stCheckbox:has([aria-disabled="true"]),
section[data-testid="stSidebar"] .stCheckbox:has([disabled]) {
    opacity: 0.40 !important;
}
section[data-testid="stSidebar"] .stButton > button:disabled {
    opacity: 0.35 !important;
    background-color: #3A4556 !important;
    color: #6B7A8C !important;
}
section[data-testid="stSidebar"] .stButton > button:disabled * {
    color: #6B7A8C !important;
}

/* ── Buttons ────────────────────────────────────────── */
.stButton > button {
    border: none;
    border-radius: 4px;
    font-weight: 500;
    letter-spacing: 0.3px;
    transition: background-color 0.2s, box-shadow 0.2s;
    white-space: nowrap;
}
button[kind="primary"],
.stButton > button[kind="primary"] {
    background-color: #6200EE !important;
    color: #FFFFFF !important;
    font-weight: 600;
}
button[kind="primary"] *,
.stButton > button[kind="primary"] * {
    color: #FFFFFF !important;
}
.stTabs .stButton > button[kind="primary"] {
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
button[kind="primary"]:hover,
.stButton > button[kind="primary"]:hover {
    background-color: #7C4DFF !important;
    box-shadow: 0 2px 8px rgba(98, 0, 238, 0.25);
}
/* Mode selector — pull buttons up */
div:has(> [data-testid="stMarkdown"] .mode-selector-anchor) {
    margin-bottom: -2rem !important;
}
div:has(> [data-testid="stMarkdown"] .mode-selector-anchor) + div {
    margin-top: -2rem !important;
}
/* Mode selector secondary buttons — outlined purple */
div:has(> [data-testid="stMarkdown"] .mode-selector-anchor) + div .stButton > button[kind="secondary"] {
    background-color: #FFFFFF !important;
    color: #6200EE !important;
    border: 1.5px solid #6200EE !important;
    font-weight: 600;
}
div:has(> [data-testid="stMarkdown"] .mode-selector-anchor) + div .stButton > button[kind="secondary"] * {
    color: #6200EE !important;
}
div:has(> [data-testid="stMarkdown"] .mode-selector-anchor) + div .stButton > button[kind="secondary"]:hover {
    background-color: #F3E8FF !important;
    color: #6200EE !important;
    border-color: #7C4DFF !important;
    box-shadow: 0 2px 8px rgba(98, 0, 238, 0.15);
}
section[data-testid="stSidebar"] .stButton > button {
    background-color: #6200EE;
    color: #FFFFFF !important;
    font-size: 0.85em;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    padding: 0.4rem 0.6rem;
}
section[data-testid="stSidebar"] .stButton > button * {
    color: #FFFFFF !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #7C4DFF;
    color: #FFFFFF !important;
    box-shadow: 0 2px 8px rgba(98, 0, 238, 0.25);
}
/* Theme toggle — pull up to header level and neutral gray (light) */
div:has(> [data-testid="stMarkdown"] .theme-toggle-anchor) {
    margin-top: -2.5rem !important;
    margin-bottom: -1rem !important;
}
div:has(> [data-testid="stMarkdown"] .theme-toggle-anchor) + div {
    margin-top: -1rem !important;
    margin-bottom: 0 !important;
}
div:has(> [data-testid="stMarkdown"] .theme-toggle-anchor) + div button {
    background-color: #E0E0E0 !important;
    color: #424242 !important;
    border: 1px solid #BDBDBD !important;
    font-size: 1em !important;
    padding: 0.2rem 0.5rem !important;
    box-shadow: none !important;
}
div:has(> [data-testid="stMarkdown"] .theme-toggle-anchor) + div button:hover {
    background-color: #BDBDBD !important;
    color: #212121 !important;
    box-shadow: none !important;
}
[data-testid="stExpander"] .stButton > button,
[data-testid="stExpander"] .stButton button {
    background-color: #EEEEEE;
    color: #212121;
    border: 1px solid #E0E0E0;
    text-transform: none;
    font-weight: 400;
    letter-spacing: 0;
    font-size: 0.85em;
    padding: 0.25rem 0.5rem;
}
[data-testid="stExpander"] .stButton > button:hover,
[data-testid="stExpander"] .stButton button:hover {
    background-color: #E3F2FD;
    border-color: #5B8DEF;
    color: #212121;
    box-shadow: none;
}
[data-testid="stExpander"] .stButton > button[kind="primary"],
[data-testid="stExpander"] .stButton button[kind="primary"] {
    background-color: #E3F2FD;
    border-color: #5B8DEF;
    color: #1565C0;
}

/* ── Success-style button ──────────────────────────── */
.success-btn > button {
    background-color: #00BFA5 !important;
    color: #FFFFFF !important;
    font-weight: 600;
}
.success-btn > button:hover {
    background-color: #00E5CC !important;
    box-shadow: 0 2px 8px rgba(0, 191, 165, 0.3) !important;
}

/* ── Chat messages ──────────────────────────────────── */
.stChatMessage {
    border-radius: 8px;
    border: 1px solid #E0E0E0;
    background-color: #F5F5F5 !important;
}
.stChatMessage:hover {
    background-color: #F5F5F5 !important;
    border-color: #E0E0E0 !important;
}
.stChatMessage:hover [data-testid="stExpander"],
.stChatMessage:hover [data-testid="stExpander"] details,
.stChatMessage:hover [data-testid="stExpander"] summary,
.stChatMessage:hover [data-testid="stExpanderDetails"] {
    background-color: #FAFAFA !important;
    color: #212121 !important;
}
.stChatMessage:hover .stCodeBlock,
.stChatMessage:hover .stCodeBlock *,
.stChatMessage:hover [data-testid="stCode"],
.stChatMessage:hover [data-testid="stCode"] * {
    background-color: #1E1E1E !important;
    color: #E0E0E0 !important;
}
/* Chat avatars — keep default styling in light mode */
/* Download buttons */
.stDownloadButton > button {
    background-color: #E0E0E0 !important;
    color: #37474F !important;
    border: 1px solid #BDBDBD !important;
}
.stDownloadButton > button * {
    color: #37474F !important;
}
.stDownloadButton > button:hover {
    background-color: #D0D0D0 !important;
    border-color: #9E9E9E !important;
}
/* ── Chat input ────────────────────────────────────── */
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] div {
    background-color: #FFFFFF !important;
    border-color: #D0D0D0 !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] textarea {
    background-color: #FFFFFF !important;
    border: none !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #9E9E9E !important;
}
[data-testid="stChatInput"] button {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}
[data-testid="stChatInput"] button[disabled] {
    background-color: #D0D0D0 !important;
    color: #9E9E9E !important;
}
[data-testid="stChatInput"] button[disabled] svg {
    fill: #9E9E9E !important;
}
[data-testid="stChatInput"] button:not([disabled]) {
    background-color: #6200EE !important;
    color: #FFFFFF !important;
}
[data-testid="stChatInput"] button:not([disabled]) svg {
    fill: #FFFFFF !important;
}
[data-testid="stChatInput"] button:not([disabled]):hover {
    background-color: #7C4DFF !important;
}
[data-testid="stBottom"],
[data-testid="stBottom"] > div {
    background-color: #FAFAFA !important;
}

/* ── Tabs ───────────────────────────────────────────── */
.stTabs [data-baseweb="tab"] {
    color: #212121;
    font-size: 1.05rem !important;
}
.stTabs [data-baseweb="tab"] * {
    font-size: 1.05rem !important;
}
.stTabs [aria-selected="true"] {
    color: #6200EE;
    border-bottom-color: #6200EE;
}

/* ── Expanders ──────────────────────────────────────── */
.streamlit-expanderHeader {
    color: #212121;
    font-size: 0.85em;
}
[data-testid="stExpander"],
[data-testid="stExpander"] details,
[data-testid="stExpander"] summary,
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    background-color: #FAFAFA !important;
    border-color: #E0E0E0 !important;
    color: #212121 !important;
}
[data-testid="stExpander"]:hover,
[data-testid="stExpander"]:hover details,
[data-testid="stExpander"]:hover summary,
[data-testid="stExpander"]:hover [data-testid="stExpanderDetails"] {
    background-color: #FAFAFA !important;
    border-color: #E0E0E0 !important;
    color: #212121 !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"],
section[data-testid="stSidebar"] [data-testid="stExpander"] details,
section[data-testid="stSidebar"] [data-testid="stExpander"] summary,
section[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    background-color: #F0F0F0 !important;
}

/* ── Text inputs & text areas ───────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
[data-testid="stChatInput"] textarea {
    background-color: #FFFFFF;
    color: #212121;
    caret-color: #212121 !important;
    border: 1px solid #BDBDBD;
    border-radius: 4px;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus,
[data-testid="stChatInput"] textarea:focus {
    border-color: #6200EE;
    box-shadow: 0 0 4px rgba(98, 0, 238, 0.2);
    caret-color: #6200EE !important;
}
/* ── Password visibility toggle ────────────────────── */
.stTextInput button svg {
    fill: #9E9E9E !important;
    stroke: #9E9E9E !important;
    color: #9E9E9E !important;
}
.stTextInput button {
    color: #9E9E9E !important;
    background-color: #FFFFFF !important;
}
.stTextInput [data-baseweb="input"],
.stTextInput [data-baseweb="base-input"] {
    background-color: #FFFFFF !important;
}

/* ── Select boxes ───────────────────────────────────── */
.stSelectbox > div > div {
    background-color: #FFFFFF;
    border: 1px solid #BDBDBD;
    border-radius: 4px;
}

/* ── Upload hero box ───────────────────────────────── */
.upload-hero-box {
    border: 2px dashed #BDBDBD;
    border-radius: 10px;
    padding: 32px 16px;
    text-align: center;
    margin-bottom: 16px;
    background: #F5F5F5;
}
.upload-hero-title {
    color: #6200EE;
    font-size: 1.1em;
    margin: 0 0 4px 0;
}
.upload-hero-subtitle {
    color: #757575;
    font-size: 0.85em;
    margin: 0;
}

/* ── File uploader ──────────────────────────────────── */
.stFileUploader,
.stFileUploader > div,
.stFileUploader section,
.stFileUploader [data-testid="stFileUploaderDropzone"] {
    background-color: #F5F5F5 !important;
    border-color: #BDBDBD !important;
}
.stFileUploader button {
    background-color: #E0E0E0 !important;
    color: #212121 !important;
    border: 1px solid #BDBDBD !important;
}
.stFileUploader button:hover {
    background-color: #D0D0D0 !important;
}
.stFileUploader > div {
    border: 1px dashed #BDBDBD;
    border-radius: 4px;
}

/* ── Code blocks ────────────────────────────────────── */
.stCodeBlock,
.stCodeBlock [data-testid="stCode"],
[data-testid="stCode"] {
    border: 1px solid #E0E0E0;
    border-radius: 4px;
    background-color: #1E1E1E !important;
}
.stCodeBlock code,
.stCodeBlock pre,
[data-testid="stCode"] code,
[data-testid="stCode"] pre,
.stCodeBlock *,
[data-testid="stCode"] * {
    color: #E0E0E0 !important;
    background-color: #1E1E1E !important;
}
.stCodeBlock:hover,
.stCodeBlock:hover *,
[data-testid="stCode"]:hover,
[data-testid="stCode"]:hover * {
    color: #E0E0E0 !important;
    background-color: #1E1E1E !important;
}

/* ── JSON viewer ───────────────────────────────────── */
[data-testid="stJson"],
[data-testid="stJson"] > div {
    background-color: #FFFFFF !important;
    color: #212121 !important;
    border: 1px solid #E0E0E0;
    border-radius: 4px;
}

/* ── Dataframe / table ─────────────────────────────── */
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] > div,
[data-testid="stTable"],
[data-testid="stTable"] > div {
    background-color: #FFFFFF !important;
    color: #212121 !important;
    border-radius: 4px;
}

/* ── Scrollbar (webkit) ─────────────────────────────── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #F5F5F5;
}
::-webkit-scrollbar-thumb {
    background: #BDBDBD;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #6200EE;
}

/* ── Headings ───────────────────────────────────────── */
h1 {
    color: #4A148C !important;
}
h2, h3 {
    color: #212121 !important;
}

/* ── Agent working spinner ──────────────────────────── */
@keyframes scilink-pulse {
    0%, 100% { opacity: 0.4; transform: scale(1); }
    50%      { opacity: 1;   transform: scale(1.05); }
}
.agent-spinner-container {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    background: linear-gradient(135deg, #F5F5F5 0%, #EEEEEE 100%);
    border: 1px solid #E0E0E0;
    border-left: 3px solid #6200EE;
    border-radius: 6px;
}
.agent-spinner-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #6200EE;
    animation: scilink-pulse 1.4s ease-in-out infinite;
    font-size: 0;
    line-height: 0;
    overflow: hidden;
}
.agent-spinner-dot:nth-child(2) { animation-delay: 0.2s; }
.agent-spinner-dot:nth-child(3) { animation-delay: 0.4s; }
.agent-spinner-heart {
    font-size: 1.1em;
    animation: scilink-pulse 1.4s ease-in-out infinite;
}
.agent-spinner-heart:nth-child(2) { animation-delay: 0.2s; }
.agent-spinner-heart:nth-child(3) { animation-delay: 0.4s; }
.agent-spinner-label {
    color: #212121;
    font-size: 0.9em;
    font-weight: 500;
}

/* ── Verbose output toggle — text label ────────────── */
.stApp [data-testid="stToggle"] label span,
[data-testid="stToggle"] label span {
    color: #212121 !important;
    font-size: 0.9em;
    font-weight: 600;
}
/* ── Stop button ───────────────────────────────────── */
[data-testid="stHorizontalBlock"]:has(.agent-spinner-container)
    > [data-testid="stColumn"]:last-child .stButton {
    padding-top: 10px;
}
/* Stop button — styled via JS MutationObserver below */

/* ── Live log viewer ────────────────────────────────── */
.live-log-viewer {
    height: 280px;
    overflow-y: auto;
    margin: 0;
    background: #FAFAFA;
    padding: 8px;
    border-radius: 4px;
    border: 1px solid #E0E0E0;
    font-family: monospace;
    font-size: 13px;
    white-space: pre-wrap;
    color: #212121;
}

/* ── Hide Streamlit chrome (deploy, menu, stop) ────── */
.stDeployButton,
[data-testid="stAppDeployButton"],
#MainMenu,
[data-testid="stMainMenu"],
header [data-testid="stStatusWidget"] {
    display: none !important;
}

/* ── Floating background emojis ───────────────────── */
.floating-emojis {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
    animation: emojis-fade-in 2s ease-out forwards;
}
@keyframes emojis-fade-in {
    from { opacity: 0; }
    to   { opacity: 1; }
}
.floating-emojis span {
    position: absolute;
    display: block;
    font-size: var(--emoji-size, 28px);
    opacity: 0;
    animation: emoji-float var(--duration, 18s) var(--delay, 0s) ease-in-out infinite;
}
.floating-emojis span.rocket {
    animation: rocket-float var(--duration, 18s) var(--delay, 0s) linear infinite;
}
@keyframes emoji-float {
    0% {
        opacity: 0;
        transform: translateY(100vh) rotate(0deg);
    }
    3% {
        opacity: var(--peak-opacity, 0.12);
    }
    93% {
        opacity: var(--peak-opacity, 0.12);
    }
    100% {
        opacity: 0;
        transform: translateY(-10vh) rotate(var(--rotation, 360deg));
    }
}
@keyframes ufo-zip {
    0% {
        opacity: 0;
        transform: translateX(-10vw) translateY(0);
    }
    2% {
        opacity: var(--peak-opacity, 0.18);
    }
    50% {
        transform: translateX(50vw) translateY(-30px);
    }
    98% {
        opacity: var(--peak-opacity, 0.18);
    }
    100% {
        opacity: 0;
        transform: translateX(110vw) translateY(0);
    }
}
.floating-emojis span.ufo {
    bottom: auto;
    animation: ufo-zip var(--duration, 40s) var(--delay, 20s) linear infinite;
}
@keyframes rocket-float {
    0% {
        opacity: 0;
        transform: translateY(100vh) rotate(-45deg);
    }
    3% {
        opacity: var(--peak-opacity, 0.12);
    }
    93% {
        opacity: var(--peak-opacity, 0.12);
    }
    100% {
        opacity: 0;
        transform: translateY(-10vh) rotate(-45deg);
    }
}
</style>
"""

_FLOATING_HTML = """
<div class="floating-emojis" aria-hidden="true">
{spans}
</div>
"""


def _build_positivity_spans(n_hearts: int = 7, n_pluses: int = 7) -> str:
    """Generate hearts and pluses spans."""
    import random

    heart_emojis = [
        "\U0001f49c",   # 💜 purple
        "\u2764\ufe0f", # ❤️ red
        "\U0001f499",   # 💙 blue
    ]
    plus_colors = ["#00BCD4", "#FFD600", "#E040FB"]  # teal, yellow, magenta

    items: list[tuple[str, object]] = []
    for _ in range(n_hearts):
        items.append(("heart", random.choice(heart_emojis)))
    for _ in range(n_pluses):
        items.append(("plus", random.choice(plus_colors)))
    random.shuffle(items)

    spans: list[str] = []
    for kind, value in items:
        left = random.randint(2, 96)
        size = random.randint(20, 40)
        duration = round(random.uniform(16, 30), 1)
        delay = round(random.uniform(0, 3), 1)
        if kind == "heart":
            rotation = random.choice([-360, -180, 180, 360])
            opacity = round(random.uniform(0.10, 0.20), 2)
            spans.append(
                f'<span style="left:{left}%;'
                f"--emoji-size:{size}px;"
                f"--duration:{duration}s;"
                f"--delay:{delay}s;"
                f"--rotation:{rotation}deg;"
                f'--peak-opacity:{opacity}">{value}</span>'
            )
        else:
            opacity = round(random.uniform(0.12, 0.22), 2)
            spans.append(
                f'<span style="left:{left}%;'
                f"--emoji-size:{size}px;"
                f"--duration:{duration}s;"
                f"--delay:{delay}s;"
                f"--rotation:0deg;"
                f"--peak-opacity:{opacity};"
                f"color:{value};"
                f'font-weight:900">\u271A</span>'
            )
    return "\n".join(spans)


def _build_space_spans(n_rockets: int = 7, n_ufos: int = 1) -> str:
    """Generate rocket and UFO spans."""
    import random

    spans: list[str] = []
    for _ in range(n_rockets):
        left = random.randint(2, 96)
        size = random.randint(20, 40)
        duration = round(random.uniform(16, 30), 1)
        delay = round(random.uniform(0, 3), 1)
        opacity = round(random.uniform(0.10, 0.20), 2)
        spans.append(
            f'<span class="rocket" style="left:{left}%;'
            f"--emoji-size:{size}px;"
            f"--duration:{duration}s;"
            f"--delay:{delay}s;"
            f"--rotation:0deg;"
            f'--peak-opacity:{opacity}">\U0001f680</span>'
        )
    for i in range(n_ufos):
        top = random.randint(10, 80)
        size = random.randint(24, 36)
        duration = round(random.uniform(35, 55), 1)
        delay = round(random.uniform(0, 5) if i == 0 else random.uniform(15, 45), 1)
        opacity = round(random.uniform(0.15, 0.25), 2)
        spans.append(
            f'<span class="ufo" style="top:{top}%;left:0;'
            f"--emoji-size:{size}px;"
            f"--duration:{duration}s;"
            f"--delay:{delay}s;"
            f'--peak-opacity:{opacity}">\U0001f6f8</span>'
        )
    return "\n".join(spans)


_COLLISION_JS = """
<script>
(function() {
    const doc = window.parent.document;
    const container = doc.querySelector('.floating-emojis');
    if (!container) return;
    function rectsOverlap(a, b) {
        return !(a.right < b.left || a.left > b.right || a.bottom < b.top || a.top > b.bottom);
    }
    function boom(x, y) {
        const el = doc.createElement('div');
        el.textContent = '\U0001f4a5';
        el.style.cssText =
            'position:fixed;z-index:10000;pointer-events:none;font-size:48px;' +
            'left:' + x + 'px;top:' + y + 'px;transform:translate(-50%,-50%);' +
            'animation:boom-fade 0.8s ease-out forwards;';
        container.appendChild(el);
        setTimeout(function() { el.remove(); }, 900);
    }
    // Inject boom keyframes once
    if (!doc.getElementById('boom-style')) {
        const s = doc.createElement('style');
        s.id = 'boom-style';
        s.textContent = '@keyframes boom-fade{0%{opacity:1;transform:translate(-50%,-50%) scale(1)}100%{opacity:0;transform:translate(-50%,-50%) scale(2.5)}}';
        doc.head.appendChild(s);
    }
    const cooldowns = new WeakMap();
    setInterval(function() {
        const rockets = container.querySelectorAll('.rocket');
        const ufos = container.querySelectorAll('.ufo');
        rockets.forEach(function(r) {
            if (cooldowns.get(r) > Date.now()) return;
            const rr = r.getBoundingClientRect();
            if (rr.width === 0) return;
            ufos.forEach(function(u) {
                if (cooldowns.get(u) > Date.now()) return;
                const ur = u.getBoundingClientRect();
                if (ur.width === 0) return;
                if (rectsOverlap(rr, ur)) {
                    const cx = (rr.left + rr.right + ur.left + ur.right) / 4;
                    const cy = (rr.top + rr.bottom + ur.top + ur.bottom) / 4;
                    boom(cx, cy);
                    var fate = Math.floor(Math.random() * 3);
                    if (fate === 0 || fate === 2) { u.style.display = 'none'; }
                    if (fate === 1 || fate === 2) { r.style.display = 'none'; }
                    return;
                }
            });
        });
    }, 150);
})();
</script>
"""


def inject_theme() -> None:
    """Inject the Material Design CSS into the current page."""
    import streamlit.components.v1 as components

    theme_mode = st.session_state.get("theme_mode", "dark")
    css = _MATERIAL_CSS_DARK if theme_mode == "dark" else _MATERIAL_CSS_LIGHT
    st.markdown(css, unsafe_allow_html=True)

    # Force-restyle widgets via JS injected through components.html()
    # (st.markdown strips <script> tags; components.html() runs in an
    # iframe so we reach the parent document via window.parent.document).
    if theme_mode == "light":
        _LIGHT_WIDGET_JS = """<script>
(function(){
    var doc = window.parent.document;
    function isLightMode(){
        /* Check if light-mode CSS is active by looking for #FAFAFA background */
        var app = doc.querySelector('.stApp');
        if (!app) return false;
        var bg = getComputedStyle(app).backgroundColor;
        return bg.indexOf('250') >= 0; /* rgb(250,250,250) = #FAFAFA */
    }
    function styleWidgets(){
        if (!isLightMode()) return; /* Stop if switched to dark mode */
        /* Stop button — find by content or tooltip, skip sidebar */
        doc.querySelectorAll('button[kind="secondary"]').forEach(function(btn){
            if (btn.closest('[data-testid="stSidebar"]')) return;
            var tip = btn.getAttribute('title') || '';
            var txt = (btn.textContent || '').trim();
            if (tip.indexOf('Stop') >= 0 || txt === '\\u25A0') {
                btn.style.setProperty('background', 'linear-gradient(135deg,#F5F5F5 0%,#EEEEEE 100%)', 'important');
                btn.style.setProperty('color', '#37474F', 'important');
                btn.style.setProperty('border', '1px solid #E0E0E0', 'important');
                btn.style.setProperty('border-radius', '6px', 'important');
            }
        });
    }
    styleWidgets();
    new MutationObserver(styleWidgets)
        .observe(doc.body, {childList:true, subtree:true, attributes:true});
})();
</script>"""
        components.html(_LIGHT_WIDGET_JS, height=0)
    else:
        # Keep DOM slot count stable; also undo any inline styles
        # left by the light-mode observer on the stop button.
        components.html("""<script>
(function(){
    var doc = window.parent.document;
    function cleanup(){
        doc.querySelectorAll('button[kind="secondary"]').forEach(function(btn){
            if (btn.closest('[data-testid="stSidebar"]')) return;
            btn.style.removeProperty('background');
            btn.style.removeProperty('color');
            btn.style.removeProperty('border');
            btn.style.removeProperty('border-radius');
        });
    }
    cleanup();
    new MutationObserver(cleanup)
        .observe(doc.body, {childList:true, subtree:true});
})();
</script>""", height=0)

    vibe = st.session_state.get("vibe_theme", "Professional")

    # Always emit both slots so switching vibes doesn't shift the layout.
    floating_spans = ""
    use_collision_js = False

    if vibe == "Positivity boost":
        n_hearts = st.session_state.get("vibe_hearts", 7)
        n_pluses = st.session_state.get("vibe_pluses", 7)
        if n_hearts or n_pluses:
            floating_spans = _build_positivity_spans(n_hearts, n_pluses)
    elif vibe == "Space nerd":
        n_rockets = st.session_state.get("vibe_rockets", 7)
        n_ufos = st.session_state.get("vibe_ufos", 1)
        if n_rockets or n_ufos:
            floating_spans = _build_space_spans(n_rockets, n_ufos)
        if n_rockets and n_ufos:
            use_collision_js = True

    # Slot 1: floating emojis (always present, may be empty)
    st.markdown(
        _FLOATING_HTML.format(spans=floating_spans),
        unsafe_allow_html=True,
    )
    # Slot 2: collision JS (always present, no-op script when inactive)
    components.html(
        _COLLISION_JS if use_collision_js else "<script>/* noop */</script>",
        height=0,
    )
