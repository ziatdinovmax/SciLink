"""HTTP backend for the SciLink React web frontend (``scilink-web``).

A headless re-implementation of what the Streamlit UI does around the
orchestrators: session lifecycle, one background thread per chat turn with
stdout/logging capture, human-in-the-loop question parking, artifact sweeps,
and file serving — exposed as REST + SSE instead of Streamlit reruns. The
orchestrators themselves are consumed exactly as Streamlit consumes them
(``agent.chat()`` / ``restore_from_checkpoint``); nothing in the agent stack
changes.
"""
