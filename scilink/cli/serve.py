"""CLI entry point for the SciLink MCP server.

Usage::

    scilink serve                                      # defaults
    scilink serve --model gemini-3.1-pro-preview       # specific model
    scilink serve --mode analyze                       # analysis tools only
    scilink serve --autonomy co-pilot                  # require approval
    scilink serve --transport sse --port 8000           # SSE transport
"""

import argparse
import os
import sys
from pathlib import Path


def _print_mcp_json(args) -> int:
    """Emit a client config entry for this server invocation.

    The entry is secret-free by design: the server reads
    ``~/.scilink/credentials.env`` at startup, so client configs never need
    an env block. When ``uvx`` is on PATH the spec is zero-install (the
    client machine needs uv, nothing else); otherwise it points at this
    installation's ``scilink`` executable.
    """
    import json
    import shutil

    try:
        from importlib.metadata import version as _pkg_version
        _v = _pkg_version("scilink")
    except Exception:  # noqa: BLE001 - editable/dev installs
        _v = None

    serve_args = ["serve", "--mode", args.mode, "--autonomy", args.autonomy]
    if args.model:
        serve_args += ["--model", args.model]
    if args.session_dir:
        serve_args += ["--session-dir", args.session_dir]
    if args.transport == "sse":
        entry = {"type": "sse", "url": f"http://{args.host}:{args.port}/sse"}
        note = (f"# Start the server yourself:\n"
                f"#   scilink serve {' '.join(serve_args[1:])} "
                f"--transport sse --host {args.host} --port {args.port}")
    elif shutil.which("uvx"):
        entry = {"command": "uvx",
                 "args": ["--from",
                          f"scilink=={_v}" if _v else "scilink",
                          "scilink"] + serve_args}
        note = "# Zero-install spec: the client machine only needs uv."
    else:
        exe = shutil.which("scilink") or sys.argv[0]
        entry = {"command": exe, "args": serve_args}
        note = "# Points at this machine's scilink executable."
    cred = Path.home() / ".scilink" / "credentials.env"
    print(note, file=sys.stderr)
    print(f"# LLM credentials: put KEY=VALUE lines in {cred}", file=sys.stderr)
    print(f"# (loaded by the server at startup; {'present' if cred.exists() else 'NOT present yet'})",
          file=sys.stderr)
    print(json.dumps({"mcpServers": {"scilink": entry}}, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="scilink serve",
        description="Start SciLink as an MCP tool server.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("SCILINK_MODEL", "gemini-3.1-pro-preview"),
        help="LLM model name (default: gemini-3.1-pro-preview)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (default: auto-detect from env vars)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="OpenAI-compatible endpoint URL",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["analyze", "plan", "simulate", "both", "meta"],
        default="both",
        help=(
            "Which tool sets to expose (default: both, i.e. analyze+plan). "
            "'simulate' exposes the simulation tools (structure building, "
            "engine inputs, run + refine; DFT/MD/MLIP) as a standalone "
            "surface. 'meta' exposes SciLink's meta orchestrator instead — "
            "one surface that routes across the specialists (analysis, "
            "planning, simulation), with parallel multi-modal fan-out + "
            "fusion and cross-mode delegation."
        ),
    )
    parser.add_argument(
        "--autonomy",
        type=str,
        choices=["autonomous", "autopilot", "co-pilot"],
        default="autonomous",
        help="Autonomy level (default: autonomous)",
    )
    parser.add_argument(
        "--session-dir",
        type=str,
        default=None,
        help="Session directory for outputs (default: auto-generated)",
    )
    parser.add_argument(
        "--hitl-timeout",
        type=float,
        default=1800.0,
        help=(
            "Seconds an agent question waits for scilink_respond before "
            "falling back to its default answer (default: 1800)"
        ),
    )
    parser.add_argument(
        "--transport",
        type=str,
        choices=["stdio", "sse"],
        default="stdio",
        help="MCP transport (default: stdio)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Bind address for SSE transport (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Bind port for SSE transport (default: 8000)",
    )
    parser.add_argument(
        "--futurehouse-key",
        type=str,
        default=None,
        help="FutureHouse/Edison API key for novelty assessment",
    )
    parser.add_argument(
        "--print-mcp-json",
        action="store_true",
        help=(
            "Print a ready-to-paste .mcp.json entry for this server "
            "configuration (Claude Desktop / Deep Agents style) and exit. "
            "Uses a zero-install uvx spec when uvx is available; contains "
            "no secrets — put LLM credentials in ~/.scilink/credentials.env "
            "(KEY=VALUE lines), which the server loads at startup."
        ),
    )

    args = parser.parse_args()

    if args.print_mcp_json:
        return _print_mcp_json(args)

    # Resolve API key
    api_key = args.api_key
    if api_key is None:
        for env_var in [
            "SCILINK_API_KEY",
            "GEMINI_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
        ]:
            api_key = os.environ.get(env_var)
            if api_key:
                break

    # The MCP stdio transport reads sys.stdin.buffer and writes
    # sys.stdout.buffer.  Save the real stdout before redirecting
    # Python-level sys.stdout to stderr, so print() calls from
    # orchestrator init / tool execution go to stderr instead of
    # corrupting the JSON-RPC stream.
    import logging
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    _real_stdout = sys.stdout
    sys.stdout = sys.stderr

    try:
        from scilink.mcp_server import create_server, run_stdio, run_sse
    except ImportError as exc:
        print(
            f"Error: {exc}\n"
            "Install MCP support with: pip install scilink",
            file=sys.stderr,
        )
        return 1

    server = create_server(
        api_key=api_key,
        model_name=args.model,
        base_url=args.base_url,
        mode=args.mode,
        session_dir=args.session_dir,
        analysis_mode=args.autonomy,
        futurehouse_api_key=args.futurehouse_key,
        hitl_timeout_s=args.hitl_timeout,
    )

    # Initialize orchestrators eagerly so tools/list responds instantly.
    # Claude Desktop times out after ~5 seconds.
    print(f"Initializing SciLink MCP server (mode={args.mode}, autonomy={args.autonomy})...",
          file=sys.stderr)
    server.eager_init()

    transport_label = (f"SSE on http://{args.host}:{args.port}/sse"
                       if args.transport == "sse" else "stdio")
    print(f"SciLink MCP server ready ({transport_label}). "
          f"Waiting for MCP client connections...",
          file=sys.stderr, flush=True)

    sys.stderr.flush()

    if args.transport == "sse":
        run_sse(server, host=args.host, port=args.port)
    else:
        import asyncio
        asyncio.run(run_stdio(server, real_stdout=_real_stdout))

    return 0


if __name__ == "__main__":
    sys.exit(main())
