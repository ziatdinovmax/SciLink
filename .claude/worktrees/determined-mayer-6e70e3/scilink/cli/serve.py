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
        choices=["analyze", "plan", "both"],
        default="both",
        help="Which tool sets to expose (default: both)",
    )
    parser.add_argument(
        "--autonomy",
        type=str,
        choices=["autonomous", "supervised", "co-pilot"],
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

    args = parser.parse_args()

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
