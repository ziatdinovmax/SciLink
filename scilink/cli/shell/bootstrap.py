"""Getting a session ready — one copy of what four CLIs each did.

Credentials, the session directory, the code-execution consent and the
registration of custom skills / tools / agents / MCP servers. Nothing here
prints raw ``input()`` prompts: every question goes through the ``ask``
callable the shell supplies (a prompt_toolkit prompt in the terminal, a
no-op that returns the default in headless mode).
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

from scilink import auth
from scilink.providers import provider_for
from scilink.ui import vocabulary as V

# ask(prompt, *, secret=False, default="") -> str
Ask = Callable[..., str]


class BootstrapError(RuntimeError):
    """A condition the shell reports and exits on (no traceback)."""


# ── Credentials ──────────────────────────────────────────────────

@dataclass
class Credentials:
    api_key: Optional[str]
    base_url: Optional[str]
    provider: str      # display name of the provider
    source: str        # where the key came from: a flag, an env var, a prompt, auto


def resolve_llm_credentials(model: str, base_url: Optional[str],
                            api_key: Optional[str], ask: Ask) -> Credentials:
    """The four CLIs' key-resolution blocks, merged.

    A ``base_url`` means the internal proxy: the key is ``SCILINK_API_KEY``
    (never a vendor key). Otherwise the vendor's conventional env var is
    used when set; Bedrock accepts a bearer token or the default AWS
    credential chain; and when nothing is set the user is asked once, with
    Enter meaning "let LiteLLM auto-detect".
    """
    spec = provider_for(model)
    if api_key:
        return Credentials(api_key, base_url, spec.name, "flag")
    if base_url:
        key = auth.get_internal_proxy_key()
        if key:
            return Credentials(key, base_url, spec.name, auth.INTERNAL_PROXY_KEY)
        key = ask(f"Proxy API key ({auth.INTERNAL_PROXY_KEY}): ", secret=True).strip()
        if not key:
            raise BootstrapError(
                "An API key is required for the internal proxy: set "
                f"{auth.INTERNAL_PROXY_KEY} or pass --api-key.")
        return Credentials(key, base_url, spec.name, "prompt")
    if model.startswith("bedrock/"):
        if os.getenv("AWS_BEARER_TOKEN_BEDROCK") or os.getenv("AWS_ACCESS_KEY_ID"):
            return Credentials(None, None, spec.name, "AWS environment")
        token = ask("Bedrock bearer token (Enter = default AWS credential "
                    "discovery): ", secret=True).strip()
        if token:
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = token
            return Credentials(None, None, spec.name, "prompt")
        return Credentials(None, None, spec.name, "auto")
    kv = auth.find_env_var_for_model(model)
    if kv:
        return Credentials(kv[1], None, spec.name, kv[0])
    key = ask(f"{spec.name} {spec.key_label} (Enter = let LiteLLM auto-detect): ",
              secret=True).strip()
    return Credentials(key or None, None, spec.name, "prompt" if key else "auto")


def resolve_optional_key(flag_value: Optional[str], *env_vars: str) -> Optional[str]:
    """A flag wins; otherwise the first env var that is set; never a prompt."""
    if flag_value:
        return flag_value
    for var in env_vars:
        v = os.getenv(var)
        if v:
            return v
    return None


# ── Session directory ────────────────────────────────────────────

def session_root() -> Path:
    """Where NEW sessions go: the current folder, or SCILINK_SESSION_ROOT for
    a central store. Either way every session is registered in the index."""
    env = os.environ.get("SCILINK_SESSION_ROOT")
    return Path(env).expanduser() if env else Path(".")


def new_session_dir(mode: str, root: Optional[Path] = None) -> Path:
    prefix = V.mode(mode)["session_prefix"]
    return (root if root is not None else session_root()) / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}"


# ── Code-execution consent ───────────────────────────────────────

def sandbox_already_approved() -> bool:
    from scilink import executors
    return bool(executors._GLOBAL_SANDBOX_APPROVED)


def ensure_sandbox_consent(*, assume_yes: bool, ask: Ask, console) -> bool:
    """Ask once, up front, with the same sentence the web UI's checkbox
    shows; a detected sandbox or an explicit override needs no question.

    Returns True when generated code may run. A decline is not fatal: the
    agents ask again at execution time, through the shell's question
    widget, if a turn actually needs to run code.
    """
    from scilink import executors

    if executors._GLOBAL_SANDBOX_APPROVED:
        return True
    if (assume_yes
            or os.environ.get("UNSAFE_EXECUTION_OK", "").lower() == "true"
            or os.environ.get("SCILINK_ACCEPT_CODE_EXECUTION", "").lower() in ("1", "true", "yes")):
        executors._GLOBAL_SANDBOX_APPROVED = True
        return True
    try:
        score, indicators = executors.check_security_sandbox_indicators(verbose=False)
    except Exception:  # noqa: BLE001 - detection must never block startup
        score, indicators = 0, []
    if score >= 4:
        executors._GLOBAL_SANDBOX_APPROVED = True
        console.print(f"[dim]Sandbox detected ({indicators[0] if indicators else 'sandbox'}); "
                      "generated code runs inside it.[/]")
        return True
    console.print(f"\n[bold]{V.CONSENT_TEXT}[/]")
    console.print("[dim]No sandbox detected — a Docker container, VM or Colab is "
                  "recommended. Answer no to decide later, per run.[/]")
    answer = ask("Allow code execution? [y/N]: ").strip().lower()
    if answer in ("y", "yes"):
        executors._GLOBAL_SANDBOX_APPROVED = True
        return True
    console.print("[dim]Code execution not pre-approved; you will be asked when a run needs it.[/]")
    return False


# ── Custom extras: skills, tools, agents, MCP ────────────────────

def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    prev = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = prev
    return module


def register_skill_files(agent, files: Iterable[str], console) -> None:
    for file_path in files:
        path = Path(file_path).expanduser().resolve()
        try:
            name = agent.register_skill(str(path))
            console.print(f"  [green]✓[/] skill [bold]{name}[/] registered")
        except Exception as e:  # noqa: BLE001
            console.print(f"  [red]✗[/] {path.name}: {e}")


def register_tool_files(agent, files: Iterable[str], console) -> None:
    """Load OpenAI-format tool schemas + a factory from each .py file.

    A file exposes (1) a list of tool schemas — ``tool_schemas`` /
    ``openai_schemas``, or any top-level list of ``{"type": "function"}``
    dicts — and (2) a factory: ``create_tool_functions`` or any function
    ending in ``_tool_functions`` returning ``{name: callable}``.
    """
    for file_path in files:
        path = Path(file_path).resolve()
        try:
            module = _load_module(path)
        except Exception as e:  # noqa: BLE001
            console.print(f"  [red]✗[/] {path.name}: {e}")
            continue
        schemas = getattr(module, "tool_schemas", None) or getattr(module, "openai_schemas", None)
        if schemas is None:
            for attr in dir(module):
                obj = getattr(module, attr, None)
                if (isinstance(obj, list) and obj and isinstance(obj[0], dict)
                        and obj[0].get("type") == "function"):
                    schemas = obj
                    break
        if not schemas:
            console.print(f"  [yellow]![/] {path.name}: no tool schemas found "
                          "(define 'tool_schemas' as a list of OpenAI-format tool dicts)")
            continue
        factory = getattr(module, "create_tool_functions", None)
        if factory is None:
            for name, fn in inspect.getmembers(module, inspect.isfunction):
                if name.endswith("_tool_functions") and fn.__module__ == module.__name__:
                    factory = fn
                    break
        if factory is None:
            console.print(f"  [yellow]![/] {path.name}: no factory found "
                          "(define 'create_tool_functions(data)' returning {name: callable})")
            continue
        agent.register_tools(schemas, factory)
        count = sum(1 for s in schemas if s.get("type") == "function")
        console.print(f"  [green]✓[/] {count} tool(s) from [bold]{path.name}[/]")


def register_agent_files(agent, files: Iterable[str], console) -> None:
    """Analyze mode: register BaseAnalysisAgent subclasses from .py files."""
    from scilink.agents.exp_agents.base_agent import BaseAnalysisAgent

    for file_path in files:
        path = Path(file_path).resolve()
        try:
            module = _load_module(path)
        except Exception as e:  # noqa: BLE001
            console.print(f"  [red]✗[/] {path.name}: {e}")
            continue
        found = 0
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if (issubclass(cls, BaseAnalysisAgent) and cls is not BaseAnalysisAgent
                    and cls.__module__ == module.__name__):
                next_id = max(agent._agent_registry.keys()) + 1
                agent.register_agent(next_id, cls)
                console.print(f"  [green]✓[/] agent [bold]{cls.__name__}[/] registered as ID {next_id}")
                found += 1
        if not found:
            console.print(f"  [yellow]![/] {path.name}: no BaseAnalysisAgent subclasses found")


def parse_mcp_entry(entry: str) -> dict:
    """One ``--mcp`` / ``/mcp`` entry → ``connect_mcp_server`` kwargs.

    ``stdio:name:cmd,arg1,arg2`` · ``sse:name:url`` · ``http:name:url`` ·
    or a JSON config file ({name, command, url, transport, headers, env};
    ``${VAR}`` in header values is expanded from the environment).
    """
    for scheme, transport in (("stdio:", None), ("sse:", None), ("http:", "http")):
        if entry.startswith(scheme):
            name, _, rest = entry[len(scheme):].partition(":")
            if scheme == "stdio:":
                return {"name": name, "command": rest.split(",") if rest else []}
            return {"name": name, "url": rest, "transport": transport}
    path = Path(entry).resolve()
    with open(path) as f:
        cfg = json.load(f)
    headers = cfg.get("headers")
    if headers:
        headers = {k: os.path.expandvars(v) for k, v in headers.items()}
    return {"name": cfg.get("name", path.stem), "command": cfg.get("command"),
            "url": cfg.get("url"), "env": cfg.get("env"),
            "transport": cfg.get("transport"), "headers": headers}


def connect_mcp_entries(agent, entries: Iterable[str], console) -> None:
    if not hasattr(agent, "connect_mcp_server"):
        console.print("  [yellow]![/] this mode's agent does not support MCP servers")
        return
    for entry in entries:
        try:
            kwargs = parse_mcp_entry(entry)
            name = kwargs.pop("name")
            count = agent.connect_mcp_server(name, **{k: v for k, v in kwargs.items()
                                                      if v is not None})
            console.print(f"  [green]✓[/] {count} tool(s) from MCP server [bold]{name}[/]")
        except Exception as e:  # noqa: BLE001
            console.print(f"  [red]✗[/] MCP {entry!r}: {e}")


def register_extras(agent, *, skill_files=(), tool_files=(), agent_files=(),
                    mcp_entries=(), console) -> None:
    if agent_files:
        register_agent_files(agent, agent_files, console)
    if tool_files:
        register_tool_files(agent, tool_files, console)
    if skill_files:
        register_skill_files(agent, skill_files, console)
    if mcp_entries:
        connect_mcp_entries(agent, mcp_entries, console)
