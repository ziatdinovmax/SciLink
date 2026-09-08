import { useCallback, useEffect, useState } from "react";
import { api, type ToolInventory } from "../api";

/** MCP tab — connect MCP servers to the session's agent and disconnect
 * them; each server card lists the tools it registered. Nothing else: the
 * agent's own built-in tools are not listed (in a meta session that would
 * be only the meta's routing tools, not what the specialists can do), and
 * there is deliberately no tool-file uploader — users hand code to the
 * agents as scripts attached in chat (adapted by codegen) or as MCP
 * servers (run verbatim, any language). */

type Transport = "stdio" | "sse" | "http";

export function ToolsPanel({
  sessionId,
  active,
  localFiles,
}: {
  sessionId: string;
  active: boolean;
  localFiles: boolean;
}) {
  const [inv, setInv] = useState<ToolInventory | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [transport, setTransport] = useState<Transport>("stdio");
  const [name, setName] = useState("");
  const [addr, setAddr] = useState("");
  const [headers, setHeaders] = useState("");

  const refresh = useCallback(() => {
    api.tools(sessionId).then(setInv).catch((e) => setError(String(e)));
  }, [sessionId]);

  useEffect(() => {
    if (active) refresh();
  }, [active, refresh]);

  const connect = async () => {
    setError(null);
    let hdrs: Record<string, string> | undefined;
    if (transport !== "stdio" && headers.trim()) {
      try {
        const parsed = JSON.parse(headers) as unknown;
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed))
          throw new Error("must be a JSON object");
        hdrs = parsed as Record<string, string>;
      } catch (e) {
        setError(`Headers ${e instanceof Error ? e.message : String(e)}`);
        return;
      }
    }
    setBusy(`Connecting to ${name}…`);
    try {
      const r = await api.connectMcp(sessionId, {
        name: name.trim(),
        transport,
        command: transport === "stdio" ? addr.trim() : undefined,
        url: transport !== "stdio" ? addr.trim() : undefined,
        headers: hdrs,
      });
      setInv(r.inventory);
      setName("");
      setAddr("");
      setHeaders("");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const disconnect = async (server: string) => {
    setBusy(`Disconnecting ${server}…`);
    try {
      const r = await api.disconnectMcp(sessionId, server);
      setInv(r.inventory);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const mcpToolNames = new Set((inv?.mcp_servers ?? []).flatMap((s) => s.tools));
  const otherExternal = (inv?.external ?? []).filter((t) => !mcpToolNames.has(t.name));

  return (
    <div className="tools-panel">
      <section className="tools-section">
        <h3>MCP servers</h3>
        <p className="caption">
          Extend the agent with external services — instrument APIs, databases,
          computational tools, other agents. MCP is the open standard for it;
          any language, no SciLink-specific format.
          {!localFiles && (
            <>
              {" "}
              On this shared server a <code>stdio</code> command runs on the server
              machine.
            </>
          )}
        </p>
        {inv?.mcp_supported === false ? (
          <p className="caption warn">This session's agent does not support MCP servers.</p>
        ) : (
          <form
            className="mcp-form"
            onSubmit={(e) => {
              e.preventDefault();
              if (name.trim() && addr.trim()) void connect();
            }}
          >
            <div className="mcp-transport">
              {(["stdio", "sse", "http"] as Transport[]).map((t) => (
                <label key={t}>
                  <input
                    type="radio"
                    name="mcp-transport"
                    checked={transport === t}
                    onChange={() => setTransport(t)}
                  />{" "}
                  {t}
                </label>
              ))}
              <span className="caption">
                {transport === "http"
                  ? "streamable HTTP — remote / hosted servers"
                  : transport === "sse"
                    ? "server-sent events endpoint"
                    : "a local command (subprocess)"}
              </span>
            </div>
            <div className="mcp-row">
              <input
                type="text"
                placeholder="Server name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                disabled={busy !== null}
              />
              <input
                type="text"
                className="mcp-addr"
                placeholder={
                  transport === "stdio"
                    ? "npx -y @modelcontextprotocol/server-name /path"
                    : transport === "sse"
                      ? "http://localhost:8080/sse"
                      : "https://host/mcp"
                }
                value={addr}
                onChange={(e) => setAddr(e.target.value)}
                disabled={busy !== null}
              />
              <button className="primary" type="submit" disabled={busy !== null || !name.trim() || !addr.trim()}>
                {busy && busy.startsWith("Connecting") ? "Connecting…" : "Connect"}
              </button>
            </div>
            {transport !== "stdio" && (
              <input
                type="text"
                placeholder='Headers (JSON, optional) — {"Authorization": "Bearer ${MY_API_KEY}"}'
                value={headers}
                onChange={(e) => setHeaders(e.target.value)}
                disabled={busy !== null}
              />
            )}
          </form>
        )}
        {error && <p className="caption warn">{error}</p>}
        {inv && inv.mcp_servers.length === 0 && (
          <p className="caption">No MCP servers connected.</p>
        )}
        {inv?.mcp_servers.map((s) => (
          <div key={s.name} className="mcp-server">
            <div className="mcp-server-head">
              <strong>{s.name}</strong>
              <span className="caption">
                {" "}
                · {s.transport} · {s.tools.length} tool{s.tools.length === 1 ? "" : "s"}
              </span>
              <button
                type="button"
                className="icon-btn"
                title={`Disconnect ${s.name}`}
                disabled={busy !== null}
                onClick={() => void disconnect(s.name)}
              >
                ✕
              </button>
            </div>
            <div className="tool-chips">
              {s.tools.map((t) => (
                <span key={t} className="file-chip" title={inv.external.find((e) => e.name === t)?.description}>
                  {t}
                </span>
              ))}
            </div>
          </div>
        ))}
      </section>

      {otherExternal.length > 0 && (
        <section className="tools-section">
          <h3>Registered external tools</h3>
          <ul className="tool-list">
            {otherExternal.map((t) => (
              <li key={t.name}>
                <code>{t.name}</code>
                <span className="caption"> — {t.description}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

    </div>
  );
}
