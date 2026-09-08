import { useState } from "react";

/** Access-token sign-in for a shared scilink-web server. The token is
 * exchanged for an HttpOnly cookie by the backend (never stored by the
 * page); a link of the form `/?token=…` signs in without this screen. */
export function LoginScreen({
  onLogin,
  error,
  busy,
}: {
  onLogin: (token: string) => void;
  error: string | null;
  busy: boolean;
}) {
  const [token, setToken] = useState("");
  return (
    <div className="welcome">
      <h2>Sign in to SciLink</h2>
      <p className="tagline">
        This server requires an access token. Ask the person who runs it, or
        open the sign-in link they gave you.
      </p>
      <form
        className="login-form"
        onSubmit={(e) => {
          e.preventDefault();
          if (token.trim()) onLogin(token.trim());
        }}
      >
        <input
          type="password"
          autoFocus
          autoComplete="off"
          placeholder="Access token"
          value={token}
          onChange={(e) => setToken(e.target.value)}
          disabled={busy}
        />
        <button className="primary" type="submit" disabled={busy || !token.trim()}>
          {busy ? "Signing in…" : "Sign in"}
        </button>
      </form>
      {error && <p className="caption warn">{error}</p>}
    </div>
  );
}
