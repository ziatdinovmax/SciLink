import { createContext, useContext } from "react";

/** Cosmetic "vibes" ported from the Streamlit UI: a background layer of
 * floating emojis and a themed working-spinner. Purely presentational,
 * per-browser (localStorage), never sent to the server. */
export type Vibe = "Professional" | "Positivity boost" | "Space nerd";

export const VIBES: Vibe[] = ["Professional", "Positivity boost", "Space nerd"];

export interface VibeSettings {
  vibe: Vibe;
  hearts: number;
  pluses: number;
  rockets: number;
  ufos: number;
}

export const DEFAULT_VIBE: VibeSettings = {
  vibe: "Professional",
  hearts: 7,
  pluses: 7,
  rockets: 7,
  ufos: 1,
};

const KEY = "scilink.vibe";

export function loadVibe(): VibeSettings {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return DEFAULT_VIBE;
    const v = { ...DEFAULT_VIBE, ...JSON.parse(raw) } as VibeSettings;
    return VIBES.includes(v.vibe) ? v : DEFAULT_VIBE;
  } catch {
    return DEFAULT_VIBE;
  }
}

export function saveVibe(v: VibeSettings): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(v));
  } catch {
    /* private mode */
  }
}

/** The glyph the working-spinner pulses. */
export const SPINNER_ICON: Record<Vibe, string> = {
  Professional: "•",
  "Positivity boost": "🐾",
  "Space nerd": "📡",
};

export const VibeContext = createContext<Vibe>("Professional");
export const useVibe = () => useContext(VibeContext);
