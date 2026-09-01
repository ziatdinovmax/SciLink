import { createContext, useContext } from "react";

/** Cross-panel actions: any component (artifact cards in chat, rows in the
 * file explorer) can jump to a file in the Files tab or drop a file
 * reference into the chat draft. */
export interface UIActions {
  openInFiles: (path: string) => void;
  attachToChat: (path: string) => void;
}

export const UIContext = createContext<UIActions>({
  openInFiles: () => {},
  attachToChat: () => {},
});

export const useUIActions = () => useContext(UIContext);
