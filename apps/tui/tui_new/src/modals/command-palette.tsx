// Modal: Command palette (Ctrl+P) — quick actions

import { useState, useMemo } from "react";
import { useKeyboard, useTerminalDimensions } from "@opentui/react";
import { TextAttributes } from "@opentui/core";
import { useDispatch } from "../state/app-context";
import { TAB_LIST, type TabId } from "../state/types";
import * as colors from "../theme/colors";

interface Command {
  id: string;
  label: string;
  shortcut?: string;
  action: () => void;
}

export function CommandPalette({ onClose }: { onClose: () => void }) {
  const [query, setQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const dispatch = useDispatch();
  const { width, height } = useTerminalDimensions();

  const commands: Command[] = useMemo(() => {
    const cmds: Command[] = [];

    // Tab navigation commands
    for (const tab of TAB_LIST) {
      cmds.push({
        id: `tab:${tab.id}`,
        label: `Go to ${tab.label}`,
        shortcut: tab.key,
        action: () => {
          dispatch({ type: "SET_TAB", tab: tab.id });
          onClose();
        },
      });
    }

    // Trade commands
    cmds.push({
      id: "buy",
      label: "Buy — Open buy order dialog",
      shortcut: "B",
      action: () => {
        dispatch({ type: "OPEN_MODAL", modal: "trade-buy" });
        onClose();
      },
    });
    cmds.push({
      id: "sell",
      label: "Sell — Open sell order dialog",
      shortcut: "S",
      action: () => {
        dispatch({ type: "OPEN_MODAL", modal: "trade-sell" });
        onClose();
      },
    });

    // Utility commands
    cmds.push({
      id: "lookup",
      label: "Lookup — Quick symbol search",
      shortcut: "/",
      action: () => {
        dispatch({ type: "OPEN_MODAL", modal: "lookup" });
        onClose();
      },
    });
    cmds.push({
      id: "refresh",
      label: "Refresh — Reload current data",
      shortcut: "R",
      action: () => {
        dispatch({ type: "SET_LAST_REFRESH", time: new Date().toISOString() });
        onClose();
      },
    });
    cmds.push({
      id: "mode",
      label: "Mode — Paper trading",
      shortcut: "M",
      action: () => {
        dispatch({ type: "OPEN_MODAL", modal: "mode_select" });
        onClose();
      },
    });
    cmds.push({
      id: "quit",
      label: "Quit — Exit application",
      shortcut: "Q",
      action: () => {
        process.exit(0);
      },
    });

    return cmds;
  }, [dispatch, onClose]);

  const filtered = useMemo(() => {
    if (!query.trim()) return commands;
    const q = query.toLowerCase();
    return commands.filter(
      (cmd) =>
        cmd.label.toLowerCase().includes(q) ||
        cmd.id.toLowerCase().includes(q)
    );
  }, [query, commands]);

  // Clamp selected index
  const clampedIndex = Math.min(selectedIndex, Math.max(0, filtered.length - 1));

  useKeyboard(
    (key) => {
      if (key.name === "Escape") {
        onClose();
        return;
      }
      if (key.name === "ArrowUp" || (key.ctrl && key.name === "p")) {
        setSelectedIndex(Math.max(0, clampedIndex - 1));
        return;
      }
      if (key.name === "ArrowDown" || (key.ctrl && key.name === "n")) {
        setSelectedIndex(Math.min(filtered.length - 1, clampedIndex + 1));
        return;
      }
      if (key.name === "Return") {
        const cmd = filtered[clampedIndex];
        if (cmd) cmd.action();
        return;
      }
    },
    { release: false }
  );

  const boxWidth = 50;
  const maxVisible = 12;
  const boxHeight = Math.min(filtered.length, maxVisible) + 4; // title + input + padding
  const left = Math.floor((width - boxWidth) / 2);
  const top = Math.max(2, Math.floor((height - boxHeight) / 3));

  const visibleCommands = filtered.slice(0, maxVisible);

  return (
    <box
      width={width}
      height={height}
      backgroundColor="#00000088"
      position="absolute"
      top={0}
      left={0}
    >
      <box
        position="absolute"
        left={left}
        top={top}
        width={boxWidth}
        height={boxHeight}
        backgroundColor={colors.BG_PANEL}
        borderStyle="single"
        borderColor={colors.FG_AMBER}
        flexDirection="column"
      >
        {/* Search input */}
        <box
          backgroundColor={colors.BG_SURFACE}
          height={1}
          paddingLeft={1}
          flexDirection="row"
          gap={1}
        >
          <text fg={colors.FG_AMBER} attributes={TextAttributes.BOLD}>
            {">"}
          </text>
          <input
            placeholder="Type a command..."
            onInput={(v: string) => {
              setQuery(v);
              setSelectedIndex(0);
            }}
            focused={true}
          />
        </box>

        {/* Command list */}
        {visibleCommands.length === 0 ? (
          <box height={1} paddingLeft={2}>
            <text fg={colors.FG_DIM}>No matching commands</text>
          </box>
        ) : (
          visibleCommands.map((cmd, i) => {
            const isSelected = i === clampedIndex;
            return (
              <box
                key={cmd.id}
                height={1}
                paddingLeft={1}
                paddingRight={1}
                backgroundColor={isSelected ? colors.BG_HOVER : colors.BG_PANEL}
                flexDirection="row"
                justifyContent="space-between"
              >
                <text
                  fg={isSelected ? colors.FG_BRIGHT : colors.FG_PRIMARY}
                  attributes={isSelected ? TextAttributes.BOLD : 0}
                >
                  {isSelected ? "> " : "  "}{cmd.label}
                </text>
                {cmd.shortcut && (
                  <text fg={colors.FG_DIM}>[{cmd.shortcut}]</text>
                )}
              </box>
            );
          })
        )}

        {/* Footer */}
        <box height={1} paddingLeft={1}>
          <text fg={colors.FG_DIM}>
            Up/Down: Navigate  Enter: Execute  Esc: Close
          </text>
        </box>
      </box>
    </box>
  );
}
