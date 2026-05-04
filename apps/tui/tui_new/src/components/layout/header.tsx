// Header bar — Gloomberb-style: app name, mode, NEPSE index, timestamp
// Deep blue background, white text, 1 row

import { TextAttributes } from "@opentui/core";
import { useMarketIndices } from "../../data/hooks";
import * as colors from "../../theme/colors";
import { fmtPrice, fmtChg, fmtVol, fmtTime, chgColor } from "../ui/helpers";

export function Header() {
  const { data: indices } = useMarketIndices();

  const modeLabel = " PAPER ";
  const modeFg = colors.positive;
  const now = fmtTime();

  return (
    <box
      backgroundColor={colors.header}
      height={1}
      flexDirection="row"
      paddingLeft={1}
      paddingRight={1}
      justifyContent="space-between"
    >
      {/* Left: App name + mode */}
      <box flexDirection="row" gap={1}>
        <text fg={colors.headerText} attributes={TextAttributes.BOLD}>
          NEPSE TUI
        </text>
        <text fg={modeFg} attributes={TextAttributes.BOLD}>
          {modeLabel}
        </text>
      </box>

      {/* Center: NEPSE index */}
      <box flexDirection="row" gap={1}>
        {indices ? (
          <>
            <text fg={colors.headerText} attributes={TextAttributes.BOLD}>
              NEPSE
            </text>
            <text fg={colors.headerText}>
              {fmtPrice(indices.nepse_index)}
            </text>
            <text fg={chgColor(indices.nepse_change_pct)} attributes={TextAttributes.BOLD}>
              {fmtChg(indices.nepse_change_pct)}
            </text>
            <text fg={colors.headerText}>
              Vol {fmtVol(indices.total_volume)}
            </text>
            <text fg={colors.positive}>{indices.advances}↑</text>
            <text fg={colors.negative}>{indices.declines}↓</text>
            <text fg={colors.neutral}>{indices.unchanged}→</text>
          </>
        ) : (
          <text fg={colors.textMuted}>Connecting...</text>
        )}
      </box>

      {/* Right: timestamp */}
      <text fg={colors.headerText}>{now}</text>
    </box>
  );
}
