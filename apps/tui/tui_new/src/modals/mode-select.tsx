// Modal: Paper trading mode selection

import { useKeyboard, useTerminalDimensions } from "@opentui/react";
import { TextAttributes } from "@opentui/core";
import * as colors from "../theme/colors";

export function ModeSelectModal({
  onSelect,
}: {
  onSelect: (mode: "paper") => void;
}) {
  const { width, height } = useTerminalDimensions();

  useKeyboard(
    (key) => {
      if (key.name === "Return") {
        onSelect("paper");
      } else if (key.name === "Escape") {
        // Close without changing — caller can handle
      }
    },
    { release: false }
  );

  const boxWidth = 36;
  const boxHeight = 10;
  const left = Math.floor((width - boxWidth) / 2);
  const top = Math.floor((height - boxHeight) / 2);

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
        borderColor={colors.BORDER_FOCUS}
        flexDirection="column"
      >
        {/* Title */}
        <box
          backgroundColor={colors.BG_HEADER}
          height={1}
          paddingLeft={1}
          paddingRight={1}
        >
          <text fg={colors.FG_AMBER} attributes={TextAttributes.BOLD}>
            :: PAPER TRADING MODE
          </text>
        </box>

        {/* Spacer */}
        <box height={1} />

        {/* Paper option */}
        <box
          height={2}
          paddingLeft={2}
          backgroundColor={colors.BG_FOCUS}
          flexDirection="column"
        >
          <text
            fg={colors.GAIN_HI}
            attributes={TextAttributes.BOLD}
          >
            {"> "}PAPER MODE
          </text>
          <text fg={colors.FG_DIM}>
            {"    Simulated trading with virtual capital"}
          </text>
        </box>

        {/* Footer */}
        <box height={1} paddingLeft={1}>
          <text fg={colors.FG_DIM}>Enter to continue</text>
        </box>
      </box>
    </box>
  );
}
