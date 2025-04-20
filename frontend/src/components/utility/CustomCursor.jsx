import React from "react";
import useMousePosition from "../../hooks/useMousePosition";

function CustomCursor() {
  const { x, y } = useMousePosition();

  return (
    <div
      style={{
        position: "absolute",
        left: x,
        top: y,
        width: 24,
        height: 24,
        background: "#ffffff",
        borderRadius: "50%",
        pointerEvents: "none",
        transform: "translate(-50%, -50%)",
        zIndex: 9999,
      }}
    />
  );
}

export default CustomCursor;
