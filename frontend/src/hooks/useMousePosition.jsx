import { useState, useEffect } from "react";

function useMousePosition() {
  const [mousePosition, setMousePosition] = useState({ x: 600, y: 300 });

  useEffect(() => {
    const updateMousePosition = (event) => {
      setMousePosition({ x: event.clientX, y: event.clientY });
    };
    window.addEventListener("mousemove", updateMousePosition);
    return () => window.removeEventListener("mousemove", updateMousePosition);
  }, []);

  return mousePosition;
}

export default useMousePosition;
