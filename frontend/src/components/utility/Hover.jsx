import { motion, useAnimation } from "framer-motion";
import { useState } from "react";

const Hover = ({ children, width = "fit-content" }) => {
  const controls = useAnimation();
  const [hovered, setHovered] = useState(false);

  const handleMouseEnter = () => {
    if (!hovered) {
      setHovered(true);
      controls.start({
        y: [0, 75],
        transition: {
          duration: 0.25,
          times: [0, 1],
          repeat: 1,
          repeatType: "mirror",
        },
      });
    }
  };

  const handleMouseLeave = () => {
    setHovered(false);
    controls.set({ y: 0 });
  };

  return (
    <div
      style={{ position: "relative", width, overflow: "hidden" }}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <motion.div animate={controls} style={{ display: "inline-block" }}>
        {children}
      </motion.div>
    </div>
  );
};

export default Hover;
