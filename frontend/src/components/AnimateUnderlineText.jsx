import { motion } from "framer-motion";

function AnimatedUnderlineText({ children }) {
  return (
    <span style={{ position: "relative", display: "inline-block" }}>
      {children}
      <motion.span
        style={{
          position: "absolute z-60",
          left: 0,
          bottom: 0,
          height: 2,
          width: "100%",
          background: "#ffffff",
          transformOrigin: "bottom left",
        }}
        initial={{ scaleX: 0 }}
        whileHover={{ scaleX: 1 }}
        transition={{ duration: 0.25, ease: "easeOut" }}
      />
    </span>
  );
}

export default AnimatedUnderlineText;
