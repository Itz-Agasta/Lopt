"use client";
import React, { useEffect, useState } from "react";
import { motion, useInView } from "motion/react";
import { cn } from "@/lib/utils";

export function AnimatedCounter({
  from = 0,
  to,
  duration = 2,
  className,
  suffix = "",
  prefix = "",
  separator = ",",
  ...props
}) {
  const [count, setCount] = useState(from);
  const ref = React.useRef(null);
  const isInView = useInView(ref, { once: true, threshold: 0.3 });

  useEffect(() => {
    if (!isInView) return;

    const startTime = Date.now();
    const endTime = startTime + duration * 1000;

    const timer = setInterval(() => {
      const now = Date.now();
      const progress = Math.min((now - startTime) / (endTime - startTime), 1);
      
      // Easing function for smooth animation
      const easeOutCubic = 1 - Math.pow(1 - progress, 3);
      const currentCount = Math.floor(from + (to - from) * easeOutCubic);
      
      setCount(currentCount);

      if (progress >= 1) {
        clearInterval(timer);
        setCount(to);
      }
    }, 16); // ~60fps

    return () => clearInterval(timer);
  }, [isInView, from, to, duration]);

  const formatNumber = (num) => {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, separator);
  };

  return (
    <motion.span
      ref={ref}
      className={cn("font-bold tabular-nums", className)}
      initial={{ opacity: 0, scale: 0.5 }}
      animate={isInView ? { opacity: 1, scale: 1 } : {}}
      transition={{ duration: 0.5, delay: 0.2 }}
      {...props}
    >
      {prefix}{formatNumber(count)}{suffix}
    </motion.span>
  );
}