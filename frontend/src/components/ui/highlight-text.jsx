"use client";
import React from "react";
import { motion } from "motion/react";
import { cn } from "@/lib/utils";

export function HighlightText({
  children,
  className,
  highlightClassName,
  animationDelay = 0,
  ...props
}) {
  return (
    <span className={cn("relative inline-block", className)} {...props}>
      <motion.span
        className={cn(
          "relative z-10 text-transparent bg-clip-text bg-gradient-to-r from-red-400 via-red-500 to-orange-500",
          "drop-shadow-[0_0_10px_rgba(239,68,68,0.5)]",
          highlightClassName
        )}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ 
          duration: 0.6, 
          delay: animationDelay,
          ease: "easeOut" 
        }}
      >
        {children}
      </motion.span>
      
      {/* Animated background highlight */}
      <motion.span
        className="absolute inset-0 bg-gradient-to-r from-red-500/20 via-red-400/30 to-orange-500/20 rounded-lg blur-lg"
        initial={{ scale: 0, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ 
          duration: 0.8, 
          delay: animationDelay + 0.2,
          ease: "easeOut" 
        }}
      />
      
      {/* Pulsing glow effect */}
      <motion.span
        className="absolute inset-0 bg-gradient-to-r from-red-500/10 to-orange-500/10 rounded-lg"
        animate={{ 
          scale: [1, 1.05, 1],
          opacity: [0.5, 0.8, 0.5] 
        }}
        transition={{ 
          duration: 2, 
          repeat: Infinity,
          ease: "easeInOut",
          delay: animationDelay + 0.5
        }}
      />
    </span>
  );
}