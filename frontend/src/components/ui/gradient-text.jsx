'use client';
 
import * as React from 'react';
import { motion } from 'motion/react';
 
import { cn } from '@/lib/utils';
 
function GradientText({
  text,
  className,
  gradient = 'linear-gradient(90deg, #ef4444 0%, #f97316 20%, #eab308 50%, #f97316 80%, #ef4444 100%)',
  neon = false,
  transition = { duration: 3, repeat: Infinity, ease: 'linear' },
  ...props
}) {
  const baseStyle = {
    backgroundImage: gradient,
  };
 
  return (
    <span
      data-slot="gradient-text"
      className={cn('relative inline-block', className)}
      {...props}
    >
      <motion.span
        className="m-0 text-transparent bg-clip-text bg-[length:200%_100%]"
        style={baseStyle}
        animate={{ backgroundPositionX: ['0%', '200%'] }}
        transition={transition}
      >
        {text}
      </motion.span>
 
      {neon && (
        <motion.span
          className="m-0 absolute top-0 left-0 text-transparent bg-clip-text blur-[8px] mix-blend-plus-lighter bg-[length:200%_100%]"
          style={baseStyle}
          animate={{ backgroundPositionX: ['0%', '200%'] }}
          transition={transition}
        >
          {text}
        </motion.span>
      )}
    </span>
  );
}
 
export { GradientText };