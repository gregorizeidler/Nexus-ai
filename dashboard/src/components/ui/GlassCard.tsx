import React from 'react';
import { motion } from 'framer-motion';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface GlassCardProps {
  children: React.ReactNode;
  className?: string;
  delay?: number;
  noHover?: boolean;
}

export const GlassCard = ({ children, className, delay = 0, noHover = false }: GlassCardProps) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay }}
      className={cn(
        "bg-[#111625]/60 backdrop-blur-md border border-white/5 rounded-2xl p-6",
        !noHover && "transition-all duration-300 hover:bg-[#161b2e]/80 hover:border-white/10 hover:shadow-lg hover:shadow-blue-900/5",
        className
      )}
    >
      {children}
    </motion.div>
  );
};

