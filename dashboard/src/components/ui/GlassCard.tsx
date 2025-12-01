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
      transition={{ duration: 0.4, delay }}
      className={cn(
        "bg-white border border-slate-200 rounded-xl p-6 shadow-sm",
        !noHover && "transition-all duration-300 hover:shadow-md hover:border-slate-300",
        className
      )}
    >
      {children}
    </motion.div>
  );
};

