
import React from "react";
import { cn } from "@/lib/utils";

interface GlassmorphicButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  icon?: React.ReactNode;
  children?: React.ReactNode;
  variant?: "default" | "primary" | "danger";
}

const GlassmorphicButton = ({
  icon,
  children,
  className,
  variant = "default",
  ...props
}: GlassmorphicButtonProps) => {
  return (
    <button
      className={cn(
        "flex items-center justify-center px-4 py-2 rounded-lg backdrop-blur-md border transition-all",
        "hover:shadow-md active:scale-95",
        {
          "bg-white/10 border-white/20 text-foreground": variant === "default",
          "bg-primary/10 border-primary/20 text-primary": variant === "primary",
          "bg-destructive/10 border-destructive/20 text-destructive": variant === "danger",
        },
        className
      )}
      {...props}
    >
      {icon && <span className="mr-2">{icon}</span>}
      {children}
    </button>
  );
};

export default GlassmorphicButton;
