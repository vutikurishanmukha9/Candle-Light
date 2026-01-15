import { AlertTriangle, Info, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { useState } from "react";

interface ErrorAlertProps {
  title?: string;
  message: string;
  variant?: "error" | "warning" | "info";
  dismissible?: boolean;
  className?: string;
}

export function ErrorAlert({
  title,
  message,
  variant = "error",
  dismissible = false,
  className,
}: ErrorAlertProps) {
  const [dismissed, setDismissed] = useState(false);

  if (dismissed) return null;

  const styles = {
    error: "bg-bearish/10 border-bearish/30 text-bearish",
    warning: "bg-warning/10 border-warning/30 text-warning",
    info: "bg-primary/10 border-primary/30 text-primary",
  };

  const Icon = variant === "info" ? Info : AlertTriangle;

  return (
    <div
      className={cn(
        "flex items-start gap-3 p-4 rounded-lg border",
        styles[variant],
        className
      )}
    >
      <Icon className="w-5 h-5 mt-0.5 shrink-0" />
      <div className="flex-1 min-w-0">
        {title && <p className="font-medium mb-1">{title}</p>}
        <p className="text-sm opacity-90">{message}</p>
      </div>
      {dismissible && (
        <button
          onClick={() => setDismissed(true)}
          className="p-1 hover:bg-white/10 rounded transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      )}
    </div>
  );
}
