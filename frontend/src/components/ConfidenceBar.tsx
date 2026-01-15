import { cn } from "@/lib/utils";

interface ConfidenceBarProps {
  value: number;
  label?: string;
  className?: string;
}

export function ConfidenceBar({ value, label, className }: ConfidenceBarProps) {
  const getColor = (val: number) => {
    if (val >= 70) return "bg-bullish";
    if (val >= 40) return "bg-warning";
    return "bg-bearish";
  };

  return (
    <div className={cn("space-y-2", className)}>
      {label && (
        <div className="flex justify-between items-center text-sm">
          <span className="text-muted-foreground">{label}</span>
          <span className="font-mono font-medium text-foreground">{value}%</span>
        </div>
      )}
      <div className="h-2 bg-secondary rounded-full overflow-hidden">
        <div
          className={cn("h-full rounded-full transition-all duration-500", getColor(value))}
          style={{ width: `${value}%` }}
        />
      </div>
    </div>
  );
}
