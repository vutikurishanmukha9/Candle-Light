import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import { cn } from "@/lib/utils";

interface PatternBadgeProps {
  pattern: string;
  bias?: "bullish" | "bearish" | "neutral";
  className?: string;
}

export function PatternBadge({ pattern, bias = "neutral", className }: PatternBadgeProps) {
  const biasStyles = {
    bullish: "bg-bullish/20 text-bullish border-bullish/30",
    bearish: "bg-bearish/20 text-bearish border-bearish/30",
    neutral: "bg-neutral/20 text-neutral border-neutral/30",
  };

  const BiasIcon = {
    bullish: TrendingUp,
    bearish: TrendingDown,
    neutral: Minus,
  }[bias];

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium border",
        biasStyles[bias],
        className
      )}
    >
      <BiasIcon className="w-3 h-3" />
      {pattern}
    </span>
  );
}
