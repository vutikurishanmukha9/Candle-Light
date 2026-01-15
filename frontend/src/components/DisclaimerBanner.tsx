import { AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";

interface DisclaimerBannerProps {
  className?: string;
  variant?: "inline" | "fixed";
}

export function DisclaimerBanner({ className, variant = "inline" }: DisclaimerBannerProps) {
  return (
    <div
      className={cn(
        "flex items-center gap-3 px-4 py-3 bg-warning/10 border border-warning/20 rounded-lg",
        variant === "fixed" && "fixed bottom-4 left-4 right-4 max-w-2xl mx-auto z-50",
        className
      )}
    >
      <AlertTriangle className="w-5 h-5 text-warning shrink-0" />
      <p className="text-sm text-warning/90">
        <span className="font-medium">Not Financial Advice:</span> AI analysis is for informational purposes only. Always consult a qualified financial advisor before making investment decisions.
      </p>
    </div>
  );
}
