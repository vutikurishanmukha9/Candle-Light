import { TrendingUp, TrendingDown, Minus, Calendar, Sparkles, Target } from "lucide-react";
import { cn } from "@/lib/utils";
import { PatternBadge } from "@/components/PatternBadge";
import { ConfidenceBar } from "@/components/ConfidenceBar";
import { DisclaimerBanner } from "@/components/DisclaimerBanner";
import { AIAnalysisDisplay } from "@/components/AIAnalysisDisplay";
import { EntryTimingDisplay, EntryTimingData } from "@/components/EntryTimingDisplay";

interface AnalysisResult {
  imageUrl: string;
  patterns: Array<{ name: string; bias: "bullish" | "bearish" | "neutral" }>;
  marketBias: "bullish" | "bearish" | "neutral";
  confidence: number;
  reasoning: string;
  timestamp: Date;
  entryTiming?: EntryTimingData;
}

interface AnalysisCardProps {
  analysis: AnalysisResult;
  className?: string;
}

export function AnalysisCard({ analysis, className }: AnalysisCardProps) {
  const BiasIcon = {
    bullish: TrendingUp,
    bearish: TrendingDown,
    neutral: Minus,
  }[analysis.marketBias];

  const biasStyles = {
    bullish: {
      bg: "bg-success/10",
      text: "text-success",
      border: "border-success/20",
      glow: "shadow-success/20",
    },
    bearish: {
      bg: "bg-destructive/10",
      text: "text-destructive",
      border: "border-destructive/20",
      glow: "shadow-destructive/20",
    },
    neutral: {
      bg: "bg-muted",
      text: "text-muted-foreground",
      border: "border-border",
      glow: "",
    },
  };

  const styles = biasStyles[analysis.marketBias];

  const biasLabels = {
    bullish: "Bullish",
    bearish: "Bearish",
    neutral: "Neutral",
  };

  return (
    <div className={cn("glass-card overflow-hidden", className)}>
      {/* Image Preview */}
      <div className="relative aspect-video bg-muted/30 overflow-hidden">
        <img
          src={analysis.imageUrl}
          alt="Analyzed chart"
          className="w-full h-full object-contain"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-card via-transparent to-transparent opacity-60" />

        {/* Timestamp Badge */}
        <div className="absolute top-3 right-3 flex items-center gap-1.5 px-3 py-1.5 bg-card/90 backdrop-blur-sm rounded-full border border-border/50">
          <Calendar className="w-3.5 h-3.5 text-muted-foreground" />
          <span className="text-xs font-medium text-muted-foreground">
            {analysis.timestamp.toLocaleDateString()}
          </span>
        </div>

        {/* Market Bias Badge - Floating */}
        <div className="absolute bottom-4 left-4">
          <div
            className={cn(
              "flex items-center gap-2 px-4 py-2 rounded-xl border backdrop-blur-sm",
              "shadow-lg",
              styles.bg,
              styles.border
            )}
          >
            <BiasIcon className={cn("w-5 h-5", styles.text)} />
            <span className={cn("font-semibold", styles.text)}>
              {biasLabels[analysis.marketBias]}
            </span>
          </div>
        </div>
      </div>

      <div className="p-6 space-y-6">
        {/* Confidence Section */}
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Target className="h-4 w-4 text-primary" />
            <span className="text-sm font-medium">Confidence Score</span>
          </div>
          <ConfidenceBar value={analysis.confidence} />
        </div>

        {/* Detected Patterns */}
        {analysis.patterns.length > 0 && (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-primary" />
              <span className="text-sm font-medium">Detected Patterns</span>
              <span className="text-xs text-muted-foreground ml-auto">
                {analysis.patterns.length} found
              </span>
            </div>
            <div className="flex flex-wrap gap-2">
              {analysis.patterns.map((pattern, idx) => (
                <PatternBadge
                  key={idx}
                  pattern={pattern.name}
                  bias={pattern.bias}
                />
              ))}
            </div>
          </div>
        )}

        {/* Entry Timing - When to Enter */}
        {analysis.entryTiming && (
          <EntryTimingDisplay timing={analysis.entryTiming} />
        )}

        {/* AI Reasoning - Premium Display */}
        {analysis.reasoning && (
          <AIAnalysisDisplay reasoning={analysis.reasoning} />
        )}

        {/* Disclaimer */}
        <DisclaimerBanner />
      </div>
    </div>
  );
}
