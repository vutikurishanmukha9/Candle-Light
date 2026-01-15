import { TrendingUp, TrendingDown, Minus, RotateCcw, Save, Calendar } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { PatternBadge } from "@/components/PatternBadge";
import { ConfidenceBar } from "@/components/ConfidenceBar";
import { DisclaimerBanner } from "@/components/DisclaimerBanner";

interface AnalysisResult {
  imageUrl: string;
  patterns: Array<{ name: string; bias: "bullish" | "bearish" | "neutral" }>;
  marketBias: "bullish" | "bearish" | "neutral";
  confidence: number;
  reasoning: string;
  timestamp: Date;
}

interface AnalysisCardProps {
  result: AnalysisResult;
  onNewUpload: () => void;
  onSave: () => void;
  className?: string;
}

export function AnalysisCard({
  result,
  onNewUpload,
  onSave,
  className,
}: AnalysisCardProps) {
  const BiasIcon = {
    bullish: TrendingUp,
    bearish: TrendingDown,
    neutral: Minus,
  }[result.marketBias];

  const biasColors = {
    bullish: "text-bullish",
    bearish: "text-bearish",
    neutral: "text-neutral",
  };

  const biasLabels = {
    bullish: "Bullish",
    bearish: "Bearish",
    neutral: "Neutral",
  };

  return (
    <div className={cn("glass-card overflow-hidden", className)}>
      {/* Image Preview */}
      <div className="relative aspect-video bg-secondary/50">
        <img
          src={result.imageUrl}
          alt="Analyzed chart"
          className="w-full h-full object-contain"
        />
        <div className="absolute top-3 right-3 flex items-center gap-2 px-3 py-1.5 bg-background/80 backdrop-blur-sm rounded-full">
          <Calendar className="w-3.5 h-3.5 text-muted-foreground" />
          <span className="text-xs text-muted-foreground">
            {result.timestamp.toLocaleDateString()}
          </span>
        </div>
      </div>

      <div className="p-6 space-y-6">
        {/* Market Bias */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div
              className={cn(
                "w-12 h-12 rounded-xl flex items-center justify-center",
                result.marketBias === "bullish" && "bg-bullish/20",
                result.marketBias === "bearish" && "bg-bearish/20",
                result.marketBias === "neutral" && "bg-neutral/20"
              )}
            >
              <BiasIcon className={cn("w-6 h-6", biasColors[result.marketBias])} />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Market Bias</p>
              <p className={cn("text-xl font-semibold", biasColors[result.marketBias])}>
                {biasLabels[result.marketBias]}
              </p>
            </div>
          </div>
        </div>

        {/* Confidence */}
        <ConfidenceBar value={result.confidence} label="Confidence Score" />

        {/* Detected Patterns */}
        <div className="space-y-3">
          <p className="text-sm text-muted-foreground">Detected Patterns</p>
          <div className="flex flex-wrap gap-2">
            {result.patterns.map((pattern, idx) => (
              <PatternBadge
                key={idx}
                pattern={pattern.name}
                bias={pattern.bias}
              />
            ))}
          </div>
        </div>

        {/* AI Reasoning */}
        <div className="space-y-3">
          <p className="text-sm text-muted-foreground">AI Analysis</p>
          <div className="p-4 bg-secondary/50 rounded-lg">
            <p className="text-sm leading-relaxed text-foreground/90">
              {result.reasoning}
            </p>
          </div>
        </div>

        {/* Disclaimer */}
        <DisclaimerBanner />

        {/* Actions */}
        <div className="flex gap-3">
          <Button variant="outline" onClick={onNewUpload} className="flex-1">
            <RotateCcw className="w-4 h-4" />
            New Analysis
          </Button>
          <Button variant="default" onClick={onSave} className="flex-1">
            <Save className="w-4 h-4" />
            Save Result
          </Button>
        </div>
      </div>
    </div>
  );
}
