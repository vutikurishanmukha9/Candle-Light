import { cn } from "@/lib/utils";

interface LoadingSkeletonProps {
  className?: string;
  lines?: number;
}

export function LoadingSkeleton({ className, lines = 3 }: LoadingSkeletonProps) {
  return (
    <div className={cn("space-y-3 animate-pulse", className)}>
      {Array.from({ length: lines }).map((_, i) => (
        <div
          key={i}
          className="h-4 bg-secondary rounded"
          style={{ width: `${Math.random() * 40 + 60}%` }}
        />
      ))}
    </div>
  );
}

export function AnalysisLoadingSkeleton() {
  return (
    <div className="glass-card p-6 space-y-6 animate-pulse">
      <div className="flex items-center gap-4">
        <div className="w-16 h-16 bg-secondary rounded-lg" />
        <div className="space-y-2 flex-1">
          <div className="h-5 bg-secondary rounded w-1/3" />
          <div className="h-4 bg-secondary rounded w-1/2" />
        </div>
      </div>
      <div className="space-y-3">
        <div className="h-4 bg-secondary rounded w-full" />
        <div className="h-4 bg-secondary rounded w-4/5" />
        <div className="h-4 bg-secondary rounded w-3/5" />
      </div>
      <div className="flex gap-2">
        {[1, 2, 3].map((i) => (
          <div key={i} className="h-8 w-24 bg-secondary rounded-full" />
        ))}
      </div>
    </div>
  );
}
