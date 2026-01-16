import { cn } from "@/lib/utils";

interface LoadingSkeletonProps {
  className?: string;
  lines?: number;
}

export function LoadingSkeleton({ className, lines = 3 }: LoadingSkeletonProps) {
  return (
    <div className={cn("space-y-3", className)}>
      {Array.from({ length: lines }).map((_, i) => (
        <div
          key={i}
          className="h-4 rounded-md animate-shimmer"
          style={{ width: `${85 - i * 15}%` }}
        />
      ))}
    </div>
  );
}

export function AnalysisLoadingSkeleton() {
  return (
    <div className="glass-card p-6 space-y-6">
      {/* Header skeleton */}
      <div className="flex items-center gap-4">
        <div className="w-14 h-14 rounded-xl animate-shimmer" />
        <div className="space-y-2 flex-1">
          <div className="h-5 rounded-md w-1/3 animate-shimmer" />
          <div className="h-4 rounded-md w-1/2 animate-shimmer animation-delay-100" />
        </div>
      </div>

      {/* Chart area skeleton */}
      <div className="aspect-video rounded-xl animate-shimmer" />

      {/* Content skeleton */}
      <div className="space-y-4">
        <div className="space-y-2">
          <div className="h-4 rounded-md w-full animate-shimmer animation-delay-200" />
          <div className="h-4 rounded-md w-4/5 animate-shimmer animation-delay-300" />
          <div className="h-4 rounded-md w-3/5 animate-shimmer animation-delay-400" />
        </div>

        {/* Pattern badges skeleton */}
        <div className="flex gap-2 pt-2">
          {[1, 2, 3].map((i) => (
            <div
              key={i}
              className="h-8 w-24 rounded-full animate-shimmer"
              style={{ animationDelay: `${i * 100}ms` }}
            />
          ))}
        </div>
      </div>

      {/* Footer skeleton */}
      <div className="flex items-center justify-between pt-4 border-t border-border/50">
        <div className="h-4 w-24 rounded-md animate-shimmer animation-delay-500" />
        <div className="h-4 w-32 rounded-md animate-shimmer animation-delay-500" />
      </div>
    </div>
  );
}

export function CardSkeleton() {
  return (
    <div className="glass-card p-4 space-y-4">
      <div className="aspect-video rounded-lg animate-shimmer" />
      <div className="space-y-2">
        <div className="h-4 rounded-md w-3/4 animate-shimmer" />
        <div className="h-3 rounded-md w-1/2 animate-shimmer animation-delay-100" />
      </div>
    </div>
  );
}
