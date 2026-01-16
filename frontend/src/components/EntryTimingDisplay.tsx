import {
    Clock,
    AlertTriangle,
    CheckCircle2,
    Timer,
    Target,
    TrendingUp,
    TrendingDown,
    Zap,
    Pause,
    Play,
    SkipForward,
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
    Tooltip,
    TooltipContent,
    TooltipTrigger,
} from "@/components/ui/tooltip";

export interface EntryTimingData {
    signal: "wait" | "prepare" | "ready" | "now";
    timing_description?: string;
    conditions?: string[];
    entry_price_zone?: string;
    stop_loss?: string;
    take_profit?: string;
    risk_reward?: string;
    timeframe?: string;
}

interface EntryTimingDisplayProps {
    timing: EntryTimingData;
    className?: string;
}

const signalConfig = {
    wait: {
        icon: Pause,
        label: "Wait",
        description: "Pattern forming, not ready yet",
        bgClass: "bg-muted",
        textClass: "text-muted-foreground",
        borderClass: "border-border",
        pulseClass: "",
    },
    prepare: {
        icon: Timer,
        label: "Prepare",
        description: "Setup developing, get ready",
        bgClass: "bg-warning/10",
        textClass: "text-warning",
        borderClass: "border-warning/30",
        pulseClass: "",
    },
    ready: {
        icon: Play,
        label: "Ready",
        description: "Wait for final confirmation",
        bgClass: "bg-primary/10",
        textClass: "text-primary",
        borderClass: "border-primary/30",
        pulseClass: "animate-pulse-slow",
    },
    now: {
        icon: Zap,
        label: "Enter Now",
        description: "Entry conditions met",
        bgClass: "bg-success/10",
        textClass: "text-success",
        borderClass: "border-success/30",
        pulseClass: "animate-pulse",
    },
};

export function EntryTimingDisplay({ timing, className }: EntryTimingDisplayProps) {
    const config = signalConfig[timing.signal];
    const Icon = config.icon;

    return (
        <div className={cn("space-y-4", className)}>
            {/* Header with Signal Badge */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-muted flex items-center justify-center">
                        <Clock className="h-5 w-5 text-primary" />
                    </div>
                    <div>
                        <h3 className="font-semibold text-lg">Entry Timing</h3>
                        <p className="text-xs text-muted-foreground">When to enter the market</p>
                    </div>
                </div>

                {/* Signal Badge */}
                <Tooltip>
                    <TooltipTrigger asChild>
                        <div
                            className={cn(
                                "flex items-center gap-2 px-4 py-2 rounded-xl border",
                                config.bgClass,
                                config.borderClass,
                                config.pulseClass
                            )}
                        >
                            <Icon className={cn("w-5 h-5", config.textClass)} />
                            <span className={cn("font-semibold", config.textClass)}>
                                {config.label}
                            </span>
                        </div>
                    </TooltipTrigger>
                    <TooltipContent side="left">
                        <p>{config.description}</p>
                    </TooltipContent>
                </Tooltip>
            </div>

            {/* Signal Progress Bar */}
            <div className="space-y-2">
                <div className="flex justify-between text-xs text-muted-foreground">
                    <span>Wait</span>
                    <span>Prepare</span>
                    <span>Ready</span>
                    <span>Enter</span>
                </div>
                <div className="h-2 bg-muted rounded-full overflow-hidden flex">
                    <div
                        className={cn(
                            "h-full transition-all duration-500",
                            timing.signal === "wait" && "w-1/4 bg-muted-foreground/50",
                            timing.signal === "prepare" && "w-2/4 bg-warning",
                            timing.signal === "ready" && "w-3/4 bg-primary",
                            timing.signal === "now" && "w-full bg-success"
                        )}
                    />
                </div>
            </div>

            {/* Timing Description */}
            {timing.timing_description && (
                <div className={cn(
                    "p-4 rounded-xl border",
                    config.bgClass,
                    config.borderClass.replace("/30", "/20")
                )}>
                    <p className="text-sm leading-relaxed">{timing.timing_description}</p>
                </div>
            )}

            {/* Conditions */}
            {timing.conditions && timing.conditions.length > 0 && (
                <div className="space-y-2">
                    <h4 className="text-sm font-medium flex items-center gap-2">
                        <CheckCircle2 className="h-4 w-4 text-primary" />
                        Conditions to Watch
                    </h4>
                    <ul className="space-y-1.5">
                        {timing.conditions.map((condition, idx) => (
                            <li
                                key={idx}
                                className="flex items-start gap-2 text-sm text-muted-foreground"
                            >
                                <span className="text-primary mt-1">•</span>
                                <span>{condition}</span>
                            </li>
                        ))}
                    </ul>
                </div>
            )}

            {/* Trade Levels Grid */}
            {(timing.entry_price_zone || timing.stop_loss || timing.take_profit) && (
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    {timing.entry_price_zone && (
                        <div className="p-3 rounded-lg bg-muted/50 border border-border/50">
                            <div className="flex items-center gap-2 mb-1">
                                <Target className="h-3.5 w-3.5 text-primary" />
                                <span className="text-xs text-muted-foreground">Entry Zone</span>
                            </div>
                            <p className="text-sm font-medium">{timing.entry_price_zone}</p>
                        </div>
                    )}
                    {timing.stop_loss && (
                        <div className="p-3 rounded-lg bg-destructive/5 border border-destructive/20">
                            <div className="flex items-center gap-2 mb-1">
                                <TrendingDown className="h-3.5 w-3.5 text-destructive" />
                                <span className="text-xs text-muted-foreground">Stop Loss</span>
                            </div>
                            <p className="text-sm font-medium text-destructive">{timing.stop_loss}</p>
                        </div>
                    )}
                    {timing.take_profit && (
                        <div className="p-3 rounded-lg bg-success/5 border border-success/20">
                            <div className="flex items-center gap-2 mb-1">
                                <TrendingUp className="h-3.5 w-3.5 text-success" />
                                <span className="text-xs text-muted-foreground">Take Profit</span>
                            </div>
                            <p className="text-sm font-medium text-success">{timing.take_profit}</p>
                        </div>
                    )}
                </div>
            )}

            {/* Risk/Reward & Timeframe */}
            {(timing.risk_reward || timing.timeframe) && (
                <div className="flex flex-wrap gap-3 pt-2">
                    {timing.risk_reward && (
                        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-muted/50 text-sm">
                            <span className="text-muted-foreground">Risk:Reward</span>
                            <span className="font-semibold text-primary">{timing.risk_reward}</span>
                        </div>
                    )}
                    {timing.timeframe && (
                        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-muted/50 text-sm">
                            <Clock className="h-3.5 w-3.5 text-muted-foreground" />
                            <span className="font-medium">{timing.timeframe}</span>
                        </div>
                    )}
                </div>
            )}

            {/* Warning */}
            <div className="flex items-start gap-2 px-3 py-2 rounded-lg bg-warning/5 border border-warning/20 text-xs text-muted-foreground">
                <AlertTriangle className="h-3.5 w-3.5 shrink-0 mt-0.5 text-warning" />
                <span>
                    Entry timing is a prediction based on pattern analysis. Always wait for confirmation and manage your risk accordingly.
                </span>
            </div>
        </div>
    );
}
