import { useMemo } from "react";
import {
    AlertTriangle,
    CheckCircle2,
    TrendingUp,
    TrendingDown,
    Target,
    Lightbulb,
    BarChart3,
    Shield,
    HelpCircle,
    Info
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
    Tooltip,
    TooltipContent,
    TooltipTrigger,
} from "@/components/ui/tooltip";

interface AIAnalysisDisplayProps {
    reasoning: string;
    className?: string;
}

interface ParsedSection {
    type: "summary" | "pattern" | "signals" | "caution" | "levels" | "recommendation" | "text";
    title?: string;
    content: string;
    plainEnglish?: string;
}

// Technical terms dictionary with beginner-friendly explanations
const technicalTerms: Record<string, string> = {
    "double bottom": "A pattern where price drops, rises, drops to similar level, then rises again - like a 'W' shape. Often signals price may go up.",
    "hammer": "A candlestick with small body and long lower shadow. Shows buyers pushed price back up after sellers tried to push it down.",
    "volume": "The number of shares/contracts traded. Higher volume = more traders are participating, making the signal stronger.",
    "RSI": "Relative Strength Index - measures if a stock is 'overbought' (too expensive) or 'oversold' (potentially cheap).",
    "divergence": "When price moves one direction but an indicator moves the opposite way. Can signal a trend change is coming.",
    "bullish": "Expecting the price to GO UP. Like a bull charging upward with its horns.",
    "bearish": "Expecting the price to GO DOWN. Like a bear swiping downward with its paws.",
    "support": "A price level where buyers tend to step in and prevent further drops - like a floor.",
    "resistance": "A price level where sellers tend to step in and prevent further rises - like a ceiling.",
    "breakout": "When price moves beyond a support or resistance level with conviction.",
    "wedge": "A pattern where price is squeezed between two converging trendlines. Can signal a big move coming.",
    "rising wedge": "Price making higher highs and higher lows, but the range is narrowing. Often signals a potential drop.",
    "confirmation": "Additional evidence that supports a trading signal. Waiting for confirmation reduces false signals.",
    "reversal": "When price changes direction - from going up to going down, or vice versa.",
    "trend": "The general direction price is moving over time - up (bullish), down (bearish), or sideways.",
    "candlestick": "A way to display price movement showing open, high, low, and close prices.",
};

// Parse the AI reasoning into structured sections based on markdown headers
function parseAnalysisText(text: string): ParsedSection[] {
    const sections: ParsedSection[] = [];

    // Split by markdown headers (## or ###)
    const lines = text.split('\n');
    let currentSection: ParsedSection | null = null;
    let currentContent: string[] = [];

    const getSectionType = (header: string): ParsedSection["type"] => {
        const h = header.toLowerCase();
        if (h.includes("summary") || h.includes("executive") || h.includes("overview")) return "summary";
        if (h.includes("pattern") && !h.includes("breakdown")) return "pattern";
        if (h.includes("breakdown") || h.includes("analysis")) return "pattern";
        if (h.includes("entry") || h.includes("strategy")) return "signals";
        if (h.includes("risk") || h.includes("caution") || h.includes("warning")) return "caution";
        if (h.includes("level") || h.includes("support") || h.includes("resistance") || h.includes("target")) return "levels";
        if (h.includes("recommendation") || h.includes("conclusion") || h.includes("verdict") || h.includes("action")) return "recommendation";
        return "text";
    };

    const saveCurrentSection = () => {
        if (currentSection && currentContent.length > 0) {
            currentSection.content = currentContent.join('\n').trim();
            if (currentSection.content.length > 10) {
                sections.push(currentSection);
            }
        }
        currentContent = [];
    };

    for (const line of lines) {
        // Check for markdown headers (## or ###)
        const headerMatch = line.match(/^#{2,3}\s+(.+)$/);

        if (headerMatch) {
            // Save previous section
            saveCurrentSection();

            // Start new section
            const headerText = headerMatch[1].trim();
            currentSection = {
                type: getSectionType(headerText),
                title: headerText,
                content: ""
            };
        } else if (line.trim() === "---") {
            // Skip horizontal rules
            continue;
        } else if (currentSection) {
            // Add to current section
            currentContent.push(line);
        } else if (line.trim()) {
            // No section yet, create a summary section
            currentSection = { type: "summary", content: "" };
            currentContent.push(line);
        }
    }

    // Don't forget the last section
    saveCurrentSection();

    // If no sections found, return the whole text as summary
    if (sections.length === 0) {
        return [{ type: "summary", content: text }];
    }

    return sections;
}

// Clean markdown artifacts from text for app-native display
function cleanMarkdown(text: string): string {
    let cleaned = text
        // Remove bold+italic ***text***
        .replace(/\*\*\*([^*]+)\*\*\*/g, '$1')
        // Remove bold **text**
        .replace(/\*\*([^*]+)\*\*/g, '$1')
        // Remove remaining ** (unclosed bold)
        .replace(/\*\*/g, '')
        // Remove italic *text* (with content)
        .replace(/\*([^*\s][^*]*)\*/g, '$1')
        // Remove code `text`
        .replace(/`([^`]+)`/g, '$1')
        // Remove underscore _text_
        .replace(/_([^_]+)_/g, '$1')
        // Remove strikethrough ~text~
        .replace(/~([^~]+)~/g, '$1')
        // Remove headers
        .replace(/^\s*#{1,6}\s+/gm, '')
        // Remove links [text](url) -> text
        .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
        // Remove any remaining standalone asterisks at end of lines
        .replace(/\*+\s*$/gm, '')
        // Remove any remaining asterisks at start of lines (not bullets)
        .replace(/^\s*\*+\s*(?=[A-Z])/gm, '')
        // Final cleanup: remove any remaining lone asterisks
        .replace(/\s*\*\s*/g, ' ')
        .trim();

    return cleaned;
}

// Component to render text with technical term tooltips
function TextWithTooltips({ text }: { text: string }) {
    // Clean markdown first
    const cleanedText = cleanMarkdown(text);

    // Find and highlight technical terms
    const elements: React.ReactNode[] = [];
    let remainingText = cleanedText;
    let keyIndex = 0;

    // Sort terms by length (longest first) to avoid partial matches
    const sortedTerms = Object.keys(technicalTerms).sort((a, b) => b.length - a.length);

    while (remainingText.length > 0) {
        let earliestMatch: { term: string; index: number } | null = null;

        for (const term of sortedTerms) {
            const regex = new RegExp(`\\b${term}\\b`, 'i');
            const match = remainingText.match(regex);
            if (match && match.index !== undefined) {
                if (!earliestMatch || match.index < earliestMatch.index) {
                    earliestMatch = { term, index: match.index };
                }
            }
        }

        if (earliestMatch) {
            // Add text before the match
            if (earliestMatch.index > 0) {
                elements.push(
                    <span key={keyIndex++}>{remainingText.slice(0, earliestMatch.index)}</span>
                );
            }

            // Add the tooltip for the technical term
            const matchedText = remainingText.slice(
                earliestMatch.index,
                earliestMatch.index + earliestMatch.term.length
            );

            elements.push(
                <Tooltip key={keyIndex++}>
                    <TooltipTrigger asChild>
                        <span className="underline decoration-dotted decoration-primary/50 underline-offset-2 cursor-help text-primary/90 font-medium">
                            {matchedText}
                        </span>
                    </TooltipTrigger>
                    <TooltipContent side="top" className="max-w-xs text-sm">
                        <div className="flex items-start gap-2">
                            <Info className="w-4 h-4 shrink-0 mt-0.5 text-primary" />
                            <div>
                                <p className="font-medium capitalize mb-1">{earliestMatch.term}</p>
                                <p className="text-muted-foreground">{technicalTerms[earliestMatch.term.toLowerCase()]}</p>
                            </div>
                        </div>
                    </TooltipContent>
                </Tooltip>
            );

            remainingText = remainingText.slice(earliestMatch.index + earliestMatch.term.length);
        } else {
            // No more matches, add remaining text
            elements.push(<span key={keyIndex++}>{remainingText}</span>);
            break;
        }
    }

    return <>{elements}</>;
}

const sectionConfig = {
    summary: {
        icon: BarChart3,
        title: "What the Chart Shows",
        subtitle: "Overview of the analysis",
        bgClass: "bg-primary/5",
        borderClass: "border-primary/20",
        iconClass: "text-primary",
    },
    pattern: {
        icon: TrendingUp,
        title: "Pattern Detected",
        subtitle: "The main formation spotted",
        bgClass: "bg-success/5",
        borderClass: "border-success/20",
        iconClass: "text-success",
    },
    signals: {
        icon: CheckCircle2,
        title: "Supporting Evidence",
        subtitle: "Additional signs that confirm this",
        bgClass: "bg-primary/5",
        borderClass: "border-primary/20",
        iconClass: "text-primary",
    },
    caution: {
        icon: AlertTriangle,
        title: "Watch Out For",
        subtitle: "Potential risks to be aware of",
        bgClass: "bg-warning/5",
        borderClass: "border-warning/20",
        iconClass: "text-warning",
    },
    levels: {
        icon: Target,
        title: "Key Price Levels",
        subtitle: "Important prices to watch",
        bgClass: "bg-muted/50",
        borderClass: "border-border",
        iconClass: "text-muted-foreground",
    },
    recommendation: {
        icon: Lightbulb,
        title: "What This Means",
        subtitle: "Suggested interpretation",
        bgClass: "bg-success/5",
        borderClass: "border-success/20",
        iconClass: "text-success",
    },
    text: {
        icon: Shield,
        title: "Additional Notes",
        subtitle: "More context",
        bgClass: "bg-muted/30",
        borderClass: "border-border/50",
        iconClass: "text-muted-foreground",
    },
};

export function AIAnalysisDisplay({ reasoning, className }: AIAnalysisDisplayProps) {
    const sections = useMemo(() => parseAnalysisText(reasoning), [reasoning]);

    return (
        <div className={cn("space-y-4", className)}>
            {/* Header */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <div className="relative">
                        <div className="w-10 h-10 rounded-xl gradient-primary flex items-center justify-center glow-primary">
                            <span className="text-xs font-bold text-primary-foreground">AI</span>
                        </div>
                        <div className="absolute -bottom-1 -right-1 w-4 h-4 rounded-full bg-success flex items-center justify-center">
                            <CheckCircle2 className="w-2.5 h-2.5 text-white" />
                        </div>
                    </div>
                    <div>
                        <h3 className="font-semibold text-lg">AI Analysis</h3>
                        <p className="text-xs text-muted-foreground">Hover underlined terms for explanations</p>
                    </div>
                </div>

                {/* Help indicator */}
                <Tooltip>
                    <TooltipTrigger asChild>
                        <button className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-muted/50 text-xs text-muted-foreground hover:text-foreground transition-colors">
                            <HelpCircle className="w-3.5 h-3.5" />
                            <span className="hidden sm:inline">New to trading?</span>
                        </button>
                    </TooltipTrigger>
                    <TooltipContent side="left" className="max-w-xs">
                        <p className="text-sm">
                            <strong>Tip:</strong> Hover over underlined terms to see beginner-friendly explanations of trading concepts.
                        </p>
                    </TooltipContent>
                </Tooltip>
            </div>

            {/* Sections */}
            <div className="space-y-3">
                {sections.map((section, idx) => {
                    const config = sectionConfig[section.type];
                    const Icon = config.icon;

                    // Use parsed title if available, otherwise use config title
                    const displayTitle = section.title || config.title;

                    // Render content with clean, app-native styling (no raw LLM text)
                    const renderContent = (content: string) => {
                        // Split into lines and filter empty ones
                        const lines = content.split('\n').filter(line => line.trim());

                        // Clean up markdown artifacts for app-native display
                        const cleanText = (text: string): string => {
                            return text
                                .replace(/\*\*\*([^*]+)\*\*\*/g, '$1')   // Remove bold+italic
                                .replace(/\*\*([^*]+)\*\*/g, '$1')      // Remove bold
                                .replace(/\*\*/g, '')                    // Remove unclosed bold
                                .replace(/\*([^*]+)\*/g, '$1')          // Remove italic
                                .replace(/`([^`]+)`/g, '$1')            // Remove code
                                .replace(/\*+\s*$/g, '')                 // Remove trailing asterisks
                                .replace(/^\s*[-•*]\s*/, '')             // Remove bullet markers
                                .replace(/^\s*\d+\.\s*/, '')             // Remove number markers
                                .replace(/\s*\*\s*/g, ' ')               // Remove lone asterisks
                                .trim();
                        };

                        // Group consecutive items of same type for better layout
                        const elements: React.ReactNode[] = [];
                        let currentList: { type: 'bullet' | 'number', items: string[] } | null = null;

                        const flushList = () => {
                            if (currentList && currentList.items.length > 0) {
                                if (currentList.type === 'bullet') {
                                    elements.push(
                                        <div key={`list-${elements.length}`} className="rounded-xl bg-gradient-to-br from-muted/50 to-muted/20 p-4 my-3 border border-border/30">
                                            <div className="space-y-2">
                                                {currentList.items.map((item, i) => (
                                                    <div key={i} className="flex gap-3 items-start group">
                                                        <div className={cn(
                                                            "w-2 h-2 rounded-full mt-1.5 shrink-0 ring-2 ring-offset-1 ring-offset-background",
                                                            config.iconClass.replace('text-', 'bg-'),
                                                            config.iconClass.replace('text-', 'ring-')
                                                        )} />
                                                        <span className="text-foreground/90 leading-relaxed flex-1">
                                                            <TextWithTooltips text={cleanText(item)} />
                                                        </span>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    );
                                } else {
                                    elements.push(
                                        <div key={`list-${elements.length}`} className="space-y-2 my-2">
                                            {currentList.items.map((item, i) => (
                                                <div key={i} className="flex gap-3 items-start">
                                                    <span className={cn(
                                                        "flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold",
                                                        "bg-gradient-to-br from-primary/20 to-primary/10 text-primary"
                                                    )}>
                                                        {i + 1}
                                                    </span>
                                                    <span className="flex-1 pt-0.5 text-foreground/90">
                                                        <TextWithTooltips text={cleanText(item)} />
                                                    </span>
                                                </div>
                                            ))}
                                        </div>
                                    );
                                }
                                currentList = null;
                            }
                        };

                        for (let lineIdx = 0; lineIdx < lines.length; lineIdx++) {
                            const line = lines[lineIdx];
                            const trimmedLine = line.trim();

                            // Skip empty lines
                            if (!trimmedLine) continue;

                            // Skip horizontal dividers and markdown artifacts
                            if (trimmedLine === '---' || trimmedLine.startsWith('===')) continue;

                            // Check for bullet points
                            if (trimmedLine.startsWith('-') || trimmedLine.startsWith('*') || trimmedLine.startsWith('•')) {
                                const bulletContent = trimmedLine.replace(/^[-*•]\s*/, '');
                                if (!currentList || currentList.type !== 'bullet') {
                                    flushList();
                                    currentList = { type: 'bullet', items: [] };
                                }
                                currentList.items.push(bulletContent);
                                continue;
                            }

                            // Check for numbered list items
                            const numMatch = trimmedLine.match(/^(\d+)\.\s*(.*)$/);
                            if (numMatch) {
                                if (!currentList || currentList.type !== 'number') {
                                    flushList();
                                    currentList = { type: 'number', items: [] };
                                }
                                currentList.items.push(numMatch[2]);
                                continue;
                            }

                            // Flush any pending list
                            flushList();

                            // Check for key-value pairs (Label: Value or **Label**: Value)
                            const kvMatch = trimmedLine.match(/^(?:\*\*)?([^:*]+)(?:\*\*)?:\s*(.+)$/);
                            if (kvMatch && kvMatch[1].length < 40) {
                                elements.push(
                                    <div key={lineIdx} className="flex flex-wrap gap-x-2 py-1.5 border-b border-border/30 last:border-0">
                                        <span className="font-medium text-foreground/70 min-w-[100px]">
                                            {cleanText(kvMatch[1])}
                                        </span>
                                        <span className="text-foreground flex-1">
                                            <TextWithTooltips text={cleanText(kvMatch[2])} />
                                        </span>
                                    </div>
                                );
                                continue;
                            }

                            // Check for confidence calculations (show as special badge)
                            const confMatch = trimmedLine.match(/=\s*(\d+)%?$/);
                            if (confMatch && trimmedLine.toLowerCase().includes('confidence')) {
                                const conf = parseInt(confMatch[1]);
                                elements.push(
                                    <div key={lineIdx} className="flex items-center gap-2 py-2 my-1">
                                        <span className="text-muted-foreground text-xs">Confidence Score</span>
                                        <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                                            <div
                                                className={cn(
                                                    "h-full rounded-full transition-all",
                                                    conf >= 70 ? "bg-success" : conf >= 50 ? "bg-warning" : "bg-destructive"
                                                )}
                                                style={{ width: `${Math.min(conf, 100)}%` }}
                                            />
                                        </div>
                                        <span className={cn(
                                            "text-sm font-bold",
                                            conf >= 70 ? "text-success" : conf >= 50 ? "text-warning" : "text-destructive"
                                        )}>
                                            {conf}%
                                        </span>
                                    </div>
                                );
                                continue;
                            }

                            // Regular paragraph - clean and render
                            elements.push(
                                <p key={lineIdx} className="py-1 text-foreground/90 leading-relaxed">
                                    <TextWithTooltips text={cleanText(trimmedLine)} />
                                </p>
                            );
                        }

                        // Don't forget to flush any remaining list
                        flushList();

                        return elements;
                    };

                    return (
                        <div
                            key={idx}
                            className={cn(
                                "rounded-xl border p-4 transition-all duration-300",
                                "hover:shadow-md",
                                config.bgClass,
                                config.borderClass,
                                "animate-fade-up"
                            )}
                            style={{ animationDelay: `${idx * 100}ms` }}
                        >
                            <div className="flex items-start gap-3">
                                <div className={cn(
                                    "w-9 h-9 rounded-lg flex items-center justify-center shrink-0",
                                    config.bgClass === "bg-muted/30" ? "bg-muted" : config.bgClass.replace("/5", "/15")
                                )}>
                                    <Icon className={cn("w-4.5 h-4.5", config.iconClass)} />
                                </div>
                                <div className="space-y-2 flex-1 min-w-0">
                                    <div>
                                        <h4 className="text-sm font-semibold">{displayTitle}</h4>
                                        <p className="text-xs text-muted-foreground">{config.subtitle}</p>
                                    </div>

                                    <div className="text-sm leading-relaxed text-foreground/85">
                                        {renderContent(section.content)}
                                    </div>
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* Legend */}
            <div className="flex flex-wrap items-center gap-3 pt-3 border-t border-border/50 text-xs text-muted-foreground">
                <div className="flex items-center gap-1.5">
                    <div className="w-3 h-3 rounded-full bg-success/20" />
                    <span>Bullish signals</span>
                </div>
                <div className="flex items-center gap-1.5">
                    <div className="w-3 h-3 rounded-full bg-warning/20" />
                    <span>Caution areas</span>
                </div>
                <div className="flex items-center gap-1.5">
                    <span className="underline decoration-dotted decoration-primary/50">Underlined</span>
                    <span>= has tooltip</span>
                </div>
            </div>

            {/* Footer Note */}
            <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-muted/30 text-xs text-muted-foreground">
                <Shield className="w-3.5 h-3.5 shrink-0" />
                <span>This is AI-generated analysis for educational purposes. Always do your own research before making trading decisions.</span>
            </div>
        </div>
    );
}
