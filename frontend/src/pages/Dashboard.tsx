import { useState, useCallback } from "react";
import { Link } from "react-router-dom";
import {
  TrendingUp,
  LayoutDashboard,
  Upload,
  History,
  Settings,
  LogOut,
  Menu,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { UploadDropzone } from "@/components/UploadDropzone";
import { AnalysisCard } from "@/components/AnalysisCard";
import { AnalysisLoadingSkeleton } from "@/components/LoadingSkeleton";
import { ThemeToggle } from "@/components/ThemeToggle";
import { cn } from "@/lib/utils";

const navItems = [
  { icon: LayoutDashboard, label: "Dashboard", href: "/dashboard", active: true },
  { icon: Upload, label: "Upload", href: "/dashboard" },
  { icon: History, label: "History", href: "/history" },
  { icon: Settings, label: "Settings", href: "/settings" },
];

// Mock analysis result
const mockResult = {
  imageUrl: "",
  patterns: [
    { name: "Double Bottom", bias: "bullish" as const },
    { name: "Hammer", bias: "bullish" as const },
    { name: "Rising Wedge", bias: "bearish" as const },
  ],
  marketBias: "bullish" as const,
  confidence: 78,
  reasoning:
    "The chart shows a clear double bottom formation near the $42,500 support level, followed by a bullish hammer candlestick on increased volume. The price has broken above the neckline resistance at $44,200, suggesting a potential continuation to the upside. However, the rising wedge pattern forming on the shorter timeframe indicates some caution is warranted. Key resistance levels to watch are $45,800 and $47,200. A break below $43,500 would invalidate this bullish setup.",
  timestamp: new Date(),
};

export default function Dashboard() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<typeof mockResult | null>(null);

  const handleFileSelect = useCallback((file: File) => {
    setSelectedFile(file);
    setResult(null);
  }, []);

  const handleAnalyze = useCallback(async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);

    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 3000));

    // Create preview URL from selected file
    const imageUrl = URL.createObjectURL(selectedFile);

    setResult({ ...mockResult, imageUrl });
    setIsAnalyzing(false);
  }, [selectedFile]);

  const handleNewUpload = () => {
    setSelectedFile(null);
    setResult(null);
  };

  const handleSave = () => {
    // TODO: Implement save functionality
    console.log("Saving analysis...");
  };

  return (
    <div className="min-h-screen bg-background flex">
      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={cn(
          "fixed lg:static inset-y-0 left-0 z-50 w-64 bg-sidebar border-r border-sidebar-border transform transition-transform duration-300 lg:transform-none",
          sidebarOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"
        )}
      >
        <div className="flex flex-col h-full">
          {/* Logo */}
          <div className="p-6 border-b border-sidebar-border">
            <Link to="/" className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg gradient-primary flex items-center justify-center">
                <TrendingUp className="w-4 h-4 text-primary-foreground" />
              </div>
              <span className="font-semibold text-sidebar-foreground">Candle-Light</span>
            </Link>
          </div>

          {/* Navigation */}
          <nav className="flex-1 p-4 space-y-1">
            {navItems.map((item) => (
              <Link
                key={item.label}
                to={item.href}
                className={cn(
                  "flex items-center gap-3 px-4 py-3 rounded-lg transition-colors",
                  item.active
                    ? "bg-sidebar-accent text-sidebar-primary"
                    : "text-sidebar-foreground hover:bg-sidebar-accent/50"
                )}
              >
                <item.icon className="w-5 h-5" />
                <span className="font-medium">{item.label}</span>
              </Link>
            ))}
          </nav>

          {/* User section */}
          <div className="p-4 border-t border-sidebar-border">
            <button className="flex items-center gap-3 px-4 py-3 w-full rounded-lg text-sidebar-foreground hover:bg-sidebar-accent/50 transition-colors">
              <LogOut className="w-5 h-5" />
              <span className="font-medium">Sign Out</span>
            </button>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 min-w-0">
        {/* Header */}
        <header className="sticky top-0 z-30 bg-background/80 backdrop-blur-xl border-b border-border">
          <div className="flex items-center justify-between px-4 lg:px-8 h-16">
            <div className="flex items-center gap-4">
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="lg:hidden p-2 hover:bg-secondary rounded-lg transition-colors"
              >
                {sidebarOpen ? (
                  <X className="w-5 h-5" />
                ) : (
                  <Menu className="w-5 h-5" />
                )}
              </button>
              <h1 className="text-xl font-semibold">Dashboard</h1>
            </div>
            <div className="flex items-center gap-3">
              <ThemeToggle />
              <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center">
                <span className="text-sm font-medium text-primary">U</span>
              </div>
            </div>
          </div>
        </header>

        {/* Content */}
        <div className="p-4 lg:p-8">
          <div className="max-w-4xl mx-auto">
            {!result && !isAnalyzing && (
              <div className="space-y-6">
                <div className="text-center mb-8">
                  <h2 className="text-2xl font-bold mb-2">Upload Your Chart</h2>
                  <p className="text-muted-foreground">
                    Drop a candlestick chart image to get AI-powered analysis
                  </p>
                </div>
                <div className="glass-card p-6">
                  <UploadDropzone
                    onFileSelect={handleFileSelect}
                    onAnalyze={handleAnalyze}
                    isAnalyzing={isAnalyzing}
                  />
                </div>
              </div>
            )}

            {isAnalyzing && (
              <div className="space-y-6">
                <div className="text-center mb-8">
                  <h2 className="text-2xl font-bold mb-2">Analyzing Chart...</h2>
                  <p className="text-muted-foreground">
                    Our AI is scanning for patterns and generating insights
                  </p>
                </div>
                <AnalysisLoadingSkeleton />
              </div>
            )}

            {result && !isAnalyzing && (
              <div className="space-y-6">
                <div className="text-center mb-8">
                  <h2 className="text-2xl font-bold mb-2">Analysis Complete</h2>
                  <p className="text-muted-foreground">
                    Here's what our AI found in your chart
                  </p>
                </div>
                <AnalysisCard
                  result={result}
                  onNewUpload={handleNewUpload}
                  onSave={handleSave}
                />
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
