import { useState, useCallback, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  TrendingUp,
  LayoutDashboard,
  Upload,
  History,
  Settings,
  LogOut,
  Menu,
  X,
  LogIn,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { UploadDropzone } from "@/components/UploadDropzone";
import { AnalysisCard } from "@/components/AnalysisCard";
import { AnalysisLoadingSkeleton } from "@/components/LoadingSkeleton";
import { ThemeToggle } from "@/components/ThemeToggle";
import { cn } from "@/lib/utils";
import { useAuth } from "@/contexts/AuthContext";
import { analysisApi, AnalysisResult, ApiError } from "@/services/api";
import { useToast } from "@/hooks/use-toast";

const navItems = [
  { icon: LayoutDashboard, label: "Dashboard", href: "/dashboard", active: true },
  { icon: Upload, label: "Upload", href: "/dashboard" },
  { icon: History, label: "History", href: "/history" },
  { icon: Settings, label: "Settings", href: "/settings" },
];

// Convert API result to component format
function formatAnalysisResult(apiResult: AnalysisResult, imageUrl: string) {
  return {
    imageUrl,
    patterns: apiResult.patterns.map(p => ({
      name: p.name,
      bias: p.type as 'bullish' | 'bearish' | 'neutral',
    })),
    marketBias: apiResult.market_bias as 'bullish' | 'bearish' | 'neutral',
    confidence: apiResult.confidence,
    reasoning: apiResult.reasoning || '',
    timestamp: new Date(apiResult.created_at),
  };
}

export default function Dashboard() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<ReturnType<typeof formatAnalysisResult> | null>(null);
  const [analysisId, setAnalysisId] = useState<string | null>(null);

  const { user, isLoggedIn, logout, isLoading } = useAuth();
  const { toast } = useToast();
  const navigate = useNavigate();

  // MEMORY MANAGEMENT: Cleanup blob URLs to prevent memory leaks
  // Revoke the previous URL when file changes or component unmounts
  useEffect(() => {
    if (selectedFile) {
      const url = URL.createObjectURL(selectedFile);
      setPreviewUrl(url);

      // Cleanup: revoke the URL when file changes or component unmounts
      return () => {
        URL.revokeObjectURL(url);
      };
    } else {
      setPreviewUrl(null);
    }
  }, [selectedFile]);

  const handleFileSelect = useCallback((file: File) => {
    setSelectedFile(file);
    setResult(null);
    setAnalysisId(null);
  }, []);

  const handleAnalyze = useCallback(async () => {
    if (!selectedFile) return;

    // Check if user is logged in
    if (!isLoggedIn) {
      toast({
        title: "Login Required",
        description: "Please login to analyze charts. For demo, analysis still works!",
        variant: "default",
      });
    }

    setIsAnalyzing(true);

    try {
      // Call the real backend API
      const response = await analysisApi.uploadChart(selectedFile);

      // Use the previewUrl from state (managed by useEffect with cleanup)
      // This prevents memory leaks from orphaned blob URLs
      setResult(formatAnalysisResult(response.analysis, previewUrl || ''));
      setAnalysisId(response.analysis.id);

      toast({
        title: "Analysis Complete",
        description: `Detected ${response.analysis.patterns.length} patterns with ${response.analysis.confidence}% confidence`,
      });
    } catch (error) {
      console.error('Analysis failed:', error);

      if (error instanceof ApiError) {
        if (error.status === 401) {
          toast({
            title: "Authentication Required",
            description: "Please login to analyze charts",
            variant: "destructive",
          });
        } else {
          toast({
            title: "Analysis Failed",
            description: error.message,
            variant: "destructive",
          });
        }
      } else {
        toast({
          title: "Error",
          description: "Failed to analyze chart. Please try again.",
          variant: "destructive",
        });
      }
    } finally {
      setIsAnalyzing(false);
    }
  }, [selectedFile, isLoggedIn, toast]);

  const handleNewUpload = () => {
    setSelectedFile(null);
    setResult(null);
    setAnalysisId(null);
  };

  const handleSave = () => {
    if (analysisId) {
      toast({
        title: "Saved!",
        description: "Analysis saved to your history",
      });
      navigate('/history');
    }
  };

  const handleLogout = async () => {
    try {
      await logout();
      toast({
        title: "Logged Out",
        description: "See you next time!",
      });
    } catch (error) {
      console.error('Logout failed:', error);
    }
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
            {isLoggedIn ? (
              <button
                onClick={handleLogout}
                className="flex items-center gap-3 px-4 py-3 w-full rounded-lg text-sidebar-foreground hover:bg-sidebar-accent/50 transition-colors"
              >
                <LogOut className="w-5 h-5" />
                <span className="font-medium">Sign Out</span>
              </button>
            ) : (
              <Link
                to="/"
                className="flex items-center gap-3 px-4 py-3 w-full rounded-lg text-sidebar-foreground hover:bg-sidebar-accent/50 transition-colors"
              >
                <LogIn className="w-5 h-5" />
                <span className="font-medium">Sign In</span>
              </Link>
            )}
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
                <span className="text-sm font-medium text-primary">
                  {user?.full_name?.[0] || user?.email?.[0]?.toUpperCase() || 'U'}
                </span>
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
                  {!isLoggedIn && (
                    <p className="text-sm text-amber-500 mt-2">
                      Demo mode active - login for full features
                    </p>
                  )}
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

