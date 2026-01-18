import { useState, useCallback, useEffect } from "react";
import { useNavigate, Link } from "react-router-dom";
import {
  Sparkles,
  ImagePlus,
  Trash2,
  Save,
  RotateCcw,
  Target,
  TrendingUp,
  TrendingDown,
  BarChart3,
  Clock,
  ChevronRight,
  Activity,
  Zap
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { UploadDropzone } from "@/components/UploadDropzone";
import { AnalysisCard } from "@/components/AnalysisCard";
import { AnalysisLoadingSkeleton } from "@/components/LoadingSkeleton";
import { AppLayout } from "@/components/layout/AppLayout";
import { cn } from "@/lib/utils";
import { useAuth } from "@/contexts/AuthContext";
import { analysisApi, userApi, AnalysisResult, ApiError } from "@/services/api";
import { useToast } from "@/hooks/use-toast";

// Stats interface
interface DashboardStats {
  totalAnalyses: number;
  bullishCount: number;
  bearishCount: number;
  neutralCount: number;
  avgConfidence: number;
  thisWeekCount: number;
}

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
    entryTiming: apiResult.entry_timing ? {
      signal: apiResult.entry_timing.signal as 'wait' | 'prepare' | 'ready' | 'now',
      timing_description: apiResult.entry_timing.timing_description,
      conditions: apiResult.entry_timing.conditions,
      entry_price_zone: apiResult.entry_timing.entry_price_zone,
      stop_loss: apiResult.entry_timing.stop_loss,
      take_profit: apiResult.entry_timing.take_profit,
      risk_reward: apiResult.entry_timing.risk_reward,
      timeframe: apiResult.entry_timing.timeframe,
    } : undefined,
  };
}

export default function Dashboard() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<ReturnType<typeof formatAnalysisResult> | null>(null);
  const [analysisId, setAnalysisId] = useState<string | null>(null);
  const [recentAnalyses, setRecentAnalyses] = useState<AnalysisResult[]>([]);
  const [stats, setStats] = useState<DashboardStats>({
    totalAnalyses: 0,
    bullishCount: 0,
    bearishCount: 0,
    neutralCount: 0,
    avgConfidence: 0,
    thisWeekCount: 0,
  });
  const [isLoadingStats, setIsLoadingStats] = useState(true);

  const { user, isLoggedIn } = useAuth();
  const { toast } = useToast();
  const navigate = useNavigate();

  // Fetch stats and recent analyses
  useEffect(() => {
    const fetchDashboardData = async () => {
      if (!isLoggedIn) {
        setIsLoadingStats(false);
        return;
      }

      try {
        // Fetch stats from dedicated endpoint
        const [statsResponse, historyResponse] = await Promise.all([
          userApi.getStats(),
          analysisApi.getHistory(1, 3),
        ]);

        setRecentAnalyses(historyResponse.items);

        // Use API stats
        setStats({
          totalAnalyses: statsResponse.total_analyses,
          bullishCount: statsResponse.bias_distribution?.bullish || 0,
          bearishCount: statsResponse.bias_distribution?.bearish || 0,
          neutralCount: statsResponse.bias_distribution?.neutral || 0,
          avgConfidence: statsResponse.avg_confidence || 0,
          thisWeekCount: statsResponse.this_week_count || 0,
        });
      } catch (error) {
        console.error('Failed to fetch dashboard data:', error);
      } finally {
        setIsLoadingStats(false);
      }
    };

    fetchDashboardData();
  }, [isLoggedIn]);

  // MEMORY MANAGEMENT: Cleanup blob URLs to prevent memory leaks
  useEffect(() => {
    if (selectedFile) {
      const url = URL.createObjectURL(selectedFile);
      setPreviewUrl(url);
      return () => URL.revokeObjectURL(url);
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

    if (!isLoggedIn) {
      toast({
        title: "Login Required",
        description: "Please login to analyze charts. For demo, analysis still works!",
        variant: "default",
      });
    }

    setIsAnalyzing(true);

    try {
      const response = await analysisApi.uploadChart(selectedFile);
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
  }, [selectedFile, isLoggedIn, toast, previewUrl]);

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

  return (
    <AppLayout
      title={`Welcome back${user?.full_name ? `, ${user.full_name.split(' ')[0]}` : ''}!`}
      subtitle="Upload a candlestick chart to get instant AI-powered analysis"
    >
      <div className="grid gap-6 lg:gap-8">
        {/* Stats Section */}
        {isLoggedIn && !selectedFile && !result && (
          <section className="animate-fade-up">
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              {/* Total Analyses */}
              <div className="glass-card p-5 hover:border-primary/30 transition-all group">
                <div className="flex items-center justify-between mb-3">
                  <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
                    <BarChart3 className="h-5 w-5 text-primary" />
                  </div>
                  {isLoadingStats ? (
                    <div className="h-6 w-12 bg-muted animate-pulse rounded" />
                  ) : (
                    <span className="text-2xl font-bold">{stats.totalAnalyses}</span>
                  )}
                </div>
                <h3 className="text-sm font-medium text-muted-foreground">Total Analyses</h3>
              </div>

              {/* This Week */}
              <div className="glass-card p-5 hover:border-primary/30 transition-all group">
                <div className="flex items-center justify-between mb-3">
                  <div className="w-10 h-10 rounded-xl bg-blue-500/10 flex items-center justify-center">
                    <Clock className="h-5 w-5 text-blue-500" />
                  </div>
                  {isLoadingStats ? (
                    <div className="h-6 w-12 bg-muted animate-pulse rounded" />
                  ) : (
                    <span className="text-2xl font-bold">{stats.thisWeekCount}</span>
                  )}
                </div>
                <h3 className="text-sm font-medium text-muted-foreground">This Week</h3>
              </div>

              {/* Avg Confidence */}
              <div className="glass-card p-5 hover:border-primary/30 transition-all group">
                <div className="flex items-center justify-between mb-3">
                  <div className="w-10 h-10 rounded-xl bg-amber-500/10 flex items-center justify-center">
                    <Zap className="h-5 w-5 text-amber-500" />
                  </div>
                  {isLoadingStats ? (
                    <div className="h-6 w-12 bg-muted animate-pulse rounded" />
                  ) : (
                    <span className="text-2xl font-bold">{stats.avgConfidence}%</span>
                  )}
                </div>
                <h3 className="text-sm font-medium text-muted-foreground">Avg Confidence</h3>
              </div>

              {/* Market Bias Distribution */}
              <div className="glass-card p-5 hover:border-primary/30 transition-all group">
                <div className="flex items-center justify-between mb-3">
                  <div className="w-10 h-10 rounded-xl bg-emerald-500/10 flex items-center justify-center">
                    <Activity className="h-5 w-5 text-emerald-500" />
                  </div>
                  {isLoadingStats ? (
                    <div className="h-6 w-20 bg-muted animate-pulse rounded" />
                  ) : (
                    <div className="flex items-center gap-2 text-sm">
                      <span className="text-emerald-500 font-medium">{stats.bullishCount}↑</span>
                      <span className="text-destructive font-medium">{stats.bearishCount}↓</span>
                      <span className="text-muted-foreground">{stats.neutralCount}−</span>
                    </div>
                  )}
                </div>
                <h3 className="text-sm font-medium text-muted-foreground">Bias Distribution</h3>
              </div>
            </div>

            {/* Recent Analyses */}
            {recentAnalyses.length > 0 && (
              <div className="mt-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold">Recent Analyses</h3>
                  <Link to="/history" className="text-sm text-primary hover:underline flex items-center gap-1">
                    View all <ChevronRight className="h-4 w-4" />
                  </Link>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {recentAnalyses.map((item) => (
                    <div
                      key={item.id}
                      className="glass-card p-4 hover:border-primary/30 transition-all cursor-pointer group"
                      onClick={() => navigate('/history')}
                    >
                      <div className="flex items-center gap-3">
                        <div className={cn(
                          "w-10 h-10 rounded-lg flex items-center justify-center shrink-0",
                          item.market_bias === 'bullish' && "bg-emerald-500/10",
                          item.market_bias === 'bearish' && "bg-destructive/10",
                          item.market_bias === 'neutral' && "bg-muted"
                        )}>
                          {item.market_bias === 'bullish' ? (
                            <TrendingUp className="h-5 w-5 text-emerald-500" />
                          ) : item.market_bias === 'bearish' ? (
                            <TrendingDown className="h-5 w-5 text-destructive" />
                          ) : (
                            <Activity className="h-5 w-5 text-muted-foreground" />
                          )}
                        </div>
                        <div className="min-w-0 flex-1">
                          <p className="font-medium capitalize text-sm group-hover:text-primary transition-colors">
                            {item.market_bias} Signal
                          </p>
                          <p className="text-xs text-muted-foreground">
                            {item.confidence}% confidence
                          </p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </section>
        )}

        {/* Upload Section */}
        <section className="animate-fade-up">
          <div className="glass-card p-6 lg:p-8">
            {!selectedFile && !result ? (
              // Upload dropzone
              <div className="space-y-6">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl gradient-primary flex items-center justify-center">
                    <ImagePlus className="h-5 w-5 text-primary-foreground" />
                  </div>
                  <div>
                    <h2 className="text-xl font-semibold">Upload Chart</h2>
                    <p className="text-sm text-muted-foreground">
                      Drag and drop or click to upload
                    </p>
                  </div>
                </div>
                <UploadDropzone onFileSelect={handleFileSelect} />
              </div>
            ) : (
              // Preview and actions
              <div className="space-y-6">
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl gradient-primary flex items-center justify-center">
                      <Sparkles className="h-5 w-5 text-primary-foreground" />
                    </div>
                    <div>
                      <h2 className="text-xl font-semibold">
                        {result ? "Analysis Results" : "Ready to Analyze"}
                      </h2>
                      <p className="text-sm text-muted-foreground">
                        {result
                          ? `${result.patterns.length} patterns detected`
                          : selectedFile?.name}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    {result && (
                      <>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={handleSave}
                          className="gap-2"
                        >
                          <Save className="h-4 w-4" />
                          <span className="hidden sm:inline">View History</span>
                        </Button>
                      </>
                    )}
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleNewUpload}
                      className="gap-2"
                    >
                      <RotateCcw className="h-4 w-4" />
                      <span className="hidden sm:inline">New Upload</span>
                    </Button>
                  </div>
                </div>

                {/* Chart Preview */}
                {previewUrl && !result && (
                  <div className="relative rounded-xl overflow-hidden border border-border bg-muted/30">
                    <img
                      src={previewUrl}
                      alt="Chart preview"
                      className="w-full h-auto max-h-[400px] object-contain"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-background/80 via-transparent to-transparent" />
                    <div className="absolute bottom-4 left-4 right-4 flex justify-center">
                      <Button
                        onClick={handleAnalyze}
                        disabled={isAnalyzing}
                        className="gradient-primary glow-primary gap-2 h-12 px-8 text-base font-medium"
                      >
                        {isAnalyzing ? (
                          <>
                            <div className="h-5 w-5 animate-spin rounded-full border-2 border-current border-t-transparent" />
                            Analyzing...
                          </>
                        ) : (
                          <>
                            <Sparkles className="h-5 w-5" />
                            Analyze Chart
                          </>
                        )}
                      </Button>
                    </div>
                  </div>
                )}

                {/* Loading State */}
                {isAnalyzing && <AnalysisLoadingSkeleton />}

                {/* Results */}
                {result && !isAnalyzing && (
                  <div className="animate-fade-up">
                    <AnalysisCard analysis={result} />
                  </div>
                )}
              </div>
            )}
          </div>
        </section>

        {/* Quick Stats (when no analysis is active) */}
        {!selectedFile && !result && (
          <section className="animate-fade-up animation-delay-200">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {[
                {
                  title: "Pattern Detection",
                  description: "AI identifies candlestick patterns instantly",
                  icon: Target,
                },
                {
                  title: "Market Bias",
                  description: "Get bullish, bearish, or neutral predictions",
                  icon: TrendingUp,
                },
                {
                  title: "Confidence Score",
                  description: "Know how reliable each analysis is",
                  icon: Sparkles,
                },
              ].map((feature, i) => {
                const Icon = feature.icon;
                return (
                  <div
                    key={i}
                    className="glass-card p-5 hover:border-primary/30 transition-all group"
                  >
                    <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center mb-3">
                      <Icon className="h-5 w-5 text-primary" />
                    </div>
                    <h3 className="font-semibold mb-1 group-hover:text-primary transition-colors">
                      {feature.title}
                    </h3>
                    <p className="text-sm text-muted-foreground">
                      {feature.description}
                    </p>
                  </div>
                );
              })}
            </div>
          </section>
        )}
      </div>
    </AppLayout>
  );
}
