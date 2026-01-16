import { useState, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import {
  TrendingUp,
  TrendingDown,
  Minus,
  Eye,
  Trash2,
  Loader2,
  RefreshCw,
  Calendar,
  Search,
  Filter,
  MoreVertical,
  FileText,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { AppLayout } from "@/components/layout/AppLayout";
import { cn } from "@/lib/utils";
import { useAuth } from "@/contexts/AuthContext";
import { analysisApi, AnalysisResult, ApiError } from "@/services/api";
import { useToast } from "@/hooks/use-toast";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

const BiasIcon = {
  bullish: TrendingUp,
  bearish: TrendingDown,
  neutral: Minus,
};

const biasStyles = {
  bullish: {
    bg: "bg-success/10",
    text: "text-success",
    border: "border-success/20",
  },
  bearish: {
    bg: "bg-destructive/10",
    text: "text-destructive",
    border: "border-destructive/20",
  },
  neutral: {
    bg: "bg-muted",
    text: "text-muted-foreground",
    border: "border-border",
  },
};

function formatTimeAgo(date: Date) {
  const seconds = Math.floor((Date.now() - date.getTime()) / 1000);
  if (seconds < 60) return "Just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days}d ago`;
  return date.toLocaleDateString();
}

export default function HistoryPage() {
  const [historyItems, setHistoryItems] = useState<AnalysisResult[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [searchQuery, setSearchQuery] = useState("");
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const { isLoggedIn } = useAuth();
  const { toast } = useToast();
  const navigate = useNavigate();

  const fetchHistory = useCallback(async () => {
    if (!isLoggedIn) {
      setHistoryItems([]);
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    try {
      const response = await analysisApi.getHistory(page, 12);
      setHistoryItems(response.items);
      setTotalPages(response.pages);
    } catch (error) {
      console.error('Failed to fetch history:', error);
      if (error instanceof ApiError && error.status === 401) {
        toast({
          title: "Session Expired",
          description: "Please login again",
          variant: "destructive",
        });
      }
    } finally {
      setIsLoading(false);
    }
  }, [isLoggedIn, page, toast]);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  const handleDelete = async (id: string) => {
    setDeletingId(id);
    try {
      await analysisApi.deleteAnalysis(id);
      setHistoryItems(prev => prev.filter(item => item.id !== id));
      toast({
        title: "Deleted",
        description: "Analysis removed from history",
      });
    } catch (error) {
      toast({
        title: "Delete Failed",
        description: "Could not delete analysis",
        variant: "destructive",
      });
    } finally {
      setDeletingId(null);
    }
  };

  const handleView = (id: string) => {
    // Navigate to detailed view (could be a modal or separate page)
    toast({
      title: "View Analysis",
      description: `Opening analysis ${id.slice(0, 8)}...`,
    });
  };

  const filteredItems = historyItems.filter(item =>
    item.patterns.some(p => p.name.toLowerCase().includes(searchQuery.toLowerCase())) ||
    item.market_bias.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <AppLayout
      title="Analysis History"
      subtitle="View and manage your past chart analyses"
    >
      <div className="space-y-6">
        {/* Search and Filters */}
        <div className="flex flex-col sm:flex-row gap-4 animate-fade-up">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search patterns, bias..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9 h-11"
            />
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              onClick={fetchHistory}
              disabled={isLoading}
              className="gap-2"
            >
              <RefreshCw className={cn("h-4 w-4", isLoading && "animate-spin")} />
              <span className="hidden sm:inline">Refresh</span>
            </Button>
          </div>
        </div>

        {/* Content */}
        {isLoading ? (
          // Loading Grid
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {[...Array(6)].map((_, i) => (
              <div
                key={i}
                className="glass-card p-4 animate-pulse"
              >
                <div className="aspect-video bg-muted rounded-lg mb-4" />
                <div className="h-4 bg-muted rounded w-3/4 mb-2" />
                <div className="h-3 bg-muted rounded w-1/2" />
              </div>
            ))}
          </div>
        ) : filteredItems.length === 0 ? (
          // Empty State
          <div className="glass-card p-12 text-center animate-fade-up">
            <div className="w-16 h-16 rounded-2xl bg-muted flex items-center justify-center mx-auto mb-4">
              <FileText className="h-8 w-8 text-muted-foreground" />
            </div>
            <h3 className="text-xl font-semibold mb-2">No analyses yet</h3>
            <p className="text-muted-foreground mb-6 max-w-md mx-auto">
              {searchQuery
                ? "No analyses match your search. Try different keywords."
                : "Upload your first chart to get started with AI-powered analysis."}
            </p>
            {!searchQuery && (
              <Button
                onClick={() => navigate("/dashboard")}
                className="gradient-primary gap-2"
              >
                Upload Chart
              </Button>
            )}
          </div>
        ) : (
          // History Grid
          <>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {filteredItems.map((item, index) => {
                const bias = item.market_bias as keyof typeof biasStyles;
                const Icon = BiasIcon[bias] || Minus;
                const styles = biasStyles[bias] || biasStyles.neutral;

                return (
                  <div
                    key={item.id}
                    className={cn(
                      "glass-card overflow-hidden group animate-fade-up",
                      `animation-delay-${Math.min(index * 100, 500)}`
                    )}
                    style={{ animationDelay: `${index * 50}ms` }}
                  >
                    {/* Image */}
                    <div className="relative aspect-video bg-muted overflow-hidden">
                      {item.image_url ? (
                        <img
                          src={item.image_url}
                          alt="Chart"
                          className="w-full h-full object-cover transition-transform group-hover:scale-105"
                        />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center">
                          <TrendingUp className="h-12 w-12 text-muted-foreground/30" />
                        </div>
                      )}
                      {/* Overlay on hover */}
                      <div className="absolute inset-0 bg-gradient-to-t from-background/80 via-background/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex items-end justify-center pb-4">
                        <div className="flex gap-2">
                          <Button
                            size="sm"
                            variant="secondary"
                            onClick={() => handleView(item.id)}
                            className="gap-1"
                          >
                            <Eye className="h-4 w-4" />
                            View
                          </Button>
                        </div>
                      </div>
                    </div>

                    {/* Content */}
                    <div className="p-4 space-y-3">
                      {/* Bias Badge */}
                      <div className="flex items-center justify-between">
                        <div
                          className={cn(
                            "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-sm font-medium border",
                            styles.bg,
                            styles.text,
                            styles.border
                          )}
                        >
                          <Icon className="h-3.5 w-3.5" />
                          <span className="capitalize">{item.market_bias}</span>
                        </div>
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="icon" className="h-8 w-8">
                              <MoreVertical className="h-4 w-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuItem onClick={() => handleView(item.id)}>
                              <Eye className="h-4 w-4 mr-2" />
                              View Details
                            </DropdownMenuItem>
                            <DropdownMenuItem
                              onClick={() => handleDelete(item.id)}
                              className="text-destructive focus:text-destructive"
                              disabled={deletingId === item.id}
                            >
                              {deletingId === item.id ? (
                                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                              ) : (
                                <Trash2 className="h-4 w-4 mr-2" />
                              )}
                              Delete
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>

                      {/* Patterns */}
                      <div className="flex flex-wrap gap-1.5">
                        {item.patterns.slice(0, 3).map((pattern, i) => (
                          <span
                            key={i}
                            className="text-xs px-2 py-0.5 bg-muted rounded-md text-muted-foreground"
                          >
                            {pattern.name}
                          </span>
                        ))}
                        {item.patterns.length > 3 && (
                          <span className="text-xs px-2 py-0.5 bg-muted rounded-md text-muted-foreground">
                            +{item.patterns.length - 3}
                          </span>
                        )}
                      </div>

                      {/* Footer */}
                      <div className="flex items-center justify-between text-xs text-muted-foreground pt-2 border-t border-border/50">
                        <div className="flex items-center gap-1">
                          <Calendar className="h-3 w-3" />
                          {formatTimeAgo(new Date(item.created_at))}
                        </div>
                        <div className="font-medium">
                          {item.confidence}% confidence
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex items-center justify-center gap-2 pt-4">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPage(p => Math.max(1, p - 1))}
                  disabled={page === 1}
                >
                  Previous
                </Button>
                <span className="text-sm text-muted-foreground px-4">
                  Page {page} of {totalPages}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                  disabled={page === totalPages}
                >
                  Next
                </Button>
              </div>
            )}
          </>
        )}
      </div>
    </AppLayout>
  );
}
