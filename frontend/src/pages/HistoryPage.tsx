import { Link } from "react-router-dom";
import {
  TrendingUp,
  LayoutDashboard,
  Upload,
  History,
  Settings,
  LogOut,
  LogIn,
  Menu,
  X,
  Calendar,
  TrendingDown,
  Minus,
  Eye,
  Trash2,
  Loader2,
  RefreshCw,
} from "lucide-react";
import { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/ThemeToggle";
import { cn } from "@/lib/utils";
import { useAuth } from "@/contexts/AuthContext";
import { analysisApi, AnalysisResult, ApiError } from "@/services/api";
import { useToast } from "@/hooks/use-toast";

const navItems = [
  { icon: LayoutDashboard, label: "Dashboard", href: "/dashboard" },
  { icon: Upload, label: "Upload", href: "/dashboard" },
  { icon: History, label: "History", href: "/history", active: true },
  { icon: Settings, label: "Settings", href: "/settings" },
];

const BiasIcon = {
  bullish: TrendingUp,
  bearish: TrendingDown,
  neutral: Minus,
};

const biasColors = {
  bullish: "text-bullish bg-bullish/10",
  bearish: "text-bearish bg-bearish/10",
  neutral: "text-neutral bg-neutral/10",
};

function formatTimeAgo(date: Date) {
  const seconds = Math.floor((Date.now() - date.getTime()) / 1000);
  if (seconds < 60) return "Just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export default function HistoryPage() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [historyItems, setHistoryItems] = useState<AnalysisResult[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);

  const { user, isLoggedIn, logout } = useAuth();
  const { toast } = useToast();

  const fetchHistory = useCallback(async () => {
    if (!isLoggedIn) {
      setHistoryItems([]);
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    try {
      const response = await analysisApi.getHistory(page, 10);
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
    try {
      await analysisApi.deleteAnalysis(id);
      setHistoryItems(prev => prev.filter(item => item.id !== id));
      toast({
        title: "Deleted",
        description: "Analysis removed from history",
      });
    } catch (error) {
      console.error('Failed to delete:', error);
      toast({
        title: "Error",
        description: "Failed to delete analysis",
        variant: "destructive",
      });
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
          <div className="p-6 border-b border-sidebar-border">
            <Link to="/" className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg gradient-primary flex items-center justify-center">
                <TrendingUp className="w-4 h-4 text-primary-foreground" />
              </div>
              <span className="font-semibold text-sidebar-foreground">Candle-Light</span>
            </Link>
          </div>

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
        <header className="sticky top-0 z-30 bg-background/80 backdrop-blur-xl border-b border-border">
          <div className="flex items-center justify-between px-4 lg:px-8 h-16">
            <div className="flex items-center gap-4">
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="lg:hidden p-2 hover:bg-secondary rounded-lg transition-colors"
              >
                {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
              </button>
              <h1 className="text-xl font-semibold">Analysis History</h1>
            </div>
            <div className="flex items-center gap-3">
              <Button variant="ghost" size="sm" onClick={fetchHistory} disabled={isLoading}>
                <RefreshCw className={cn("w-4 h-4", isLoading && "animate-spin")} />
              </Button>
              <ThemeToggle />
              <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center">
                <span className="text-sm font-medium text-primary">
                  {user?.full_name?.[0] || user?.email?.[0]?.toUpperCase() || 'U'}
                </span>
              </div>
            </div>
          </div>
        </header>

        <div className="p-4 lg:p-8">
          <div className="max-w-5xl mx-auto">
            {/* Loading state */}
            {isLoading && (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="w-8 h-8 animate-spin text-primary" />
              </div>
            )}

            {/* Not logged in */}
            {!isLoading && !isLoggedIn && (
              <div className="glass-card p-12 text-center">
                <LogIn className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="text-lg font-semibold mb-2">Login Required</h3>
                <p className="text-muted-foreground mb-6">
                  Please login to view your analysis history.
                </p>
                <Link to="/">
                  <Button variant="hero">
                    Sign In
                  </Button>
                </Link>
              </div>
            )}

            {/* History list */}
            {!isLoading && isLoggedIn && historyItems.length > 0 && (
              <div className="grid gap-4">
                {historyItems.map((item) => {
                  const Icon = BiasIcon[item.market_bias as keyof typeof BiasIcon] || Minus;
                  return (
                    <div
                      key={item.id}
                      className="glass-card p-4 flex flex-col sm:flex-row items-start sm:items-center gap-4 hover:border-primary/30 transition-colors"
                    >
                      <div className="w-full sm:w-32 h-20 rounded-lg overflow-hidden bg-secondary shrink-0">
                        {item.image_url ? (
                          <img
                            src={`http://localhost:8000${item.image_url}`}
                            alt="Chart"
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center">
                            <TrendingUp className="w-8 h-8 text-muted-foreground" />
                          </div>
                        )}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-3 mb-2">
                          <h3 className="font-semibold text-lg">Chart Analysis</h3>
                          <span
                            className={cn(
                              "inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium",
                              biasColors[item.market_bias as keyof typeof biasColors] || biasColors.neutral
                            )}
                          >
                            <Icon className="w-3 h-3" />
                            {item.market_bias.charAt(0).toUpperCase() + item.market_bias.slice(1)}
                          </span>
                        </div>
                        <div className="flex flex-wrap gap-1.5 mb-2">
                          {item.patterns.slice(0, 3).map((pattern, idx) => (
                            <span
                              key={idx}
                              className="px-2 py-0.5 bg-secondary rounded text-xs text-muted-foreground"
                            >
                              {pattern.name}
                            </span>
                          ))}
                          {item.patterns.length > 3 && (
                            <span className="px-2 py-0.5 bg-secondary rounded text-xs text-muted-foreground">
                              +{item.patterns.length - 3} more
                            </span>
                          )}
                        </div>
                        <div className="flex items-center gap-4 text-sm text-muted-foreground">
                          <span className="font-mono">{item.confidence}% confidence</span>
                          <span className="flex items-center gap-1">
                            <Calendar className="w-3.5 h-3.5" />
                            {formatTimeAgo(new Date(item.created_at))}
                          </span>
                        </div>
                      </div>
                      <div className="flex gap-2">
                        <Button variant="ghost" size="sm">
                          <Eye className="w-4 h-4" />
                          View
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-destructive hover:text-destructive"
                          onClick={() => handleDelete(item.id)}
                        >
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                  );
                })}

                {/* Pagination */}
                {totalPages > 1 && (
                  <div className="flex justify-center gap-2 mt-4">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setPage(p => Math.max(1, p - 1))}
                      disabled={page === 1}
                    >
                      Previous
                    </Button>
                    <span className="flex items-center px-4 text-sm text-muted-foreground">
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
              </div>
            )}

            {/* Empty state */}
            {!isLoading && isLoggedIn && historyItems.length === 0 && (
              <div className="glass-card p-12 text-center">
                <History className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="text-lg font-semibold mb-2">No Analysis History</h3>
                <p className="text-muted-foreground mb-6">
                  Your analyzed charts will appear here.
                </p>
                <Link to="/dashboard">
                  <Button variant="hero">
                    <Upload className="w-4 h-4" />
                    Upload Your First Chart
                  </Button>
                </Link>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

