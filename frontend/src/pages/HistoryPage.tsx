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
  Calendar,
  TrendingDown,
  Minus,
  Eye,
} from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/ThemeToggle";
import { cn } from "@/lib/utils";

const navItems = [
  { icon: LayoutDashboard, label: "Dashboard", href: "/dashboard" },
  { icon: Upload, label: "Upload", href: "/dashboard" },
  { icon: History, label: "History", href: "/history", active: true },
  { icon: Settings, label: "Settings", href: "/settings" },
];

// Mock history data
const historyItems = [
  {
    id: "1",
    symbol: "BTC/USDT",
    bias: "bullish" as const,
    confidence: 82,
    patterns: ["Double Bottom", "Hammer"],
    timestamp: new Date(Date.now() - 1000 * 60 * 30),
    thumbnail: "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?w=200&h=120&fit=crop",
  },
  {
    id: "2",
    symbol: "ETH/USDT",
    bias: "bearish" as const,
    confidence: 65,
    patterns: ["Head & Shoulders", "Bearish Engulfing"],
    timestamp: new Date(Date.now() - 1000 * 60 * 60 * 2),
    thumbnail: "https://images.unsplash.com/photo-1642790551116-18e150f248e3?w=200&h=120&fit=crop",
  },
  {
    id: "3",
    symbol: "AAPL",
    bias: "neutral" as const,
    confidence: 45,
    patterns: ["Doji", "Spinning Top"],
    timestamp: new Date(Date.now() - 1000 * 60 * 60 * 24),
    thumbnail: "https://images.unsplash.com/photo-1590283603385-17ffb3a7f29f?w=200&h=120&fit=crop",
  },
  {
    id: "4",
    symbol: "TSLA",
    bias: "bullish" as const,
    confidence: 71,
    patterns: ["Morning Star", "Bullish Harami"],
    timestamp: new Date(Date.now() - 1000 * 60 * 60 * 48),
    thumbnail: "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?w=200&h=120&fit=crop",
  },
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
            <button className="flex items-center gap-3 px-4 py-3 w-full rounded-lg text-sidebar-foreground hover:bg-sidebar-accent/50 transition-colors">
              <LogOut className="w-5 h-5" />
              <span className="font-medium">Sign Out</span>
            </button>
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
              <ThemeToggle />
              <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center">
                <span className="text-sm font-medium text-primary">U</span>
              </div>
            </div>
          </div>
        </header>

        <div className="p-4 lg:p-8">
          <div className="max-w-5xl mx-auto">
            <div className="grid gap-4">
              {historyItems.map((item) => {
                const Icon = BiasIcon[item.bias];
                return (
                  <div
                    key={item.id}
                    className="glass-card p-4 flex flex-col sm:flex-row items-start sm:items-center gap-4 hover:border-primary/30 transition-colors cursor-pointer"
                  >
                    <div className="w-full sm:w-32 h-20 rounded-lg overflow-hidden bg-secondary shrink-0">
                      <img
                        src={item.thumbnail}
                        alt={item.symbol}
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-3 mb-2">
                        <h3 className="font-semibold text-lg">{item.symbol}</h3>
                        <span
                          className={cn(
                            "inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium",
                            biasColors[item.bias]
                          )}
                        >
                          <Icon className="w-3 h-3" />
                          {item.bias.charAt(0).toUpperCase() + item.bias.slice(1)}
                        </span>
                      </div>
                      <div className="flex flex-wrap gap-1.5 mb-2">
                        {item.patterns.map((pattern) => (
                          <span
                            key={pattern}
                            className="px-2 py-0.5 bg-secondary rounded text-xs text-muted-foreground"
                          >
                            {pattern}
                          </span>
                        ))}
                      </div>
                      <div className="flex items-center gap-4 text-sm text-muted-foreground">
                        <span className="font-mono">{item.confidence}% confidence</span>
                        <span className="flex items-center gap-1">
                          <Calendar className="w-3.5 h-3.5" />
                          {formatTimeAgo(item.timestamp)}
                        </span>
                      </div>
                    </div>
                    <Button variant="ghost" size="sm">
                      <Eye className="w-4 h-4" />
                      View
                    </Button>
                  </div>
                );
              })}
            </div>

            {historyItems.length === 0 && (
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
