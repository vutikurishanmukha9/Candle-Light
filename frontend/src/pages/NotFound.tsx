import { Link } from "react-router-dom";
import { Home, ArrowLeft, TrendingUp, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/ThemeToggle";

export default function NotFound() {
  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Header */}
      <header className="flex items-center justify-between p-4 sm:p-6">
        <Link to="/" className="flex items-center gap-2">
          <div className="w-10 h-10 rounded-xl gradient-primary flex items-center justify-center">
            <TrendingUp className="h-5 w-5 text-primary-foreground" />
          </div>
          <span className="font-bold text-xl">Candle-Light</span>
        </Link>
        <ThemeToggle />
      </header>

      {/* Content */}
      <main className="flex-1 flex items-center justify-center p-4">
        <div className="text-center max-w-md animate-fade-up">
          {/* 404 Illustration */}
          <div className="relative mb-8">
            <div className="text-[120px] sm:text-[160px] font-bold leading-none gradient-text opacity-20">
              404
            </div>
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-24 h-24 sm:w-32 sm:h-32 rounded-full bg-muted/50 flex items-center justify-center animate-bounce-subtle">
                <Search className="h-12 w-12 sm:h-16 sm:w-16 text-muted-foreground" />
              </div>
            </div>
          </div>

          {/* Text */}
          <h1 className="text-2xl sm:text-3xl font-bold mb-3">
            Page not found
          </h1>
          <p className="text-muted-foreground mb-8 text-lg">
            Oops! The page you're looking for doesn't exist or has been moved.
          </p>

          {/* Actions */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
            <Button
              variant="outline"
              onClick={() => window.history.back()}
              className="w-full sm:w-auto gap-2"
            >
              <ArrowLeft className="h-4 w-4" />
              Go Back
            </Button>
            <Link to="/dashboard" className="w-full sm:w-auto">
              <Button className="w-full gradient-primary gap-2">
                <Home className="h-4 w-4" />
                Go to Dashboard
              </Button>
            </Link>
          </div>

          {/* Helpful links */}
          <div className="mt-12 pt-8 border-t border-border">
            <p className="text-sm text-muted-foreground mb-4">
              Here are some helpful links:
            </p>
            <div className="flex flex-wrap justify-center gap-4 text-sm">
              <Link
                to="/dashboard"
                className="text-primary hover:text-primary/80 transition-colors"
              >
                Dashboard
              </Link>
              <Link
                to="/history"
                className="text-primary hover:text-primary/80 transition-colors"
              >
                History
              </Link>
              <Link
                to="/login"
                className="text-primary hover:text-primary/80 transition-colors"
              >
                Sign In
              </Link>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="p-4 text-center text-sm text-muted-foreground">
        © {new Date().getFullYear()} Candle-Light. All rights reserved.
      </footer>
    </div>
  );
}
