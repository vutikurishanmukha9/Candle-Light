import { Moon, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useTheme } from "@/components/ThemeProvider";
import { cn } from "@/lib/utils";

export function ThemeToggle() {
  const { theme, toggleTheme } = useTheme();
  const isDark = theme === "dark";

  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={toggleTheme}
      className={cn(
        "w-10 h-10 rounded-full relative overflow-hidden",
        "bg-muted/50 hover:bg-muted transition-colors"
      )}
      aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
    >
      {/* Sun Icon */}
      <Sun
        className={cn(
          "absolute h-5 w-5 transition-all duration-500",
          isDark
            ? "rotate-0 scale-100 text-yellow-400"
            : "-rotate-90 scale-0 text-yellow-400"
        )}
      />
      {/* Moon Icon */}
      <Moon
        className={cn(
          "absolute h-5 w-5 transition-all duration-500",
          isDark
            ? "rotate-90 scale-0 text-primary"
            : "rotate-0 scale-100 text-primary"
        )}
      />
      <span className="sr-only">Toggle theme</span>
    </Button>
  );
}
