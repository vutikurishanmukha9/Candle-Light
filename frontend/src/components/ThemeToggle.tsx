import { Moon, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useTheme } from "@/components/ThemeProvider";

export function ThemeToggle() {
  const { theme, toggleTheme } = useTheme();

  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={toggleTheme}
      className="w-9 h-9"
    >
      {theme === "dark" ? (
        <Sun className="w-5 h-5 text-muted-foreground hover:text-foreground transition-colors" />
      ) : (
        <Moon className="w-5 h-5 text-muted-foreground hover:text-foreground transition-colors" />
      )}
      <span className="sr-only">Toggle theme</span>
    </Button>
  );
}
