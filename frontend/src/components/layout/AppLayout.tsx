import { useState, useCallback } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import {
    TrendingUp,
    LayoutDashboard,
    Upload,
    History,
    Settings,
    LogOut,
    Menu,
    X,
    ChevronLeft,
    ChevronRight,
    User,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/ThemeToggle";
import { cn } from "@/lib/utils";
import { useAuth } from "@/contexts/AuthContext";
import {
    Tooltip,
    TooltipContent,
    TooltipTrigger,
} from "@/components/ui/tooltip";

const navItems = [
    { icon: LayoutDashboard, label: "Dashboard", href: "/dashboard" },
    { icon: Upload, label: "Upload", href: "/dashboard" },
    { icon: History, label: "History", href: "/history" },
    { icon: Settings, label: "Settings", href: "/settings" },
];

interface AppLayoutProps {
    children: React.ReactNode;
    title?: string;
    subtitle?: string;
}

export function AppLayout({ children, title, subtitle }: AppLayoutProps) {
    const [sidebarOpen, setSidebarOpen] = useState(false);
    const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
    const location = useLocation();
    const navigate = useNavigate();
    const { user, logout } = useAuth();

    const handleLogout = useCallback(async () => {
        await logout();
        navigate("/login");
    }, [logout, navigate]);

    const closeMobileSidebar = useCallback(() => {
        setSidebarOpen(false);
    }, []);

    return (
        <div className="min-h-screen bg-background">
            {/* Mobile Header */}
            <header className="lg:hidden fixed top-0 left-0 right-0 z-50 h-16 bg-card/95 backdrop-blur-xl border-b border-border flex items-center justify-between px-4">
                <div className="flex items-center gap-3">
                    <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => setSidebarOpen(true)}
                        className="touch-target"
                    >
                        <Menu className="h-5 w-5" />
                    </Button>
                    <Link to="/dashboard" className="flex items-center gap-2">
                        <div className="w-8 h-8 rounded-lg gradient-primary flex items-center justify-center">
                            <TrendingUp className="h-4 w-4 text-primary-foreground" />
                        </div>
                        <span className="font-semibold text-lg">Candle-Light</span>
                    </Link>
                </div>
                <ThemeToggle />
            </header>

            {/* Mobile Sidebar Overlay */}
            {sidebarOpen && (
                <div
                    className="lg:hidden fixed inset-0 z-50 bg-background/80 backdrop-blur-sm animate-fade-in"
                    onClick={closeMobileSidebar}
                />
            )}

            {/* Sidebar */}
            <aside
                className={cn(
                    // Base styles
                    "fixed top-0 left-0 z-50 h-full bg-sidebar-background border-r border-sidebar-border",
                    "flex flex-col transition-all duration-300 ease-in-out",
                    // Mobile: slide in from left
                    "lg:translate-x-0",
                    sidebarOpen ? "translate-x-0" : "-translate-x-full",
                    // Desktop: collapsible width
                    sidebarCollapsed ? "lg:w-20" : "lg:w-64",
                    "w-72"
                )}
            >
                {/* Sidebar Header */}
                <div className="h-16 flex items-center justify-between px-4 border-b border-sidebar-border">
                    <Link
                        to="/dashboard"
                        className={cn(
                            "flex items-center gap-3 transition-opacity",
                            sidebarCollapsed && "lg:opacity-0 lg:pointer-events-none"
                        )}
                    >
                        <div className="w-10 h-10 rounded-xl gradient-primary flex items-center justify-center glow-primary">
                            <TrendingUp className="h-5 w-5 text-primary-foreground" />
                        </div>
                        <span className="font-bold text-xl gradient-text">Candle-Light</span>
                    </Link>

                    {/* Mobile close button */}
                    <Button
                        variant="ghost"
                        size="icon"
                        onClick={closeMobileSidebar}
                        className="lg:hidden touch-target"
                    >
                        <X className="h-5 w-5" />
                    </Button>

                    {/* Desktop collapse button */}
                    <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
                        className="hidden lg:flex h-8 w-8"
                    >
                        {sidebarCollapsed ? (
                            <ChevronRight className="h-4 w-4" />
                        ) : (
                            <ChevronLeft className="h-4 w-4" />
                        )}
                    </Button>
                </div>

                {/* Navigation */}
                <nav className="flex-1 p-4 space-y-2 overflow-y-auto scrollbar-hide">
                    {navItems.map((item) => {
                        const isActive = location.pathname === item.href;
                        const Icon = item.icon;

                        const navLink = (
                            <Link
                                key={item.href + item.label}
                                to={item.href}
                                onClick={closeMobileSidebar}
                                className={cn(
                                    "relative flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200",
                                    "hover:bg-sidebar-accent/80 group touch-target",
                                    isActive && "bg-sidebar-accent text-sidebar-primary font-medium",
                                    sidebarCollapsed && "lg:justify-center lg:px-3"
                                )}
                            >
                                {isActive && (
                                    <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-6 bg-sidebar-primary rounded-r-full" />
                                )}
                                <Icon
                                    className={cn(
                                        "h-5 w-5 transition-colors shrink-0",
                                        isActive ? "text-sidebar-primary" : "text-sidebar-foreground/70 group-hover:text-sidebar-foreground"
                                    )}
                                />
                                <span
                                    className={cn(
                                        "transition-all duration-200",
                                        sidebarCollapsed && "lg:hidden"
                                    )}
                                >
                                    {item.label}
                                </span>
                            </Link>
                        );

                        // Wrap in tooltip when collapsed
                        if (sidebarCollapsed) {
                            return (
                                <Tooltip key={item.href + item.label}>
                                    <TooltipTrigger asChild className="hidden lg:flex">
                                        {navLink}
                                    </TooltipTrigger>
                                    <TooltipContent side="right" className="hidden lg:block">
                                        {item.label}
                                    </TooltipContent>
                                </Tooltip>
                            );
                        }

                        return navLink;
                    })}
                </nav>

                {/* Sidebar Footer */}
                <div className="p-4 border-t border-sidebar-border space-y-3">
                    {/* User Info */}
                    {user && (
                        <div
                            className={cn(
                                "flex items-center gap-3 px-3 py-2 rounded-lg bg-sidebar-accent/50",
                                sidebarCollapsed && "lg:justify-center"
                            )}
                        >
                            <div className="w-9 h-9 rounded-full bg-gradient-to-br from-primary to-primary/60 flex items-center justify-center shrink-0">
                                <User className="h-4 w-4 text-primary-foreground" />
                            </div>
                            <div className={cn("flex-1 min-w-0", sidebarCollapsed && "lg:hidden")}>
                                <p className="text-sm font-medium truncate">
                                    {user.full_name || user.email.split("@")[0]}
                                </p>
                                <p className="text-xs text-muted-foreground truncate">{user.email}</p>
                            </div>
                        </div>
                    )}

                    {/* Theme toggle and logout */}
                    <div className={cn(
                        "flex items-center gap-2",
                        sidebarCollapsed ? "lg:flex-col" : "justify-between"
                    )}>
                        <ThemeToggle />
                        <Button
                            variant="ghost"
                            size={sidebarCollapsed ? "icon" : "sm"}
                            onClick={handleLogout}
                            className={cn(
                                "text-muted-foreground hover:text-destructive hover:bg-destructive/10",
                                !sidebarCollapsed && "gap-2"
                            )}
                        >
                            <LogOut className="h-4 w-4" />
                            {!sidebarCollapsed && <span className="hidden lg:inline">Logout</span>}
                        </Button>
                    </div>
                </div>
            </aside>

            {/* Main Content */}
            <main
                className={cn(
                    "min-h-screen transition-all duration-300 pt-16 lg:pt-0",
                    sidebarCollapsed ? "lg:pl-20" : "lg:pl-64"
                )}
            >
                {/* Page Header */}
                {(title || subtitle) && (
                    <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 lg:top-0 z-40">
                        <div className="container-wide py-6">
                            {title && (
                                <h1 className="text-2xl sm:text-3xl font-bold text-balance animate-fade-up">
                                    {title}
                                </h1>
                            )}
                            {subtitle && (
                                <p className="text-muted-foreground mt-1 animate-fade-up animation-delay-100">
                                    {subtitle}
                                </p>
                            )}
                        </div>
                    </header>
                )}

                {/* Page Content */}
                <div className="container-wide py-6 lg:py-8">
                    {children}
                </div>
            </main>
        </div>
    );
}

export default AppLayout;
