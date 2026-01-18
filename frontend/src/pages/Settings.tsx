import { useState } from "react";
import {
    User,
    Palette,
    Bell,
    Shield,
    Sparkles,
    Moon,
    Sun,
    Monitor,
    Download,
    Trash2,
    Save,
    Eye,
    EyeOff,
    AlertTriangle,
    Check,
    ChevronRight
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { AppLayout } from "@/components/layout/AppLayout";
import { useAuth } from "@/contexts/AuthContext";
import { useToast } from "@/hooks/use-toast";
import { cn } from "@/lib/utils";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import {
    AlertDialog,
    AlertDialogAction,
    AlertDialogCancel,
    AlertDialogContent,
    AlertDialogDescription,
    AlertDialogFooter,
    AlertDialogHeader,
    AlertDialogTitle,
    AlertDialogTrigger,
} from "@/components/ui/alert-dialog";

// Settings section component
function SettingsSection({
    icon: Icon,
    title,
    description,
    children,
    className
}: {
    icon: React.ElementType;
    title: string;
    description: string;
    children: React.ReactNode;
    className?: string;
}) {
    return (
        <div className={cn("glass-card p-6 space-y-6", className)}>
            <div className="flex items-start gap-4">
                <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center shrink-0">
                    <Icon className="h-5 w-5 text-primary" />
                </div>
                <div>
                    <h2 className="text-lg font-semibold">{title}</h2>
                    <p className="text-sm text-muted-foreground">{description}</p>
                </div>
            </div>
            <div className="space-y-4 pl-14">
                {children}
            </div>
        </div>
    );
}

// Settings row component
function SettingsRow({
    label,
    description,
    children
}: {
    label: string;
    description?: string;
    children: React.ReactNode;
}) {
    return (
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 py-3 border-b border-border/50 last:border-0">
            <div className="space-y-0.5">
                <Label className="text-sm font-medium">{label}</Label>
                {description && (
                    <p className="text-xs text-muted-foreground">{description}</p>
                )}
            </div>
            <div className="shrink-0">
                {children}
            </div>
        </div>
    );
}

export default function Settings() {
    const { user } = useAuth();
    const { toast } = useToast();

    // Profile state
    const [displayName, setDisplayName] = useState(user?.full_name || "");
    const [isSavingProfile, setIsSavingProfile] = useState(false);

    // Password state
    const [showPasswordChange, setShowPasswordChange] = useState(false);
    const [currentPassword, setCurrentPassword] = useState("");
    const [newPassword, setNewPassword] = useState("");
    const [confirmPassword, setConfirmPassword] = useState("");
    const [showPasswords, setShowPasswords] = useState(false);

    // Preferences state
    const [theme, setTheme] = useState<"dark" | "light" | "system">("dark");
    const [autoSave, setAutoSave] = useState(true);
    const [emailNotifications, setEmailNotifications] = useState(true);
    const [analysisNotifications, setAnalysisNotifications] = useState(true);
    const [compactView, setCompactView] = useState(false);
    const [confidenceThreshold, setConfidenceThreshold] = useState("60");

    // Handle profile save
    const handleSaveProfile = async () => {
        setIsSavingProfile(true);
        try {
            // TODO: API call to update profile
            await new Promise(resolve => setTimeout(resolve, 1000));
            toast({
                title: "Profile Updated",
                description: "Your profile has been saved successfully.",
            });
        } catch (error) {
            toast({
                title: "Error",
                description: "Failed to update profile. Please try again.",
                variant: "destructive",
            });
        } finally {
            setIsSavingProfile(false);
        }
    };

    // Handle password change
    const handleChangePassword = async () => {
        if (newPassword !== confirmPassword) {
            toast({
                title: "Password Mismatch",
                description: "New password and confirmation don't match.",
                variant: "destructive",
            });
            return;
        }
        if (newPassword.length < 8) {
            toast({
                title: "Password Too Short",
                description: "Password must be at least 8 characters.",
                variant: "destructive",
            });
            return;
        }

        try {
            // TODO: API call to change password
            await new Promise(resolve => setTimeout(resolve, 1000));
            toast({
                title: "Password Changed",
                description: "Your password has been updated successfully.",
            });
            setShowPasswordChange(false);
            setCurrentPassword("");
            setNewPassword("");
            setConfirmPassword("");
        } catch (error) {
            toast({
                title: "Error",
                description: "Failed to change password. Please try again.",
                variant: "destructive",
            });
        }
    };

    // Handle data export
    const handleExportData = async () => {
        try {
            // TODO: API call to export data
            toast({
                title: "Export Started",
                description: "Your data export is being prepared. You'll receive a download link shortly.",
            });
        } catch (error) {
            toast({
                title: "Error",
                description: "Failed to export data. Please try again.",
                variant: "destructive",
            });
        }
    };

    // Handle clear history
    const handleClearHistory = async () => {
        try {
            // TODO: API call to clear history
            toast({
                title: "History Cleared",
                description: "All your analysis history has been deleted.",
            });
        } catch (error) {
            toast({
                title: "Error",
                description: "Failed to clear history. Please try again.",
                variant: "destructive",
            });
        }
    };

    // Handle theme change
    const handleThemeChange = (newTheme: "dark" | "light" | "system") => {
        setTheme(newTheme);
        // Apply theme
        const root = window.document.documentElement;
        root.classList.remove("light", "dark");
        if (newTheme === "system") {
            const systemTheme = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
            root.classList.add(systemTheme);
        } else {
            root.classList.add(newTheme);
        }
        localStorage.setItem("theme", newTheme);
    };

    return (
        <AppLayout
            title="Settings"
            subtitle="Manage your account preferences and application settings"
        >
            <div className="max-w-3xl mx-auto space-y-6">
                {/* Profile Section */}
                <SettingsSection
                    icon={User}
                    title="Profile"
                    description="Manage your personal information"
                >
                    <SettingsRow label="Email" description="Your account email address">
                        <span className="text-sm text-muted-foreground">{user?.email || "user@example.com"}</span>
                    </SettingsRow>

                    <SettingsRow label="Display Name" description="How you appear in the app">
                        <div className="flex gap-2 w-full sm:w-auto">
                            <Input
                                value={displayName}
                                onChange={(e) => setDisplayName(e.target.value)}
                                placeholder="Enter your name"
                                className="w-full sm:w-48"
                            />
                            <Button
                                size="sm"
                                onClick={handleSaveProfile}
                                disabled={isSavingProfile}
                            >
                                {isSavingProfile ? (
                                    <div className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                                ) : (
                                    <Save className="h-4 w-4" />
                                )}
                            </Button>
                        </div>
                    </SettingsRow>

                    <SettingsRow label="Password" description="Change your account password">
                        {!showPasswordChange ? (
                            <Button variant="outline" size="sm" onClick={() => setShowPasswordChange(true)}>
                                Change Password
                            </Button>
                        ) : (
                            <div className="space-y-3 w-full sm:w-64">
                                <div className="relative">
                                    <Input
                                        type={showPasswords ? "text" : "password"}
                                        value={currentPassword}
                                        onChange={(e) => setCurrentPassword(e.target.value)}
                                        placeholder="Current password"
                                    />
                                </div>
                                <Input
                                    type={showPasswords ? "text" : "password"}
                                    value={newPassword}
                                    onChange={(e) => setNewPassword(e.target.value)}
                                    placeholder="New password"
                                />
                                <Input
                                    type={showPasswords ? "text" : "password"}
                                    value={confirmPassword}
                                    onChange={(e) => setConfirmPassword(e.target.value)}
                                    placeholder="Confirm new password"
                                />
                                <div className="flex items-center gap-2">
                                    <Button size="sm" onClick={() => setShowPasswords(!showPasswords)} variant="ghost">
                                        {showPasswords ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                                    </Button>
                                    <Button size="sm" onClick={handleChangePassword}>
                                        Update Password
                                    </Button>
                                    <Button size="sm" variant="ghost" onClick={() => setShowPasswordChange(false)}>
                                        Cancel
                                    </Button>
                                </div>
                            </div>
                        )}
                    </SettingsRow>

                    <SettingsRow label="Member Since">
                        <span className="text-sm text-muted-foreground">
                            {user?.created_at ? new Date(user.created_at).toLocaleDateString() : "January 2026"}
                        </span>
                    </SettingsRow>
                </SettingsSection>

                {/* AI Preferences Section */}
                <SettingsSection
                    icon={Sparkles}
                    title="AI Preferences"
                    description="Configure how the AI analyzes your charts"
                >
                    <SettingsRow label="Confidence Threshold" description="Minimum confidence level for pattern alerts">
                        <Select value={confidenceThreshold} onValueChange={setConfidenceThreshold}>
                            <SelectTrigger className="w-32">
                                <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                                <SelectItem value="40">40%</SelectItem>
                                <SelectItem value="50">50%</SelectItem>
                                <SelectItem value="60">60%</SelectItem>
                                <SelectItem value="70">70%</SelectItem>
                                <SelectItem value="80">80%</SelectItem>
                            </SelectContent>
                        </Select>
                    </SettingsRow>

                    <SettingsRow label="Auto-Save Analyses" description="Automatically save all completed analyses">
                        <Switch checked={autoSave} onCheckedChange={setAutoSave} />
                    </SettingsRow>
                </SettingsSection>

                {/* Appearance Section */}
                <SettingsSection
                    icon={Palette}
                    title="Appearance"
                    description="Customize how the app looks"
                >
                    <SettingsRow label="Theme" description="Choose your preferred color scheme">
                        <div className="flex gap-2">
                            {[
                                { value: "light", icon: Sun, label: "Light" },
                                { value: "dark", icon: Moon, label: "Dark" },
                                { value: "system", icon: Monitor, label: "System" },
                            ].map(({ value, icon: Icon, label }) => (
                                <Button
                                    key={value}
                                    variant={theme === value ? "default" : "outline"}
                                    size="sm"
                                    onClick={() => handleThemeChange(value as "dark" | "light" | "system")}
                                    className="gap-2"
                                >
                                    <Icon className="h-4 w-4" />
                                    <span className="hidden sm:inline">{label}</span>
                                </Button>
                            ))}
                        </div>
                    </SettingsRow>

                    <SettingsRow label="Compact View" description="Use smaller spacing and elements">
                        <Switch checked={compactView} onCheckedChange={setCompactView} />
                    </SettingsRow>
                </SettingsSection>

                {/* Notifications Section */}
                <SettingsSection
                    icon={Bell}
                    title="Notifications"
                    description="Control how you receive updates"
                >
                    <SettingsRow label="Email Notifications" description="Receive important updates via email">
                        <Switch checked={emailNotifications} onCheckedChange={setEmailNotifications} />
                    </SettingsRow>

                    <SettingsRow label="Analysis Complete" description="Get notified when analysis finishes">
                        <Switch checked={analysisNotifications} onCheckedChange={setAnalysisNotifications} />
                    </SettingsRow>
                </SettingsSection>

                {/* Data & Privacy Section */}
                <SettingsSection
                    icon={Shield}
                    title="Data & Privacy"
                    description="Manage your data and account"
                    className="border-destructive/20"
                >
                    <SettingsRow label="Export Data" description="Download all your data as JSON">
                        <Button variant="outline" size="sm" onClick={handleExportData} className="gap-2">
                            <Download className="h-4 w-4" />
                            Export
                        </Button>
                    </SettingsRow>

                    <SettingsRow label="Clear History" description="Delete all your analysis history">
                        <AlertDialog>
                            <AlertDialogTrigger asChild>
                                <Button variant="outline" size="sm" className="gap-2 text-warning hover:text-warning">
                                    <Trash2 className="h-4 w-4" />
                                    Clear
                                </Button>
                            </AlertDialogTrigger>
                            <AlertDialogContent>
                                <AlertDialogHeader>
                                    <AlertDialogTitle>Clear Analysis History?</AlertDialogTitle>
                                    <AlertDialogDescription>
                                        This will permanently delete all your saved analyses. This action cannot be undone.
                                    </AlertDialogDescription>
                                </AlertDialogHeader>
                                <AlertDialogFooter>
                                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                                    <AlertDialogAction onClick={handleClearHistory} className="bg-destructive text-destructive-foreground hover:bg-destructive/90">
                                        Delete All
                                    </AlertDialogAction>
                                </AlertDialogFooter>
                            </AlertDialogContent>
                        </AlertDialog>
                    </SettingsRow>

                    <SettingsRow label="Delete Account" description="Permanently delete your account and all data">
                        <AlertDialog>
                            <AlertDialogTrigger asChild>
                                <Button variant="destructive" size="sm" className="gap-2">
                                    <AlertTriangle className="h-4 w-4" />
                                    Delete Account
                                </Button>
                            </AlertDialogTrigger>
                            <AlertDialogContent>
                                <AlertDialogHeader>
                                    <AlertDialogTitle className="flex items-center gap-2 text-destructive">
                                        <AlertTriangle className="h-5 w-5" />
                                        Delete Account?
                                    </AlertDialogTitle>
                                    <AlertDialogDescription>
                                        This will permanently delete your account, all analyses, and personal data.
                                        This action is irreversible.
                                    </AlertDialogDescription>
                                </AlertDialogHeader>
                                <AlertDialogFooter>
                                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                                    <AlertDialogAction className="bg-destructive text-destructive-foreground hover:bg-destructive/90">
                                        Delete My Account
                                    </AlertDialogAction>
                                </AlertDialogFooter>
                            </AlertDialogContent>
                        </AlertDialog>
                    </SettingsRow>
                </SettingsSection>
            </div>
        </AppLayout>
    );
}
