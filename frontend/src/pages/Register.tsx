import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { Link, useNavigate } from "react-router-dom";
import { Eye, EyeOff, TrendingUp, Loader2, Mail, Lock, User, ArrowRight, Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ThemeToggle } from "@/components/ThemeToggle";
import { useAuth } from "@/contexts/AuthContext";
import { useToast } from "@/hooks/use-toast";
import { cn } from "@/lib/utils";

const registerSchema = z.object({
    fullName: z.string().min(2, "Name must be at least 2 characters"),
    email: z.string().email("Please enter a valid email address"),
    password: z.string().min(8, "Password must be at least 8 characters"),
    confirmPassword: z.string(),
}).refine((data) => data.password === data.confirmPassword, {
    message: "Passwords don't match",
    path: ["confirmPassword"],
});

type RegisterFormData = z.infer<typeof registerSchema>;

// Password strength checker
function getPasswordStrength(password: string): { score: number; label: string; color: string } {
    let score = 0;
    if (password.length >= 8) score++;
    if (password.length >= 12) score++;
    if (/[a-z]/.test(password) && /[A-Z]/.test(password)) score++;
    if (/\d/.test(password)) score++;
    if (/[^a-zA-Z0-9]/.test(password)) score++;

    if (score <= 1) return { score, label: "Weak", color: "bg-destructive" };
    if (score <= 2) return { score, label: "Fair", color: "bg-warning" };
    if (score <= 3) return { score, label: "Good", color: "bg-primary" };
    return { score, label: "Strong", color: "bg-success" };
}

export default function Register() {
    const [showPassword, setShowPassword] = useState(false);
    const [showConfirmPassword, setShowConfirmPassword] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const navigate = useNavigate();
    const { register: registerUser } = useAuth();
    const { toast } = useToast();

    const {
        register,
        handleSubmit,
        watch,
        formState: { errors },
    } = useForm<RegisterFormData>({
        resolver: zodResolver(registerSchema),
    });

    const password = watch("password", "");
    const passwordStrength = getPasswordStrength(password);

    const onSubmit = async (data: RegisterFormData) => {
        setIsLoading(true);
        try {
            await registerUser(data.email, data.password, data.fullName);
            toast({
                title: "Account created!",
                description: "Welcome to Candle-Light. Your account is ready.",
            });
            navigate("/dashboard", { replace: true });
        } catch (error: unknown) {
            const message = error instanceof Error ? error.message : "Registration failed";
            toast({
                title: "Registration failed",
                description: message,
                variant: "destructive",
            });
        } finally {
            setIsLoading(false);
        }
    };

    const handleGoogleSignup = () => {
        const apiUrl = import.meta.env.VITE_API_URL || "http://localhost:8000";
        window.location.href = `${apiUrl}/auth/google`;
    };

    return (
        <div className="min-h-screen flex">
            {/* Left Panel - Branding (Hidden on mobile) */}
            <div className="hidden lg:flex lg:w-1/2 relative overflow-hidden bg-gradient-to-br from-primary/10 via-background to-primary/5">
                {/* Animated background elements */}
                <div className="absolute inset-0 pattern-dots opacity-30" />
                <div className="absolute top-1/3 right-1/4 w-80 h-80 bg-primary/20 rounded-full blur-3xl animate-pulse-slow" />
                <div className="absolute bottom-1/3 left-1/4 w-56 h-56 bg-primary/10 rounded-full blur-2xl animate-float" />

                {/* Content */}
                <div className="relative z-10 flex flex-col justify-center px-12 xl:px-20">
                    <div className="mb-8 animate-fade-up">
                        <div className="w-16 h-16 rounded-2xl gradient-primary flex items-center justify-center glow-primary-intense mb-6">
                            <TrendingUp className="h-8 w-8 text-primary-foreground" />
                        </div>
                        <h1 className="text-4xl xl:text-5xl font-bold mb-4 text-balance">
                            Start Your <span className="gradient-text">Trading Journey</span>
                        </h1>
                        <p className="text-xl text-muted-foreground max-w-md">
                            Join thousands of traders using AI-powered analysis for better decisions.
                        </p>
                    </div>

                    {/* Benefits */}
                    <div className="space-y-4 animate-fade-up animation-delay-200">
                        {[
                            "Free chart analysis",
                            "Unlimited pattern detection",
                            "Professional insights",
                            "Export your data",
                        ].map((benefit, i) => (
                            <div key={i} className="flex items-center gap-3 text-muted-foreground">
                                <div className="w-5 h-5 rounded-full bg-primary/20 flex items-center justify-center">
                                    <Check className="h-3 w-3 text-primary" />
                                </div>
                                <span>{benefit}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Right Panel - Register Form */}
            <div className="w-full lg:w-1/2 flex flex-col">
                {/* Header */}
                <header className="flex items-center justify-between p-4 sm:p-6">
                    <Link to="/" className="flex items-center gap-2 lg:hidden">
                        <div className="w-10 h-10 rounded-xl gradient-primary flex items-center justify-center">
                            <TrendingUp className="h-5 w-5 text-primary-foreground" />
                        </div>
                        <span className="font-bold text-xl">Candle-Light</span>
                    </Link>
                    <div className="lg:hidden" />
                    <ThemeToggle />
                </header>

                {/* Form Container */}
                <div className="flex-1 flex items-center justify-center px-4 sm:px-6 lg:px-12 xl:px-20 pb-12">
                    <div className="w-full max-w-md space-y-6 animate-fade-up">
                        {/* Welcome Text */}
                        <div className="text-center lg:text-left">
                            <h2 className="text-2xl sm:text-3xl font-bold">Create your account</h2>
                            <p className="text-muted-foreground mt-2">
                                Get started with free AI-powered analysis
                            </p>
                        </div>

                        {/* Google OAuth Button */}
                        <Button
                            type="button"
                            variant="outline"
                            className="w-full h-12 text-base font-medium gap-3 hover:bg-muted/50 transition-all"
                            onClick={handleGoogleSignup}
                        >
                            <svg className="h-5 w-5" viewBox="0 0 24 24">
                                <path
                                    d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                                    fill="#4285F4"
                                />
                                <path
                                    d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                                    fill="#34A853"
                                />
                                <path
                                    d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                                    fill="#FBBC05"
                                />
                                <path
                                    d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                                    fill="#EA4335"
                                />
                            </svg>
                            Sign up with Google
                        </Button>

                        {/* Divider */}
                        <div className="relative">
                            <div className="absolute inset-0 flex items-center">
                                <div className="w-full border-t border-border" />
                            </div>
                            <div className="relative flex justify-center text-xs uppercase">
                                <span className="bg-background px-4 text-muted-foreground">
                                    or register with email
                                </span>
                            </div>
                        </div>

                        {/* Register Form */}
                        <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
                            {/* Full Name */}
                            <div className="space-y-2">
                                <Label htmlFor="fullName" className="text-sm font-medium">
                                    Full name
                                </Label>
                                <div className="relative">
                                    <User className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
                                    <Input
                                        id="fullName"
                                        type="text"
                                        placeholder="John Doe"
                                        className={cn(
                                            "h-12 pl-10 text-base input-glow",
                                            errors.fullName && "border-destructive focus-visible:ring-destructive"
                                        )}
                                        {...register("fullName")}
                                    />
                                </div>
                                {errors.fullName && (
                                    <p className="text-sm text-destructive animate-fade-in">
                                        {errors.fullName.message}
                                    </p>
                                )}
                            </div>

                            {/* Email */}
                            <div className="space-y-2">
                                <Label htmlFor="email" className="text-sm font-medium">
                                    Email address
                                </Label>
                                <div className="relative">
                                    <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
                                    <Input
                                        id="email"
                                        type="email"
                                        placeholder="you@example.com"
                                        className={cn(
                                            "h-12 pl-10 text-base input-glow",
                                            errors.email && "border-destructive focus-visible:ring-destructive"
                                        )}
                                        {...register("email")}
                                    />
                                </div>
                                {errors.email && (
                                    <p className="text-sm text-destructive animate-fade-in">
                                        {errors.email.message}
                                    </p>
                                )}
                            </div>

                            {/* Password */}
                            <div className="space-y-2">
                                <Label htmlFor="password" className="text-sm font-medium">
                                    Password
                                </Label>
                                <div className="relative">
                                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
                                    <Input
                                        id="password"
                                        type={showPassword ? "text" : "password"}
                                        placeholder="••••••••"
                                        className={cn(
                                            "h-12 pl-10 pr-10 text-base input-glow",
                                            errors.password && "border-destructive focus-visible:ring-destructive"
                                        )}
                                        {...register("password")}
                                    />
                                    <button
                                        type="button"
                                        onClick={() => setShowPassword(!showPassword)}
                                        className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                                    >
                                        {showPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                                    </button>
                                </div>
                                {/* Password Strength Indicator */}
                                {password && (
                                    <div className="space-y-2 animate-fade-in">
                                        <div className="flex gap-1">
                                            {[1, 2, 3, 4].map((level) => (
                                                <div
                                                    key={level}
                                                    className={cn(
                                                        "h-1.5 flex-1 rounded-full transition-colors",
                                                        level <= passwordStrength.score
                                                            ? passwordStrength.color
                                                            : "bg-muted"
                                                    )}
                                                />
                                            ))}
                                        </div>
                                        <p className="text-xs text-muted-foreground">
                                            Password strength: <span className="font-medium">{passwordStrength.label}</span>
                                        </p>
                                    </div>
                                )}
                                {errors.password && (
                                    <p className="text-sm text-destructive animate-fade-in">
                                        {errors.password.message}
                                    </p>
                                )}
                            </div>

                            {/* Confirm Password */}
                            <div className="space-y-2">
                                <Label htmlFor="confirmPassword" className="text-sm font-medium">
                                    Confirm password
                                </Label>
                                <div className="relative">
                                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
                                    <Input
                                        id="confirmPassword"
                                        type={showConfirmPassword ? "text" : "password"}
                                        placeholder="••••••••"
                                        className={cn(
                                            "h-12 pl-10 pr-10 text-base input-glow",
                                            errors.confirmPassword && "border-destructive focus-visible:ring-destructive"
                                        )}
                                        {...register("confirmPassword")}
                                    />
                                    <button
                                        type="button"
                                        onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                                        className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                                    >
                                        {showConfirmPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                                    </button>
                                </div>
                                {errors.confirmPassword && (
                                    <p className="text-sm text-destructive animate-fade-in">
                                        {errors.confirmPassword.message}
                                    </p>
                                )}
                            </div>

                            <Button
                                type="submit"
                                className="w-full h-12 text-base font-medium gradient-primary hover:opacity-90 transition-all gap-2 group"
                                disabled={isLoading}
                            >
                                {isLoading ? (
                                    <>
                                        <Loader2 className="h-5 w-5 animate-spin" />
                                        Creating account...
                                    </>
                                ) : (
                                    <>
                                        Create account
                                        <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
                                    </>
                                )}
                            </Button>
                        </form>

                        {/* Terms */}
                        <p className="text-center text-xs text-muted-foreground">
                            By creating an account, you agree to our{" "}
                            <Link to="/terms" className="text-primary hover:underline">
                                Terms of Service
                            </Link>{" "}
                            and{" "}
                            <Link to="/privacy" className="text-primary hover:underline">
                                Privacy Policy
                            </Link>
                        </p>

                        {/* Sign In Link */}
                        <p className="text-center text-muted-foreground">
                            Already have an account?{" "}
                            <Link
                                to="/login"
                                className="text-primary hover:text-primary/80 font-medium transition-colors"
                            >
                                Sign in
                            </Link>
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}
