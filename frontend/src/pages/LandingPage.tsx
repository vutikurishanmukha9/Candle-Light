import { Link } from "react-router-dom";
import {
  TrendingUp,
  Brain,
  Shield,
  Zap,
  Upload,
  LineChart,
  CheckCircle,
  ArrowRight,
  BarChart3,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { DisclaimerBanner } from "@/components/DisclaimerBanner";
import { ThemeToggle } from "@/components/ThemeToggle";
import heroBg from "@/assets/hero-bg.jpg";

const features = [
  {
    icon: Brain,
    title: "AI Pattern Detection",
    description:
      "Advanced machine learning models identify candlestick patterns with high accuracy.",
  },
  {
    icon: BarChart3,
    title: "Probabilistic Insights",
    description:
      "Get confidence scores and probability assessments for each detected pattern.",
  },
  {
    icon: LineChart,
    title: "Image-Based Analysis",
    description:
      "Simply upload a screenshot of any chart. No data import required.",
  },
  {
    icon: Zap,
    title: "Fast & Secure",
    description:
      "Get results in seconds. Your data is encrypted and never stored permanently.",
  },
];

const steps = [
  {
    step: "01",
    title: "Upload Chart",
    description: "Drag and drop or click to upload any candlestick chart image.",
  },
  {
    step: "02",
    title: "AI Analysis",
    description: "Our AI scans for patterns, trends, and key technical indicators.",
  },
  {
    step: "03",
    title: "Get Insights",
    description: "Receive detailed analysis with confidence scores and recommendations.",
  },
];

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-background">
      {/* Hero Section */}
      <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
        {/* Background */}
        <div className="absolute inset-0">
          <img
            src={heroBg}
            alt=""
            className="w-full h-full object-cover opacity-30"
          />
          <div className="absolute inset-0 bg-gradient-to-b from-background/50 via-background/80 to-background" />
          <div className="absolute inset-0 pattern-grid opacity-30" />
        </div>

        {/* Header */}
        <header className="absolute top-0 left-0 right-0 z-20 p-4">
          <div className="container mx-auto flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg gradient-primary flex items-center justify-center">
                <TrendingUp className="w-4 h-4 text-primary-foreground" />
              </div>
              <span className="font-semibold">Candle-Light</span>
            </div>
            <ThemeToggle />
          </div>
        </header>

        {/* Content */}
        <div className="relative z-10 container mx-auto px-4 py-20 pt-28 text-center">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20 mb-8">
            <TrendingUp className="w-4 h-4 text-primary" />
            <span className="text-sm text-primary font-medium">
              AI-Powered Trading Analysis
            </span>
          </div>

          <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold mb-6 leading-tight">
            <span className="gradient-text">Candlestick Pattern</span>
            <br />
            Analysis with AI
          </h1>

          <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto mb-10">
            Upload a chart. Get instant technical insights. Our AI identifies patterns,
            predicts market bias, and provides actionable analysis in seconds.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link to="/dashboard">
              <Button variant="hero" size="xl">
                <Upload className="w-5 h-5" />
                Upload Chart
              </Button>
            </Link>
            <Link to="/dashboard?demo=true">
              <Button variant="outline" size="xl">
                Try Demo
                <ArrowRight className="w-5 h-5" />
              </Button>
            </Link>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-3 gap-8 max-w-xl mx-auto mt-16">
            {[
              { value: "50K+", label: "Charts Analyzed" },
              { value: "95%", label: "Accuracy Rate" },
              { value: "<3s", label: "Analysis Time" },
            ].map((stat) => (
              <div key={stat.label}>
                <p className="text-2xl md:text-3xl font-bold text-primary">
                  {stat.value}
                </p>
                <p className="text-sm text-muted-foreground">{stat.label}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Scroll indicator */}
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 animate-bounce">
          <div className="w-6 h-10 rounded-full border-2 border-muted-foreground/30 flex items-start justify-center p-2">
            <div className="w-1 h-2 bg-primary rounded-full" />
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-24 relative">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              Powerful Features for{" "}
              <span className="gradient-text">Smart Trading</span>
            </h2>
            <p className="text-muted-foreground max-w-xl mx-auto">
              Leverage cutting-edge AI technology to gain insights from your chart images.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((feature) => (
              <div
                key={feature.title}
                className="glass-card p-6 group hover:border-primary/50 transition-all duration-300"
              >
                <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center mb-4 group-hover:bg-primary/20 transition-colors">
                  <feature.icon className="w-6 h-6 text-primary" />
                </div>
                <h3 className="text-lg font-semibold mb-2">{feature.title}</h3>
                <p className="text-sm text-muted-foreground">
                  {feature.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How it Works */}
      <section className="py-24 bg-secondary/30">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              How It <span className="gradient-text">Works</span>
            </h2>
            <p className="text-muted-foreground max-w-xl mx-auto">
              Three simple steps to unlock AI-powered chart analysis.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8 max-w-4xl mx-auto">
            {steps.map((item, idx) => (
              <div key={item.step} className="relative">
                <div className="glass-card p-6 text-center h-full">
                  <div className="text-5xl font-bold text-primary/20 mb-4">
                    {item.step}
                  </div>
                  <h3 className="text-xl font-semibold mb-2">{item.title}</h3>
                  <p className="text-sm text-muted-foreground">
                    {item.description}
                  </p>
                </div>
                {idx < steps.length - 1 && (
                  <div className="hidden md:block absolute top-1/2 -right-4 transform -translate-y-1/2">
                    <ArrowRight className="w-8 h-8 text-primary/30" />
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24">
        <div className="container mx-auto px-4">
          <div className="glass-card p-8 md:p-12 text-center max-w-3xl mx-auto">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              Ready to Analyze Your Charts?
            </h2>
            <p className="text-muted-foreground mb-8 max-w-xl mx-auto">
              Join thousands of traders using AI to gain an edge in the market.
              Start your analysis today.
            </p>
            <Link to="/dashboard">
              <Button variant="hero" size="xl">
                <Upload className="w-5 h-5" />
                Get Started Free
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* Disclaimer */}
      <section className="py-8 border-t border-border">
        <div className="container mx-auto px-4">
          <DisclaimerBanner />
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 border-t border-border">
        <div className="container mx-auto px-4">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg gradient-primary flex items-center justify-center">
                <TrendingUp className="w-4 h-4 text-primary-foreground" />
              </div>
              <span className="font-semibold">Candle-Light</span>
            </div>
            <div className="flex items-center gap-6 text-sm text-muted-foreground">
              <a href="#" className="hover:text-foreground transition-colors">
                About
              </a>
              <a href="#" className="hover:text-foreground transition-colors">
                Privacy
              </a>
              <a href="#" className="hover:text-foreground transition-colors">
                Terms
              </a>
              <a href="#" className="hover:text-foreground transition-colors">
                Contact
              </a>
            </div>
            <p className="text-sm text-muted-foreground">
              © 2026 Candle-Light. All rights reserved.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
