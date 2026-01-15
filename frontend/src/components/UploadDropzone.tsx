import { useCallback, useState } from "react";
import { Upload, Image, X, AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

interface UploadDropzoneProps {
  onFileSelect: (file: File) => void;
  onAnalyze: () => void;
  isAnalyzing?: boolean;
  className?: string;
}

export function UploadDropzone({
  onFileSelect,
  onAnalyze,
  isAnalyzing = false,
  className,
}: UploadDropzoneProps) {
  const [preview, setPreview] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string>("");
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string>("");

  const maxSize = 10 * 1024 * 1024; // 10MB
  const acceptedTypes = ["image/png", "image/jpeg", "image/webp"];

  const handleFile = useCallback(
    (file: File) => {
      setError("");

      if (!acceptedTypes.includes(file.type)) {
        setError("Please upload a PNG, JPEG, or WebP image.");
        return;
      }

      if (file.size > maxSize) {
        setError("File size must be less than 10MB.");
        return;
      }

      const reader = new FileReader();
      reader.onload = (e) => {
        setPreview(e.target?.result as string);
        setFileName(file.name);
        onFileSelect(file);
      };
      reader.readAsDataURL(file);
    },
    [onFileSelect]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const clearPreview = () => {
    setPreview(null);
    setFileName("");
    setError("");
  };

  return (
    <div className={cn("space-y-4", className)}>
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={cn(
          "relative border-2 border-dashed rounded-xl transition-all duration-300",
          isDragging
            ? "border-primary bg-primary/5 scale-[1.02]"
            : "border-border hover:border-primary/50",
          preview ? "p-4" : "p-8"
        )}
      >
        {preview ? (
          <div className="relative">
            <img
              src={preview}
              alt="Chart preview"
              className="w-full max-h-64 object-contain rounded-lg"
            />
            <button
              onClick={clearPreview}
              className="absolute top-2 right-2 p-1.5 bg-background/80 hover:bg-background rounded-full transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
            <div className="mt-3 flex items-center gap-2 text-sm text-muted-foreground">
              <Image className="w-4 h-4" />
              <span className="truncate">{fileName}</span>
            </div>
          </div>
        ) : (
          <label className="flex flex-col items-center gap-4 cursor-pointer">
            <div className="w-16 h-16 rounded-full bg-secondary flex items-center justify-center">
              <Upload className="w-7 h-7 text-primary" />
            </div>
            <div className="text-center">
              <p className="text-foreground font-medium">
                Drop your chart image here
              </p>
              <p className="text-sm text-muted-foreground mt-1">
                or click to browse • PNG, JPG, WebP up to 10MB
              </p>
            </div>
            <input
              type="file"
              accept="image/png,image/jpeg,image/webp"
              onChange={handleInputChange}
              className="hidden"
            />
          </label>
        )}
      </div>

      {error && (
        <div className="flex items-center gap-2 text-sm text-bearish">
          <AlertCircle className="w-4 h-4" />
          {error}
        </div>
      )}

      <Button
        variant="hero"
        size="lg"
        onClick={onAnalyze}
        disabled={!preview || isAnalyzing}
        className="w-full"
      >
        {isAnalyzing ? (
          <>
            <div className="w-4 h-4 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
            Analyzing Chart...
          </>
        ) : (
          <>
            <Upload className="w-4 h-4" />
            Analyze Chart
          </>
        )}
      </Button>
    </div>
  );
}
