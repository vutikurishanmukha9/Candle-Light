import { useCallback, useState } from "react";
import { Upload, Image, X, AlertCircle, CloudUpload } from "lucide-react";
import { cn } from "@/lib/utils";

interface UploadDropzoneProps {
  onFileSelect: (file: File) => void;
  className?: string;
}

export function UploadDropzone({
  onFileSelect,
  className,
}: UploadDropzoneProps) {
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

      onFileSelect(file);
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

  return (
    <div className={cn("space-y-3", className)}>
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={cn(
          "relative border-2 border-dashed rounded-2xl transition-all duration-300 cursor-pointer group",
          "min-h-[200px] sm:min-h-[280px] flex items-center justify-center",
          isDragging
            ? "border-primary bg-primary/5 scale-[1.02]"
            : "border-border/60 hover:border-primary/50 hover:bg-muted/30"
        )}
      >
        <label className="flex flex-col items-center gap-4 sm:gap-6 cursor-pointer p-6 sm:p-8 w-full">
          {/* Upload Icon */}
          <div
            className={cn(
              "w-16 h-16 sm:w-20 sm:h-20 rounded-2xl flex items-center justify-center transition-all duration-300",
              isDragging
                ? "bg-primary/20 scale-110"
                : "bg-muted group-hover:bg-primary/10 group-hover:scale-105"
            )}
          >
            <CloudUpload
              className={cn(
                "w-8 h-8 sm:w-10 sm:h-10 transition-colors",
                isDragging
                  ? "text-primary"
                  : "text-muted-foreground group-hover:text-primary"
              )}
            />
          </div>

          {/* Text */}
          <div className="text-center space-y-2">
            <p className="text-lg sm:text-xl font-medium text-foreground">
              {isDragging ? "Drop your chart here" : "Drop your chart image here"}
            </p>
            <p className="text-sm sm:text-base text-muted-foreground">
              or <span className="text-primary font-medium">click to browse</span>
            </p>
            <p className="text-xs sm:text-sm text-muted-foreground/70 pt-2">
              PNG, JPG, WebP up to 10MB
            </p>
          </div>

          <input
            type="file"
            accept="image/png,image/jpeg,image/webp"
            onChange={handleInputChange}
            className="hidden"
          />
        </label>

        {/* Animated border gradient on drag */}
        {isDragging && (
          <div className="absolute inset-0 rounded-2xl overflow-hidden pointer-events-none">
            <div className="absolute inset-0 bg-gradient-to-r from-primary/20 via-primary/5 to-primary/20 animate-shimmer" />
          </div>
        )}
      </div>

      {/* Error message */}
      {error && (
        <div className="flex items-center gap-2 text-sm text-destructive animate-fade-in">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}
