"""
Export Service

Handles exporting analysis data to various formats (PDF, CSV, JSON).
"""

import io
import csv
import json
from datetime import datetime
from typing import List, Optional, Literal
from dataclasses import dataclass

from app.models.analysis import Analysis


@dataclass
class ExportResult:
    """Result of an export operation"""
    content: bytes
    filename: str
    content_type: str


class ExportService:
    """Service for exporting analysis data."""
    
    def export_analysis(
        self,
        analysis: Analysis,
        format: Literal["json", "csv", "txt"] = "json"
    ) -> ExportResult:
        """
        Export a single analysis to the specified format.
        
        Args:
            analysis: The analysis to export
            format: Export format (json, csv, txt)
            
        Returns:
            ExportResult with content, filename, and content_type
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        if format == "json":
            return self._export_json(analysis, timestamp)
        elif format == "csv":
            return self._export_csv([analysis], timestamp)
        else:  # txt
            return self._export_txt(analysis, timestamp)
    
    def export_history(
        self,
        analyses: List[Analysis],
        format: Literal["json", "csv"] = "csv"
    ) -> ExportResult:
        """
        Export multiple analyses (history) to the specified format.
        
        Args:
            analyses: List of analyses to export
            format: Export format (json, csv)
            
        Returns:
            ExportResult with content, filename, and content_type
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        if format == "json":
            return self._export_history_json(analyses, timestamp)
        else:
            return self._export_csv(analyses, timestamp)
    
    def _export_json(self, analysis: Analysis, timestamp: str) -> ExportResult:
        """Export single analysis as JSON"""
        data = {
            "id": analysis.id,
            "created_at": analysis.created_at.isoformat() if analysis.created_at else None,
            "market_bias": analysis.market_bias,
            "confidence": analysis.confidence,
            "reasoning": analysis.reasoning,
            "ai_provider": analysis.ai_provider,
            "processing_time_ms": analysis.processing_time_ms,
            "patterns": analysis.patterns or [],
            "status": analysis.status,
        }
        
        content = json.dumps(data, indent=2).encode('utf-8')
        
        return ExportResult(
            content=content,
            filename=f"analysis_{analysis.id[:8]}_{timestamp}.json",
            content_type="application/json"
        )
    
    def _export_history_json(self, analyses: List[Analysis], timestamp: str) -> ExportResult:
        """Export multiple analyses as JSON"""
        data = {
            "export_date": datetime.utcnow().isoformat(),
            "total_analyses": len(analyses),
            "analyses": [
                {
                    "id": a.id,
                    "created_at": a.created_at.isoformat() if a.created_at else None,
                    "market_bias": a.market_bias,
                    "confidence": a.confidence,
                    "ai_provider": a.ai_provider,
                    "patterns_count": len(a.patterns) if a.patterns else 0,
                    "status": a.status,
                }
                for a in analyses
            ]
        }
        
        content = json.dumps(data, indent=2).encode('utf-8')
        
        return ExportResult(
            content=content,
            filename=f"analysis_history_{timestamp}.json",
            content_type="application/json"
        )
    
    def _export_csv(self, analyses: List[Analysis], timestamp: str) -> ExportResult:
        """Export analyses as CSV"""
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow([
            "ID",
            "Date",
            "Market Bias",
            "Confidence",
            "AI Provider",
            "Patterns",
            "Processing Time (ms)",
            "Status"
        ])
        
        # Data rows
        for a in analyses:
            patterns = ", ".join([p.get("name", "Unknown") for p in (a.patterns or [])])
            writer.writerow([
                a.id,
                a.created_at.isoformat() if a.created_at else "",
                a.market_bias or "",
                f"{a.confidence:.2f}" if a.confidence else "",
                a.ai_provider or "",
                patterns,
                a.processing_time_ms or "",
                a.status or ""
            ])
        
        content = output.getvalue().encode('utf-8')
        
        return ExportResult(
            content=content,
            filename=f"analysis_history_{timestamp}.csv",
            content_type="text/csv"
        )
    
    def _export_txt(self, analysis: Analysis, timestamp: str) -> ExportResult:
        """Export single analysis as formatted text"""
        lines = [
            "=" * 60,
            "CANDLESTICK PATTERN ANALYSIS REPORT",
            "=" * 60,
            "",
            f"Analysis ID: {analysis.id}",
            f"Date: {analysis.created_at.strftime('%Y-%m-%d %H:%M:%S') if analysis.created_at else 'N/A'}",
            f"AI Provider: {analysis.ai_provider or 'N/A'}",
            f"Processing Time: {analysis.processing_time_ms or 0}ms",
            "",
            "-" * 60,
            "RESULTS",
            "-" * 60,
            "",
            f"Market Bias: {(analysis.market_bias or 'N/A').upper()}",
            f"Confidence: {analysis.confidence:.1%}" if analysis.confidence else "Confidence: N/A",
            "",
            "Detected Patterns:",
        ]
        
        if analysis.patterns:
            for i, pattern in enumerate(analysis.patterns, 1):
                name = pattern.get("name", "Unknown")
                ptype = pattern.get("type", "neutral")
                conf = pattern.get("confidence", 0)
                lines.append(f"  {i}. {name} ({ptype}) - {conf:.1%} confidence")
        else:
            lines.append("  No patterns detected")
        
        lines.extend([
            "",
            "-" * 60,
            "REASONING",
            "-" * 60,
            "",
            analysis.reasoning or "No reasoning provided.",
            "",
            "=" * 60,
            "Generated by Candle-Light",
            "=" * 60,
        ])
        
        content = "\n".join(lines).encode('utf-8')
        
        return ExportResult(
            content=content,
            filename=f"analysis_{analysis.id[:8]}_{timestamp}.txt",
            content_type="text/plain"
        )


# Singleton instance
export_service = ExportService()
