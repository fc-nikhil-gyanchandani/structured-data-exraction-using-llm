#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from pydantic import BaseModel, Field

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('chunker.log')
    ]
)
logger = logging.getLogger(__name__)


# Layer 2: chunking (LangChain) - import these first as they're more stable
try:
    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter,
        MarkdownHeaderTextSplitter,
        HTMLHeaderTextSplitter,
    )
except ImportError as e:
    logger.error(f"Failed to import LangChain text splitters: {e}")
    sys.exit(1)

# Additional imports
import csv
import re

# Layer 1: extraction - with safer imports - DEFER until needed
UNSTRUCTURED_AVAILABLE = False
Element = None

def init_unstructured():
    """Initialize unstructured library - required for processing."""
    global UNSTRUCTURED_AVAILABLE, Element
    
    if UNSTRUCTURED_AVAILABLE:
        return True
    
    
    try:
        # Import Element first (this usually works)
        from unstructured.documents.elements import Element as UnstructuredElement
        try:
            from unstructured.partition import pdf, html, text
            
            # Create a wrapper function that routes to appropriate partitioner
            def partition(filename, **kwargs):
                """Wrapper function that routes to appropriate partitioner based on file type."""
                from pathlib import Path
                file_path = Path(filename)
                ext = file_path.suffix.lower()
                
                if ext == '.pdf':
                    return pdf.partition_pdf(filename, **kwargs)
                elif ext in ['.html', '.htm']:
                    return html.partition_html(filename, **kwargs)
                elif ext in ['.txt', '.md']:
                    return text.partition_text(filename, **kwargs)
                else:
                    # Default to text partitioner
                    return text.partition_text(filename, **kwargs)
            
            
        except Exception as specific_error:
            
            # Only try partition.auto if specific imports failed
            try:
                from unstructured.partition.auto import partition
                    
            except Exception as partition_error:
                logger.error(f"Failed to import partition.auto: {partition_error}")
                raise specific_error  # Re-raise the specific import error
        
        UNSTRUCTURED_AVAILABLE = True
        Element = UnstructuredElement
        
        # Make partition function available globally
        import sys
        current_module = sys.modules[__name__]
        current_module.partition = partition
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize unstructured library: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error("Unstructured library is required for document processing")
        UNSTRUCTURED_AVAILABLE = False
        Element = None
        return False

# ---------- Unified Chunk Schema ----------

class Chunk(BaseModel):
    chunk_id: str
    source_path: str
    filetype: str                        # "pdf" | "html" | "txt" | "csv"
    page_range: Optional[List[int]] = None  # [start_page, end_page] for PDFs/HTML
    section_path: Optional[List[str]] = None  # hierarchical path like ["Chapter 64", "Footwear", "Rates"]
    dom_path: Optional[str] = None       # DOM path for HTML elements
    table_id: Optional[str] = None       # table identifier if chunk is from a table
    bbox: Optional[List[float]] = None   # [x0, y0, x1, y1] bounding box coordinates
    text: str
    tokens_est: int                      # estimated token count
    overlap_with_prev: int = 0           # overlap with previous chunk in tokens
    fingerprint: str                     # SHA1 hash of chunk content
    schema_cues: List[str] = Field(default_factory=list)  # detected schema elements
    created_at: str                      # ISO timestamp

# ---------- Helpers ----------

import hashlib
from datetime import datetime

def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 characters per token."""
    return max(1, len(text) // 4)

def generate_fingerprint(text: str) -> str:
    """Generate SHA1 fingerprint of chunk content."""
    return f"sha1:{hashlib.sha1(text.encode('utf-8')).hexdigest()[:8]}"

def get_current_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.utcnow().isoformat() + "Z"

def extract_schema_cues(text: str) -> List[str]:
    """Extract potential schema cues from text content."""
    cues = []
    text_lower = text.lower()
    
    # Common schema patterns
    patterns = [
        r'\b(hs\s*code|tariff\s*code|classification)\b',
        r'\b(rate\s*of\s*duty|duty\s*rate|tax\s*rate)\b',
        r'\b(heading|subheading|chapter)\s*\d+',
        r'\b(description|product\s*description)\b',
        r'\b(unit\s*of\s*measure|uom)\b',
        r'\b(origin|country\s*of\s*origin)\b',
        r'\b(value|price|cost)\b',
        r'\b(quantity|amount|weight)\b'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        cues.extend([match.title() for match in matches])
    
    return list(set(cues))  # Remove duplicates



def detect_source_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in [".pdf"]:
        return "pdf"
    if ext in [".html", ".htm"]:
        return "html"
    if ext in [".txt", ".md", ".rst"]:
        return "txt"
    if ext in [".csv", ".tsv", ".xlsx"]:
        return "csv"
    # let unstructured handle many more, but we'll label as txt
    return "txt"

def elements_to_text(element: Any) -> str:
    # unstructured elements have .text; tables may have .metadata.text_as_html etc.
    try:
        text = (element.text or "").strip()
        return text
    except Exception as e:
        return ""

def element_chunk_type(element: Any) -> str:
    # Map unstructured element types to a few stable kinds
    et = element.category if hasattr(element, "category") else element.__class__.__name__
    et_lower = str(et).lower()
    
    if "title" in et_lower or "header" in et_lower:
        chunk_type = "title"
    elif "list" in et_lower:
        chunk_type = "list"
    elif "table" in et_lower:
        chunk_type = "table"
    else:
        chunk_type = "text"
    
    return chunk_type

def base_metadata(element: Any) -> Dict[str, Any]:
    md: Dict[str, Any] = {}
    try:
        if hasattr(element, "metadata") and element.metadata:
            m = element.metadata
            # common fields exposed by unstructured
            md["filename"] = getattr(m, "filename", None)
            md["last_modified"] = getattr(m, "last_modified", None)
            md["languages"] = getattr(m, "languages", None)
            md["text_as_html"] = getattr(m, "text_as_html", None) if hasattr(m, "text_as_html") else None
            # layout/provenance if available
            md["page_number"] = getattr(m, "page_number", None)
            # "coordinates" may be present (x,y info)
            coords = getattr(m, "coordinates", None)
            if coords and getattr(coords, "points", None):
                # store as bbox if provided
                md["coordinates_points"] = coords.points
    except Exception as e:
        pass
    return {k: v for k, v in md.items() if v is not None}


# ---------- Layer 1: Extraction / Normalization ----------

def extract_unstructured(path: Path) -> List[Chunk]:
    """Use Unstructured to partition PDFs/HTML/TXT into elements, then normalize to Chunk."""
    if not init_unstructured():
        logger.error(f"Unstructured library not available, cannot process {path.name}")
        raise RuntimeError("Unstructured library is required for document processing")
    
    try:
        
        # Get the partition function from our global scope
        import sys
        current_module = sys.modules[__name__]
        partition_func = getattr(current_module, 'partition', None)
        
        if partition_func is None:
            # Fallback: try to get it from the init function
            logger.warning("Partition function not found in global scope, trying to recreate...")
            from unstructured.partition import pdf, html, text
            
            def partition(filename, **kwargs):
                from pathlib import Path
                file_path = Path(filename)
                ext = file_path.suffix.lower()
                
                if ext == '.pdf':
                    return pdf.partition_pdf(filename, **kwargs)
                elif ext in ['.html', '.htm']:
                    return html.partition_html(filename, **kwargs)
                elif ext in ['.txt', '.md']:
                    return text.partition_text(filename, **kwargs)
                else:
                    return text.partition_text(filename, **kwargs)
        
        elements = partition_func(str(path))
        
        chunks: List[Chunk] = []
        source_type = detect_source_type(path)
        
        for i, el in enumerate(elements):
            text = elements_to_text(el)
            if not text:
                continue
                
            ctype = element_chunk_type(el)
            md = base_metadata(el)
            page_number = md.get("page_number")
            bbox = None  # Unstructured may expose coordinates via metadata if configured with OCR/layout; keep None by default

            # Generate chunk ID with more descriptive format
            chunk_id = f"{path.stem}_{page_number:03d}_p{page_number}_t{i}" if page_number is not None else f"{path.stem}_t{i}"
            
            # Extract table ID if this is a table chunk
            table_id = None
            if ctype == "table" and "table_id" in md:
                table_id = md["table_id"]
            elif ctype == "table":
                table_id = f"t{i}"
            
            # Extract section path from metadata if available
            section_path = None
            if "section_path" in md:
                section_path = md["section_path"]
            
            chunk = Chunk(
                chunk_id=chunk_id,
                source_path=str(path),
                filetype=source_type,
                page_range=[page_number, page_number] if page_number is not None else None,
                section_path=section_path,
                table_id=table_id,
                bbox=bbox,
                text=text,
                tokens_est=estimate_tokens(text),
                fingerprint=generate_fingerprint(text),
                schema_cues=extract_schema_cues(text),
                created_at=get_current_timestamp()
            )
            chunks.append(chunk)
        
        return chunks
    except Exception as e:
        logger.error(f"Unstructured processing failed for {path.name}: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return []

def extract_csv(path: Path, max_rows: Optional[int] = None) -> List[Chunk]:
    """Process CSV/TSV/XLSX files with 100 rows per chunk using old schema format."""
    
    # Read the file based on extension
    if path.suffix.lower() == ".xlsx":
        try:
            df = pd.read_excel(path)
        except ImportError:
            logger.error("openpyxl not available for XLSX processing. Install with: pip install openpyxl")
            return []
    elif path.suffix.lower() == ".tsv":
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)
    
    if max_rows is not None:
        df = df.head(max_rows)
    
    chunks: List[Chunk] = []
    rows_per_chunk = 100
    
    # Process in chunks of 100 rows
    for start_idx in range(0, len(df), rows_per_chunk):
        end_idx = min(start_idx + rows_per_chunk - 1, len(df) - 1)
        chunk_df = df.iloc[start_idx:end_idx + 1]
        
        # Convert chunk to list of dictionaries
        rows_data = [row.to_dict() for _, row in chunk_df.iterrows()]
        text = json.dumps(rows_data, ensure_ascii=False)
        
        # Generate fingerprint for this chunk
        fingerprint = generate_fingerprint(text)[5:]  # Remove "sha1:" prefix
        
        # Create chunk ID with row range and fingerprint
        chunk_id = f"{path.name}:rows-{start_idx}-{end_idx}-{fingerprint}"
        
        # Create chunk using old schema format
        chunk = Chunk(
            chunk_id=chunk_id,
            source_path=str(path),
            filetype="csv",
            text=text,
            tokens_est=estimate_tokens(text),
            fingerprint=generate_fingerprint(text),
            schema_cues=extract_schema_cues(text),
            created_at=get_current_timestamp()
        )
        
        # Convert to old schema format for CSV chunks
        old_schema_chunk = {
            "id": chunk_id,
            "source_path": str(path),
            "source_type": "csv",
            "chunk_type": "row",
            "text": text,
            "metadata": {
                "columns": list(df.columns),
                "row_range": [start_idx, end_idx],
                "tokens_est": estimate_tokens(text),
                "fingerprint": fingerprint,
                "overlap_with_prev": 0,
                "created_at": get_current_timestamp()
            },
            "page_number": None,
            "bbox": None,
            "rownum": None
        }
        
        # Create a special chunk object that will serialize to the old format
        class OldSchemaChunk:
            def __init__(self, data):
                self.data = data
                # Add tokens_est attribute for compatibility with normalization
                self.tokens_est = data['metadata']['tokens_est']
            
            def model_dump(self):
                return self.data
        
        chunks.append(OldSchemaChunk(old_schema_chunk))
    
    return chunks

def extract_layer1(path: Path, max_csv_rows: Optional[int] = None) -> List[Chunk]:
    st = detect_source_type(path)
    
    if st == "csv":
        return extract_csv(path, max_csv_rows)
    
    # Use unstructured for all other file types
    return extract_unstructured(path)

# ---------- Layer 2: Chunking with LangChain ----------

@dataclass
class SplitterConfig:
    target_chunk_tokens: int = 500         # target tokens per chunk
    chunk_overlap_tokens: int = 50         # overlap tokens
    respect_markdown: bool = True
    use_html_headers: bool = True

def normalize_chunk_sizes(chunks: List[Chunk], target_tokens: int = 500, overlap_tokens: int = 50) -> List[Chunk]:
    """Post-process chunks to ensure more even token distribution around target_tokens."""
    if not chunks:
        return chunks
    
    
    normalized = []
    i = 0
    
    while i < len(chunks):
        current_chunk = chunks[i]
        # Handle both new schema chunks and old schema chunks
        if hasattr(current_chunk, 'tokens_est'):
            current_tokens = current_chunk.tokens_est
        elif hasattr(current_chunk, 'data') and 'metadata' in current_chunk.data:
            current_tokens = current_chunk.data['metadata'].get('tokens_est', 0)
        else:
            # Skip chunks without token information
            i += 1
            continue
        
        # If chunk is too small, try to merge with next chunk(s)
        if current_tokens < target_tokens * 0.75:  # Less than 75% of target
            # Get text from chunk
            if hasattr(current_chunk, 'text'):
                merged_text = current_chunk.text
            elif hasattr(current_chunk, 'data'):
                merged_text = current_chunk.data.get('text', '')
            else:
                merged_text = ''
            merged_tokens = current_tokens
            j = i + 1
            
            # Keep merging while we're under target and have more chunks
            while (j < len(chunks) and 
                   merged_tokens < target_tokens * 1.25 and  # Don't exceed 125% of target
                   getattr(chunks[j], 'source_path', '') == getattr(current_chunk, 'source_path', '')):  # Same source file
                
                next_chunk = chunks[j]
                # Get text from next chunk
                if hasattr(next_chunk, 'text'):
                    next_text = next_chunk.text
                elif hasattr(next_chunk, 'data'):
                    next_text = next_chunk.data.get('text', '')
                else:
                    next_text = ''
                
                merged_text += " " + next_text
                merged_tokens = estimate_tokens(merged_text)
                j += 1
            
            # Create merged chunk
            if j > i + 1:  # We merged some chunks
                # Get attributes from current chunk
                if hasattr(current_chunk, 'chunk_id'):
                    chunk_id = current_chunk.chunk_id
                elif hasattr(current_chunk, 'data'):
                    chunk_id = current_chunk.data.get('id', f'merged_chunk_{i}')
                else:
                    chunk_id = f'merged_chunk_{i}'
                
                if hasattr(current_chunk, 'source_path'):
                    source_path = current_chunk.source_path
                elif hasattr(current_chunk, 'data'):
                    source_path = current_chunk.data.get('source_path', '')
                else:
                    source_path = ''
                
                if hasattr(current_chunk, 'filetype'):
                    filetype = current_chunk.filetype
                elif hasattr(current_chunk, 'data'):
                    filetype = current_chunk.data.get('source_type', 'unknown')
                else:
                    filetype = 'unknown'
                page_range = getattr(current_chunk, 'page_range', None)
                section_path = getattr(current_chunk, 'section_path', None)
                dom_path = getattr(current_chunk, 'dom_path', None)
                table_id = getattr(current_chunk, 'table_id', None)
                bbox = getattr(current_chunk, 'bbox', None)
                overlap_with_prev = getattr(current_chunk, 'overlap_with_prev', 0)
                schema_cues = getattr(current_chunk, 'schema_cues', [])
                
                # Collect schema cues from merged chunks
                all_schema_cues = schema_cues.copy()
                for chunk in chunks[i+1:j]:
                    if hasattr(chunk, 'schema_cues'):
                        all_schema_cues.extend(chunk.schema_cues)
                    elif hasattr(chunk, 'data') and 'schema_cues' in chunk.data:
                        all_schema_cues.extend(chunk.data['schema_cues'])
                
                new_chunk = Chunk(
                    chunk_id=f"{chunk_id}_merged",
                    source_path=source_path,
                    filetype=filetype,
                    page_range=page_range,
                    section_path=section_path,
                    dom_path=dom_path,
                    table_id=table_id,
                    bbox=bbox,
                    text=merged_text,
                    tokens_est=merged_tokens,
                    overlap_with_prev=overlap_with_prev,
                    fingerprint=generate_fingerprint(merged_text),
                    schema_cues=list(set(all_schema_cues)),
                    created_at=get_current_timestamp()
                )
                normalized.append(new_chunk)
                i = j
            else:
                # Couldn't merge, keep as is
                normalized.append(current_chunk)
                i += 1
        
        # If chunk is too large, try to split it further
        elif current_tokens > target_tokens * 1.5:  # More than 150% of target
            # Get text from chunk
            if hasattr(current_chunk, 'text'):
                text = current_chunk.text
            elif hasattr(current_chunk, 'data'):
                text = current_chunk.data.get('text', '')
            else:
                text = ''
            
            words = text.split()
            words_per_chunk = len(words) * target_tokens // current_tokens
            
            for k in range(0, len(words), words_per_chunk):
                chunk_words = words[k:k + words_per_chunk]
                if not chunk_words:
                    continue
                    
                chunk_text = " ".join(chunk_words)
                chunk_tokens = estimate_tokens(chunk_text)
                
                # Get attributes from current chunk
                if hasattr(current_chunk, 'chunk_id'):
                    chunk_id = current_chunk.chunk_id
                elif hasattr(current_chunk, 'data'):
                    chunk_id = current_chunk.data.get('id', f'split_chunk_{i}')
                else:
                    chunk_id = f'split_chunk_{i}'
                
                if hasattr(current_chunk, 'source_path'):
                    source_path = current_chunk.source_path
                elif hasattr(current_chunk, 'data'):
                    source_path = current_chunk.data.get('source_path', '')
                else:
                    source_path = ''
                
                if hasattr(current_chunk, 'filetype'):
                    filetype = current_chunk.filetype
                elif hasattr(current_chunk, 'data'):
                    filetype = current_chunk.data.get('source_type', 'unknown')
                else:
                    filetype = 'unknown'
                page_range = getattr(current_chunk, 'page_range', None)
                section_path = getattr(current_chunk, 'section_path', None)
                dom_path = getattr(current_chunk, 'dom_path', None)
                table_id = getattr(current_chunk, 'table_id', None)
                bbox = getattr(current_chunk, 'bbox', None)
                overlap_with_prev = getattr(current_chunk, 'overlap_with_prev', 0)
                
                new_chunk = Chunk(
                    chunk_id=f"{chunk_id}_split{k//words_per_chunk}",
                    source_path=source_path,
                    filetype=filetype,
                    page_range=page_range,
                    section_path=section_path,
                    dom_path=dom_path,
                    table_id=table_id,
                    bbox=bbox,
                    text=chunk_text,
                    tokens_est=chunk_tokens,
                    overlap_with_prev=overlap_tokens if k > 0 else overlap_with_prev,
                    fingerprint=generate_fingerprint(chunk_text),
                    schema_cues=extract_schema_cues(chunk_text),
                    created_at=get_current_timestamp()
                )
                normalized.append(new_chunk)
            
            i += 1
        
        else:
            # Chunk is within acceptable range, keep as is
            normalized.append(current_chunk)
            i += 1
    
    return normalized

def split_text_payloads(chunks: List[Chunk], cfg: SplitterConfig) -> List[Chunk]:
    """Apply LangChain splitters to the text field while preserving metadata."""
    
    out: List[Chunk] = []

    # Use TokenTextSplitter for accurate token-based chunking
    try:
        from langchain_text_splitters import TokenTextSplitter
        rsplit = TokenTextSplitter(
            chunk_size=cfg.target_chunk_tokens,
            chunk_overlap=cfg.chunk_overlap_tokens,
            model_name="gpt-3.5-turbo"  # Use a standard model for token counting
        )
    except ImportError:
        logger.warning("TokenTextSplitter not available, falling back to character-based splitting")
        # Fallback to character-based with better estimation
        char_size = cfg.target_chunk_tokens * 4  # Rough conversion
        char_overlap = cfg.chunk_overlap_tokens * 4
        rsplit = RecursiveCharacterTextSplitter(
            chunk_size=char_size,
            chunk_overlap=char_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    for i, ch in enumerate(chunks):
        # Handle both new schema chunks and old schema chunks
        if hasattr(ch, 'data'):  # OldSchemaChunk
            # For CSV chunks with old schema, don't split them further
            out.append(ch)
            continue
        
        text = ch.text or ""
        
        # Specialized handling for Markdown/HTML if you want more structure-aware splitting
        # Here we keep it simple: run RecursiveCharacterTextSplitter for all.
        splits = rsplit.split_text(text)

        if len(splits) == 0:
            logger.warning(f"Chunk {ch.chunk_id} produced no splits, skipping")
            continue

        for j, piece in enumerate(splits):
            # Calculate overlap with previous chunk
            overlap_with_prev = 0
            if j > 0:
                # Estimate overlap based on chunk_overlap setting
                overlap_with_prev = min(cfg.chunk_overlap // 4, estimate_tokens(piece))  # Convert chars to tokens
            
            # Generate new chunk ID with split indicator
            new_chunk_id = f"{ch.chunk_id}_s{j}"
            
            out.append(
                Chunk(
                    chunk_id=new_chunk_id,
                    source_path=ch.source_path,
                    filetype=ch.filetype,
                    page_range=ch.page_range,
                    section_path=ch.section_path,
                    dom_path=ch.dom_path,
                    table_id=ch.table_id,
                    bbox=ch.bbox,
                    text=piece,
                    tokens_est=estimate_tokens(piece),
                    overlap_with_prev=overlap_with_prev,
                    fingerprint=generate_fingerprint(piece),
                    schema_cues=extract_schema_cues(piece),
                    created_at=get_current_timestamp()
                )
            )
    
    
    # Post-process to ensure more even token distribution
    out = normalize_chunk_sizes(out, target_tokens=cfg.target_chunk_tokens, overlap_tokens=cfg.chunk_overlap_tokens)
    
    return out

# ---------- I/O ----------

def write_jsonl(chunks: Iterable[Chunk], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    chunk_count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            # Handle both new schema chunks and old schema chunks
            if hasattr(ch, 'model_dump'):
                f.write(json.dumps(ch.model_dump(), ensure_ascii=False) + "\n")
            else:
                # Handle old schema chunks
                f.write(json.dumps(ch, ensure_ascii=False) + "\n")
            chunk_count += 1
    

# ---------- CLI ----------

def main():
    
    # Initialize unstructured library at startup
    if not init_unstructured():
        logger.error("Failed to initialize unstructured library. Exiting.")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(
        description="Two-layer chunking pipeline using Unstructured (extraction) + LangChain (chunking)."
    )
    parser.add_argument("input", help="File or directory path")
    parser.add_argument("--out", default="out/chunks.jsonl", help="Output JSONL file")
    parser.add_argument("--max-csv-rows", type=int, default=None, help="Cap CSV rows for quick tests")
    parser.add_argument("--chunk-tokens", type=int, default=500, help="Target tokens per chunk")
    parser.add_argument("--chunk-overlap-tokens", type=int, default=50, help="Overlap tokens between chunks")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--test-single", action="store_true", help="Test processing of just the first file found")
    args = parser.parse_args()

    # Set log level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")


    input_path = Path(args.input)
    out_path = Path(args.out)
    
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {out_path}")

    cfg = SplitterConfig(target_chunk_tokens=args.chunk_tokens, chunk_overlap_tokens=args.chunk_overlap_tokens)
    logger.info(f"Using chunking config: {cfg}")

    all_chunks: List[Chunk] = []

    if input_path.is_dir():
        files = list(sorted(input_path.rglob("*")))
        logger.info(f"Found {len(files)} total items in directory")
        
        # Limit to first file if testing
        if args.test_single:
            files = [f for f in files if f.is_file()][:1]
            logger.info(f"Test mode: processing only first file: {files[0].name if files else 'None'}")
        
        for i, p in enumerate(files):
            if p.is_file():
                try:
                    layer1 = extract_layer1(p, max_csv_rows=args.max_csv_rows)
                    layer2 = split_text_payloads(layer1, cfg)
                    all_chunks.extend(layer2)
                    
                    # Exit after first file if testing
                    if args.test_single:
                        logger.info("Test mode: stopping after first file")
                        break
                        
                except Exception as e:
                    logger.error(f"Failed to process {p.name}: {e}")
                    if args.test_single:
                        logger.error("Test mode: stopping due to error")
                        break
                    continue
    else:
        layer1 = extract_layer1(input_path, max_csv_rows=args.max_csv_rows)
        layer2 = split_text_payloads(layer1, cfg)
        
        all_chunks.extend(layer2)

    logger.info(f"Pipeline complete. Writing {len(all_chunks)} total chunks to output")
    write_jsonl(all_chunks, out_path)
    logger.info(f"Successfully processed {len(all_chunks)} chunks -> {out_path}")
    
    # Add summary
    logger.info("=" * 60)
    logger.info("PROCESSING SUMMARY:")
    logger.info("=" * 60)
    logger.info(f"Total files processed: {len([f for f in Path(args.input).rglob('*') if f.is_file()]) if Path(args.input).is_dir() else 1}")
    logger.info(f"Total chunks created: {len(all_chunks)}")
    logger.info(f"Output file: {out_path}")
    logger.info("=" * 60)

if __name__ == "__main__":
    # Add a simple test mode
    if len(sys.argv) > 1 and sys.argv[1] == "--test-imports":
        logger.info("Testing imports only...")
        try:
            logger.info("Testing basic imports...")
            import pandas as pd
            logger.info("pandas imported successfully")
            
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            logger.info("LangChain splitters imported successfully")
            
            logger.info("Testing unstructured import...")
            if init_unstructured():
                logger.info("unstructured library initialized successfully")
            else:
                logger.info("unstructured library failed to initialize")
            
            logger.info("All import tests completed")
            
        except Exception as e:
            logger.error(f"Import test failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            sys.exit(1)
    else:
        main()
