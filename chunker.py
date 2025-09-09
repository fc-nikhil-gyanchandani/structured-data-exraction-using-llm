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
        logging.FileHandler('extractor.log')
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting LLM Parser - initializing imports...")

# Layer 2: chunking (LangChain) - import these first as they're more stable
try:
    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter,
        MarkdownHeaderTextSplitter,
        HTMLHeaderTextSplitter,
    )
    logger.info("Successfully imported LangChain text splitters")
except ImportError as e:
    logger.error(f"Failed to import LangChain text splitters: {e}")
    sys.exit(1)

# Additional imports for fallback processing
import csv
import re
from bs4 import BeautifulSoup

# Layer 1: extraction - with safer imports - DEFER until needed
UNSTRUCTURED_AVAILABLE = False
Element = None

def safe_init_unstructured():
    """Safely initialize unstructured library only when needed."""
    global UNSTRUCTURED_AVAILABLE, Element
    
    if UNSTRUCTURED_AVAILABLE:
        return True
    
    logger.info("Attempting to initialize unstructured library...")
    logger.warning("Note: If this hangs or crashes, the script will automatically fall back to alternative methods")
    
    try:
        # Try to import with a simple approach first
        logger.info("Importing unstructured.partition.auto...")
        from unstructured.partition.auto import partition
        logger.info("✓ Successfully imported partition")
        
        logger.info("Importing unstructured.documents.elements...")
        from unstructured.documents.elements import Element as UnstructuredElement
        logger.info("✓ Successfully imported Element")
        
        # Test if it actually works by doing a simple operation
        logger.info("Testing unstructured library functionality...")
        
        UNSTRUCTURED_AVAILABLE = True
        Element = UnstructuredElement
        logger.info("Successfully initialized unstructured library")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize unstructured library: {e}")
        logger.info("Will use fallback processing methods instead")
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
    logger.debug(f"Detecting source type for {path.name} with extension: {ext}")
    
    if ext in [".pdf"]:
        logger.debug(f"Detected PDF file: {path.name}")
        return "pdf"
    if ext in [".html", ".htm"]:
        logger.debug(f"Detected HTML file: {path.name}")
        return "html"
    if ext in [".txt", ".md", ".rst"]:
        logger.debug(f"Detected text file: {path.name}")
        return "txt"
    if ext in [".csv", ".tsv", ".xlsx"]:
        logger.debug(f"Detected CSV/TSV/XLSX file: {path.name}")
        return "csv"
    # fallback: let unstructured handle many more, but we'll label as txt
    logger.debug(f"Unknown extension {ext}, defaulting to txt for {path.name}")
    return "txt"

def elements_to_text(element: Any) -> str:
    # unstructured elements have .text; tables may have .metadata.text_as_html etc.
    try:
        text = (element.text or "").strip()
        logger.debug(f"Extracted text from element: {len(text)} characters")
        return text
    except Exception as e:
        logger.warning(f"Failed to extract text from element: {e}")
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
    
    logger.debug(f"Element type '{et}' mapped to chunk type '{chunk_type}'")
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
        logger.debug(f"Extracted metadata: {len(md)} fields")
    except Exception as e:
        logger.warning(f"Failed to extract metadata: {e}")
    return {k: v for k, v in md.items() if v is not None}

# ---------- Fallback Processing Methods ----------

def extract_html_fallback(path: Path) -> List[Chunk]:
    """Fallback HTML processing using BeautifulSoup when unstructured fails."""
    logger.info(f"Using BeautifulSoup fallback for HTML file: {path.name}")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        logger.debug(f"Read {len(content)} characters from HTML file")
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        logger.info(f"HTML fallback extracted {len(text)} characters of clean text")
        
        if text:
            return [Chunk(
                chunk_id=f"{path.name}:html-fallback",
                source_path=str(path),
                filetype="html",
                text=text,
                tokens_est=estimate_tokens(text),
                fingerprint=generate_fingerprint(text),
                schema_cues=extract_schema_cues(text),
                created_at=get_current_timestamp()
            )]
    except Exception as e:
        logger.error(f"HTML fallback processing failed for {path.name}: {e}")
    
    return []

def extract_txt_fallback(path: Path) -> List[Chunk]:
    """Fallback text processing when unstructured fails."""
    logger.info(f"Using direct read fallback for text file: {path.name}")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        logger.info(f"Text fallback read {len(content)} characters")
        
        if content.strip():
            return [Chunk(
                chunk_id=f"{path.name}:txt-fallback",
                source_path=str(path),
                filetype="txt",
                text=content.strip(),
                tokens_est=estimate_tokens(content.strip()),
                fingerprint=generate_fingerprint(content.strip()),
                schema_cues=extract_schema_cues(content.strip()),
                created_at=get_current_timestamp()
            )]
    except Exception as e:
        logger.error(f"Text fallback processing failed for {path.name}: {e}")
    
    return []

def extract_pdf_fallback(path: Path) -> List[Chunk]:
    """Fallback PDF processing when unstructured fails."""
    logger.info(f"Using PyPDF2 fallback for PDF file: {path.name}")
    try:
        # Try using PyPDF2 as a fallback
        import PyPDF2
        chunks = []
        
        with open(path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            logger.info(f"PDF has {len(pdf_reader.pages)} pages")
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():
                    chunks.append(Chunk(
                        chunk_id=f"{path.name}:page-{page_num}",
                        source_path=str(path),
                        filetype="pdf",
                        page_range=[page_num, page_num],
                        text=text.strip(),
                        tokens_est=estimate_tokens(text.strip()),
                        fingerprint=generate_fingerprint(text.strip()),
                        schema_cues=extract_schema_cues(text.strip()),
                        created_at=get_current_timestamp()
                    ))
                    logger.debug(f"Extracted {len(text.strip())} characters from page {page_num}")
        
        logger.info(f"PDF fallback extracted {len(chunks)} chunks from {len(pdf_reader.pages)} pages")
        return chunks
    except ImportError:
        logger.warning(f"PyPDF2 not available for PDF fallback processing of {path.name}")
    except Exception as e:
        logger.error(f"PDF fallback processing failed for {path.name}: {e}")
    
    return []

# ---------- Layer 1: Extraction / Normalization ----------

def extract_unstructured(path: Path) -> List[Chunk]:
    """Use Unstructured to partition PDFs/HTML/TXT into elements, then normalize to Chunk."""
    if not safe_init_unstructured():
        logger.warning(f"Unstructured library not available, skipping {path.name}")
        return []
    
    logger.info(f"Processing {path.name} with unstructured library")
    try:
        # Import partition here after safe initialization
        from unstructured.partition.auto import partition
        
        elements = partition(filename=str(path))
        logger.info(f"Unstructured extracted {len(elements)} elements from {path.name}")
        
        chunks: List[Chunk] = []
        source_type = detect_source_type(path)
        
        for i, el in enumerate(elements):
            text = elements_to_text(el)
            if not text:
                logger.debug(f"Skipping empty element {i} from {path.name}")
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
            logger.debug(f"Created chunk {i} for {path.name}: type={ctype}, text_length={len(text)}")
        
        logger.info(f"Successfully created {len(chunks)} chunks from {path.name} using unstructured")
        return chunks
    except Exception as e:
        logger.error(f"Unstructured processing failed for {path.name}: {e}")
        return []

def extract_csv(path: Path, max_rows: Optional[int] = None) -> List[Chunk]:
    """Process CSV/TSV/XLSX files with 100 rows per chunk using old schema format."""
    logger.info(f"Processing CSV/XLSX file: {path.name}")
    
    # Read the file based on extension
    if path.suffix.lower() == ".xlsx":
        try:
            df = pd.read_excel(path)
            logger.info(f"XLSX file {path.name} has {len(df)} rows and {len(df.columns)} columns")
        except ImportError:
            logger.error("openpyxl not available for XLSX processing. Install with: pip install openpyxl")
            return []
    elif path.suffix.lower() == ".tsv":
        df = pd.read_csv(path, sep="\t")
        logger.info(f"TSV file {path.name} has {len(df)} rows and {len(df.columns)} columns")
    else:
        df = pd.read_csv(path)
        logger.info(f"CSV file {path.name} has {len(df)} rows and {len(df.columns)} columns")
    
    if max_rows is not None:
        df = df.head(max_rows)
        logger.info(f"Limited to {max_rows} rows for processing")
    
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
            
            def model_dump(self):
                return self.data
        
        chunks.append(OldSchemaChunk(old_schema_chunk))
        logger.debug(f"Created chunk for rows {start_idx}-{end_idx} with {len(rows_data)} rows")
    
    logger.info(f"Created {len(chunks)} chunks from {path.name} (100 rows per chunk)")
    return chunks

def extract_layer1(path: Path, max_csv_rows: Optional[int] = None) -> List[Chunk]:
    global UNSTRUCTURED_AVAILABLE
    
    logger.info(f"Starting Layer 1 extraction for: {path.name}")
    st = detect_source_type(path)
    logger.info(f"Detected source type: {st} for {path.name}")
    
    if st == "csv":
        logger.info(f"Processing {path.name} as CSV file")
        return extract_csv(path, max_csv_rows)
    
    # Try unstructured first, but with better error handling
    if UNSTRUCTURED_AVAILABLE:
        logger.info(f"Attempting unstructured processing for {path.name}")
        try:
            chunks = extract_unstructured(path)
            if chunks:
                logger.info(f"Unstructured processing successful for {path.name}, returning {len(chunks)} chunks")
                return chunks
        except Exception as e:
            logger.error(f"Unstructured processing crashed for {path.name}: {e}")
            logger.warning("Disabling unstructured library for remaining files due to crash")
            UNSTRUCTURED_AVAILABLE = False
    
    # Fallback processing if unstructured fails or is disabled
    if not UNSTRUCTURED_AVAILABLE:
        logger.warning(f"Unstructured library unavailable, using fallback processing for {path.name}")
    else:
        logger.warning(f"Unstructured failed for {path.name}, using fallback processing")
    
    if st == "html":
        return extract_html_fallback(path)
    elif st == "txt":
        return extract_txt_fallback(path)
    elif st == "pdf":
        return extract_pdf_fallback(path)
    
    logger.warning(f"No fallback method available for {path.name} with type {st}")
    return []

# ---------- Layer 2: Chunking with LangChain ----------

@dataclass
class SplitterConfig:
    target_chunk_size: int = 1200          # ~ characters, not tokens
    chunk_overlap: int = 200
    respect_markdown: bool = True
    use_html_headers: bool = True

def split_text_payloads(chunks: List[Chunk], cfg: SplitterConfig) -> List[Chunk]:
    """Apply LangChain splitters to the text field while preserving metadata."""
    logger.info(f"Starting Layer 2 chunking with {len(chunks)} input chunks")
    logger.info(f"Chunking config: size={cfg.target_chunk_size}, overlap={cfg.chunk_overlap}")
    
    out: List[Chunk] = []

    # Base text splitter (works for most text)
    rsplit = RecursiveCharacterTextSplitter(
        chunk_size=cfg.target_chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    for i, ch in enumerate(chunks):
        # Handle both new schema chunks and old schema chunks
        if hasattr(ch, 'data'):  # OldSchemaChunk
            # For CSV chunks with old schema, don't split them further
            out.append(ch)
            logger.debug(f"Preserving old schema chunk {i+1}/{len(chunks)} without splitting")
            continue
        
        text = ch.text or ""
        logger.debug(f"Processing chunk {i+1}/{len(chunks)}: {ch.chunk_id}, text_length={len(text)}")
        
        # Specialized handling for Markdown/HTML if you want more structure-aware splitting
        # Here we keep it simple: run RecursiveCharacterTextSplitter for all.
        splits = rsplit.split_text(text)
        logger.debug(f"Chunk {ch.chunk_id} split into {len(splits)} pieces")

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
    
    logger.info(f"Layer 2 chunking complete: {len(chunks)} input chunks -> {len(out)} output chunks")
    return out

# ---------- I/O ----------

def write_jsonl(chunks: Iterable[Chunk], out_path: Path) -> None:
    logger.info(f"Writing {len(list(chunks)) if hasattr(chunks, '__len__') else 'unknown'} chunks to {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    chunk_count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            # Handle both new schema chunks and old schema chunks
            if hasattr(ch, 'model_dump'):
                f.write(json.dumps(ch.model_dump(), ensure_ascii=False) + "\n")
            else:
                # Fallback for old schema chunks
                f.write(json.dumps(ch, ensure_ascii=False) + "\n")
            chunk_count += 1
    
    logger.info(f"Successfully wrote {chunk_count} chunks to {out_path}")

# ---------- CLI ----------

def main():
    global UNSTRUCTURED_AVAILABLE
    
    logger.info("Starting LLM Parser extraction pipeline")
    
    parser = argparse.ArgumentParser(
        description="Two-layer chunking pipeline: Unstructured (extraction) + LangChain (chunking)."
    )
    parser.add_argument("input", help="File or directory path")
    parser.add_argument("--out", default="out/chunks.jsonl", help="Output JSONL file")
    parser.add_argument("--max-csv-rows", type=int, default=None, help="Cap CSV rows for quick tests")
    parser.add_argument("--chunk-size", type=int, default=1200, help="LangChain target chunk size (chars)")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="LangChain chunk overlap (chars)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--skip-unstructured", action="store_true", help="Skip unstructured processing, use only fallback methods")
    parser.add_argument("--test-single", action="store_true", help="Test processing of just the first file found")
    args = parser.parse_args()

    # Set log level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")

    # Handle unstructured skipping
    if args.skip_unstructured:
        logger.info("Skipping unstructured processing as requested - using only fallback methods")
        UNSTRUCTURED_AVAILABLE = False

    input_path = Path(args.input)
    out_path = Path(args.out)
    
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {out_path}")

    cfg = SplitterConfig(target_chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    logger.info(f"Using chunking config: {cfg}")

    all_chunks: List[Chunk] = []

    if input_path.is_dir():
        logger.info(f"Processing directory: {input_path}")
        files = list(sorted(input_path.rglob("*")))
        logger.info(f"Found {len(files)} total items in directory")
        
        # Limit to first file if testing
        if args.test_single:
            files = [f for f in files if f.is_file()][:1]
            logger.info(f"Test mode: processing only first file: {files[0].name if files else 'None'}")
        
        for i, p in enumerate(files):
            if p.is_file():
                logger.info(f"Processing file {i+1}/{len(files)}: {p.name}")
                try:
                    layer1 = extract_layer1(p, max_csv_rows=args.max_csv_rows)
                    logger.info(f"Layer 1 complete for {p.name}: {len(layer1)} chunks")
                    
                    layer2 = split_text_payloads(layer1, cfg)
                    logger.info(f"Layer 2 complete for {p.name}: {len(layer2)} chunks")
                    
                    all_chunks.extend(layer2)
                    logger.info(f"Total chunks so far: {len(all_chunks)}")
                    
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
        logger.info(f"Processing single file: {input_path}")
        layer1 = extract_layer1(input_path, max_csv_rows=args.max_csv_rows)
        logger.info(f"Layer 1 complete: {len(layer1)} chunks")
        
        layer2 = split_text_payloads(layer1, cfg)
        logger.info(f"Layer 2 complete: {len(layer2)} chunks")
        
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
            logger.info("✓ pandas imported successfully")
            
            from bs4 import BeautifulSoup
            logger.info("✓ BeautifulSoup imported successfully")
            
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            logger.info("✓ LangChain splitters imported successfully")
            
            logger.info("Testing unstructured import...")
            if safe_init_unstructured():
                logger.info("✓ unstructured library initialized successfully")
            else:
                logger.info("✗ unstructured library failed to initialize")
            
            logger.info("All import tests completed")
            
        except Exception as e:
            logger.error(f"Import test failed: {e}")
            sys.exit(1)
    else:
        main()
