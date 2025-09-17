"""
Chunking Step - Format-based chunking as a pipeline step
"""

import json
import logging
import sqlite3
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import basic loaders only - use unstructured directly like chunker.py
from langchain.document_loaders import TextLoader

# Deferred unstructured import like chunker.py
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
        return False

logger = logging.getLogger(__name__)

class BaseChunkingStrategy:
    """Base class for format-based chunking strategies"""
    
    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk(self, document: Document) -> List[Document]:
        raise NotImplementedError

class PDFChunkingStrategy(BaseChunkingStrategy):
    """Chunking strategy for PDF files"""
    
    def chunk(self, document: Document) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        return splitter.split_documents([document])

class CSVChunkingStrategy(BaseChunkingStrategy):
    """Chunking strategy for CSV files"""
    
    def chunk(self, document: Document) -> List[Document]:
        # Check for pre-loaded DataFrame in metadata
        if 'dataframe' in document.metadata and isinstance(document.metadata['dataframe'], pd.DataFrame):
            logger.debug("Processing CSV from pre-loaded DataFrame.")
            df = document.metadata['dataframe']
            # We can remove the dataframe from metadata now as it is not serializable
            del document.metadata['dataframe']
            return self.chunk_from_dataframe(df, document)

        # If no DataFrame, it means the initial pandas load failed and we have raw text.
        # We can try to parse it again here, but it's likely to fail again.
        # The most robust fallback is to treat it as plain text.
        logger.warning(
            "No pre-loaded DataFrame found for CSV. This likely means the initial "
            "parsing with pandas failed. Falling back to text chunking."
        )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n", " ", ""]
        )
        return splitter.split_documents([document])

    def chunk_from_dataframe(self, df: pd.DataFrame, document: Document) -> List[Document]:
        """Chunk a DataFrame by rows"""
        rows_per_chunk = 100
        chunks = []
        for start_idx in range(0, len(df), rows_per_chunk):
            end_idx = min(start_idx + rows_per_chunk, len(df))
            chunk_df = df.iloc[start_idx:end_idx]
            
            # Convert to JSON for chunking
            chunk_text = chunk_df.to_json(orient='records', lines=True)
            
            chunk_doc = Document(
                page_content=chunk_text,
                metadata={
                    **document.metadata,
                    "chunk_type": "csv_rows",
                    "row_range": [start_idx, end_idx],
                    "total_rows": len(df)
                }
            )
            chunks.append(chunk_doc)
        
        return chunks

class HTMLChunkingStrategy(BaseChunkingStrategy):
    """Chunking strategy for HTML files"""
    
    def chunk(self, document: Document) -> List[Document]:
        # Use HTML-aware splitting
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["</p>", "</div>", "</section>", "\n\n", "\n", " ", ""]
        )
        return splitter.split_documents([document])

class SQLiteChunkingStrategy(BaseChunkingStrategy):
    """Chunking strategy for SQLite database files"""
    
    def chunk(self, document: Document) -> List[Document]:
        # This is a placeholder - SQLite chunking would need the actual file path
        # For now, treat as text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n", " ", ""]
        )
        return splitter.split_documents([document])
    
    def chunk_from_file(self, file_path: str) -> List[Document]:
        """Chunk SQLite database by querying tables"""
        chunks = []
        
        try:
            conn = sqlite3.connect(file_path)
            cursor = conn.cursor()
            
            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            for table_name, in tables:
                # Get table schema
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                
                # Get all data from table
                cursor.execute(f"SELECT * FROM {table_name}")
                rows = cursor.fetchall()
                
                # Chunk by rows (e.g., 100 rows per chunk)
                rows_per_chunk = 100
                for start_idx in range(0, len(rows), rows_per_chunk):
                    end_idx = min(start_idx + rows_per_chunk, len(rows))
                    chunk_rows = rows[start_idx:end_idx]
                    
                    # Convert to JSON
                    chunk_data = {
                        "table_name": table_name,
                        "columns": [col[1] for col in columns],
                        "rows": chunk_rows,
                        "row_range": [start_idx, end_idx]
                    }
                    
                    chunk_text = json.dumps(chunk_data, indent=2)
                    
                    chunk_doc = Document(
                        page_content=chunk_text,
                        metadata={
                            "chunk_type": "sqlite_table",
                            "table_name": table_name,
                            "row_range": [start_idx, end_idx],
                            "total_rows": len(rows)
                        }
                    )
                    chunks.append(chunk_doc)
            
            conn.close()
            return chunks
            
        except Exception as e:
            logger.error(f"SQLite processing failed: {e}")
            return []

class XLSXChunkingStrategy(BaseChunkingStrategy):
    """Chunking strategy for Excel files"""
    
    def chunk(self, document: Document) -> List[Document]:
        # This is a placeholder - XLSX chunking would need the actual file path
        # For now, treat as text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n", " ", ""]
        )
        return splitter.split_documents([document])
    
    def chunk_from_file(self, file_path: str) -> List[Document]:
        """Chunk Excel file by sheets and rows"""
        chunks = []
        
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Chunk by rows (e.g., 100 rows per chunk)
                rows_per_chunk = 100
                for start_idx in range(0, len(df), rows_per_chunk):
                    end_idx = min(start_idx + rows_per_chunk, len(df))
                    chunk_df = df.iloc[start_idx:end_idx]
                    
                    # Convert to JSON
                    chunk_text = chunk_df.to_json(orient='records', lines=True)
                    
                    chunk_doc = Document(
                        page_content=chunk_text,
                        metadata={
                            "chunk_type": "xlsx_sheet",
                            "sheet_name": sheet_name,
                            "row_range": [start_idx, end_idx],
                            "total_rows": len(df)
                        }
                    )
                    chunks.append(chunk_doc)
            
            return chunks
            
        except Exception as e:
            logger.error(f"XLSX processing failed: {e}")
            return []

class TextChunkingStrategy(BaseChunkingStrategy):
    """Chunking strategy for text files"""
    
    def chunk(self, document: Document) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        return splitter.split_documents([document])

class ChunkingStep:
    """Chunking step for the data extraction pipeline"""
    
    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 200, max_tokens: int = 100000, no_limit: bool = False, max_chunks: Optional[int] = None):
        # Format-based strategies
        self.strategies = {
            "pdf": PDFChunkingStrategy(),
            "csv": CSVChunkingStrategy(),
            "html": HTMLChunkingStrategy(),
            "sqlite": SQLiteChunkingStrategy(),
            "xlsx": XLSXChunkingStrategy(),
            "txt": TextChunkingStrategy()
        }
        
        # Chunking configuration
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_tokens = max_tokens
        self.no_limit = no_limit
        self.max_chunks = max_chunks
    
    def process_file(self, file_path: str, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> Dict[str, Any]:
        """Process a single file and return chunks"""
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap
        
        try:
            # Determine file format
            file_format = self._get_file_format(file_path)
            
            # Select strategy
            strategy = self.strategies.get(file_format, self.strategies["txt"])
            strategy.chunk_size = chunk_size
            strategy.chunk_overlap = chunk_overlap
            
            # Load and chunk document
            if file_format in ["sqlite", "xlsx"]:
                # Special handling for database and Excel files
                chunks = self._chunk_special_format(file_path, file_format, strategy)
            else:
                # Use standard loading with unstructured library
                document = self._load_document(file_path)
                chunks = strategy.chunk(document)
            
            # Apply max_chunks limit if specified
            if self.max_chunks is not None and len(chunks) > self.max_chunks:
                logger.info(f"Limiting chunks from {len(chunks)} to {self.max_chunks} for {file_path}")
                chunks = chunks[:self.max_chunks]
            
            # Add metadata to chunks
            final_chunks = self._add_chunk_metadata(chunks, file_path, file_format)
            
            logger.info(f"Chunking complete: {len(final_chunks)} chunks from {file_path} ({file_format})")
            
            return {
                "chunks": final_chunks,
                "file_path": file_path,
                "file_format": file_format,
                "chunking_stats": {
                    "total_chunks": len(final_chunks),
                    "format_used": file_format,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "max_chunks_limit": self.max_chunks
                }
            }
            
        except Exception as e:
            logger.error(f"Chunking failed for {file_path}: {e}")
            raise
    
    def process_directory(self, directory_path: str, output_file: str, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> Dict[str, Any]:
        """Process all files in a directory and save chunks"""
        directory = Path(directory_path)
        all_chunks = []
        processed_files = []
        
        # Find all supported files
        supported_extensions = {'.pdf', '.csv', '.html', '.htm', '.sqlite', '.db', '.xlsx', '.xls', '.txt', '.md'}
        files = [f for f in directory.rglob("*") if f.is_file() and f.suffix.lower() in supported_extensions]
        
        logger.info(f"Found {len(files)} files to process in {directory_path}")
        
        for file_path in files:
            try:
                result = self.process_file(str(file_path), chunk_size, chunk_overlap)
                all_chunks.extend(result["chunks"])
                processed_files.append({
                    "file_path": str(file_path),
                    "format": result["file_format"],
                    "chunks": len(result["chunks"])
                })
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                # Add failed file to processed_files with 0 chunks
                processed_files.append({
                    "file_path": str(file_path),
                    "format": "failed",
                    "chunks": 0
                })
                continue
        
        # Save all chunks
        self.save_chunks(all_chunks, output_file)
        
        return {
            "total_chunks": len(all_chunks),
            "processed_files": processed_files,
            "output_file": output_file
        }
    
    def _get_file_format(self, file_path: str) -> str:
        """Determine file format from extension"""
        ext = Path(file_path).suffix.lower()
        
        format_map = {
            ".pdf": "pdf",
            ".csv": "csv",
            ".html": "html",
            ".htm": "html",
            ".sqlite": "sqlite",
            ".db": "sqlite",
            ".xlsx": "xlsx",
            ".xls": "xlsx",
            ".txt": "txt",
            ".md": "txt"
        }
        
        return format_map.get(ext, "txt")
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)"""
        # Simple estimation: ~4 characters per token
        return len(text) // 4
    
    def _limit_content_by_tokens(self, content: str, max_tokens: int, no_limit: bool = False) -> str:
        """Limit content to maximum token count"""
        if not content or no_limit:
            return content
        
        estimated_tokens = self._estimate_tokens(content)
        
        if estimated_tokens <= max_tokens:
            return content
        
        # Calculate approximate character limit
        char_limit = int(max_tokens * 4 * 0.9)  # 90% of estimated limit for safety
        
        if len(content) <= char_limit:
            return content
        
        # Truncate content and add truncation notice
        truncated_content = content[:char_limit]
        truncated_content += f"\n\n[CONTENT TRUNCATED - Original length: {len(content)} chars, estimated tokens: {estimated_tokens}]"
        
        logger.warning(f"Content truncated to {char_limit} characters (estimated {max_tokens} tokens)")
        return truncated_content
    
    def _load_document(self, file_path: str) -> Document:
        """Load document using unstructured library like chunker.py"""
        try:
            file_format = self._get_file_format(file_path)
            logger.info(f"Loading {file_format} file: {file_path}")
            
            if file_format == "csv":
                # For CSV, we'll load with pandas and pass the DataFrame
                try:
                    df = pd.read_csv(file_path)
                    content = df.to_string() # Provide a string representation for context
                    return Document(
                        page_content=content,
                        metadata={
                            "source_path": file_path,
                            "file_format": file_format,
                            "dataframe": df
                        }
                    )
                except Exception as csv_error:
                    logger.warning(f"CSV parsing with pandas failed, falling back to text loading: {csv_error}")
                    # Fallback to text loading for problematic CSV files
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    return Document(page_content=content, metadata={"source_path": file_path, "file_format": file_format})
            
            elif file_format == "txt":
                # Use TextLoader for text files
                loader = TextLoader(file_path)
                documents = loader.load()
                if len(documents) > 1:
                    combined_content = "\n\n".join([doc.page_content for doc in documents])
                    combined_metadata = documents[0].metadata.copy()
                    return Document(page_content=combined_content, metadata=combined_metadata)
                return documents[0]
            
            else:
                # Use unstructured library for other formats (like chunker.py)
                if not init_unstructured():
                    raise Exception("Unstructured library not available")
                
                # Use the partition function from unstructured
                elements = partition(file_path)
                
                # Convert elements to text content
                content_parts = []
                for element in elements:
                    if hasattr(element, 'text') and element.text:
                        content_parts.append(element.text)
                
                content = "\n\n".join(content_parts)
                
                # Apply token limit (if not disabled)
                content = self._limit_content_by_tokens(content, self.max_tokens, getattr(self, 'no_limit', False))
                
                return Document(
                    page_content=content,
                    metadata={"source_path": file_path, "file_format": file_format}
                )
            
        except Exception as e:
            logger.error(f"Failed to load document {file_path}: {e}")
            raise
    
    def _chunk_special_format(self, file_path: str, file_format: str, strategy: BaseChunkingStrategy) -> List[Document]:
        """Handle special formats that need file path access"""
        if file_format == "sqlite":
            return strategy.chunk_from_file(file_path)
        elif file_format == "xlsx":
            return strategy.chunk_from_file(file_path)
        else:
            # Fallback to standard processing
            document = self._load_document(file_path)
            return strategy.chunk(document)
    
    def _add_chunk_metadata(self, chunks: List[Document], file_path: str, file_format: str) -> List[Document]:
        """Add metadata to chunks"""
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": f"{Path(file_path).stem}_chunk_{i}",
                "source_path": file_path,
                "file_format": file_format,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "tokens_est": len(chunk.page_content) // 4
            })
        
        return chunks
    
    def save_chunks(self, chunks: List[Document], output_file: str) -> None:
        """Save chunks to JSONL file"""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                chunk_data = {
                    "chunk_id": chunk.metadata.get("chunk_id"),
                    "text": chunk.page_content,
                    "source_path": chunk.metadata.get("source_path"),
                    "filetype": Path(chunk.metadata.get("source_path", "")).suffix[1:],
                    "file_format": chunk.metadata.get("file_format"),
                    "tokens_est": chunk.metadata.get("tokens_est"),
                    "fingerprint": f"sha1:{hash(chunk.page_content) % 100000000:08x}",
                    "created_at": __import__('datetime').datetime.now().isoformat() + "Z",
                    "metadata": chunk.metadata
                }
                f.write(json.dumps(chunk_data, ensure_ascii=False) + "\n")
        
        logger.info(f"Saved {len(chunks)} chunks to {output_file}")

# CLI interface for standalone usage
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Chunking Step - Process documents into chunks for LLM processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single file
  python -m data_extraction_agent.chunking_step ca_input_files/canada_tax_rates_html.html --output ca_output/all_chunks.jsonl
  
  # Process entire directory
  python -m data_extraction_agent.chunking_step ca_input_files --output ca_output/all_chunks.jsonl
  
  # Process with custom chunk settings
  python -m data_extraction_agent.chunking_step ca_input_files --chunk-size 800 --chunk-overlap 100 --verbose
  
  # Process with token limit
  python -m data_extraction_agent.chunking_step ca_input_files --max-tokens 50000 --verbose
  
  # Process with max chunks limit (e.g., only 5 chunks per file)
  python -m data_extraction_agent.chunking_step ca_input_files --max-chunks 5 --verbose
  
  # Process all documents fully (no truncation)
  python -m data_extraction_agent.chunking_step ca_input_files --no-limit --verbose
  
  # Process with all custom settings
  python -m data_extraction_agent.chunking_step ca_input_files --chunk-size 1000 --chunk-overlap 150 --max-tokens 75000 --max-chunks 10 --verbose
        """
    )
    
    parser.add_argument("input", help="Input file or directory to process")
    parser.add_argument("--output", default="output/chunks.jsonl", help="Output JSONL file for chunks")
    parser.add_argument("--chunk-size", type=int, default=1200, help="Chunk size in characters (default: 1200)")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap in characters (default: 200)")
    parser.add_argument("--max-tokens", type=int, default=100000, help="Maximum tokens per file to process (default: 100000)")
    parser.add_argument("--max-chunks", type=int, help="Maximum number of chunks per file (default: no limit)")
    parser.add_argument("--no-limit", action="store_true", help="Process all documents fully without token limits")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Set logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level)
    if args.verbose:
        log_level = logging.DEBUG
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    try:
        # Validate input
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input path does not exist: {input_path}")
            sys.exit(1)
        
        # Initialize chunking step
        logger.info("Initializing Chunking Step...")
        chunking_step = ChunkingStep(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, max_tokens=args.max_tokens, no_limit=args.no_limit, max_chunks=args.max_chunks)
        
        print(f"\n🔧 Chunking Configuration:")
        print(f"   Chunk Size: {args.chunk_size} characters")
        print(f"   Chunk Overlap: {args.chunk_overlap} characters")
        if args.max_chunks:
            print(f"   Max Chunks per File: {args.max_chunks}")
        else:
            print(f"   Max Chunks per File: No limit")
        if args.no_limit:
            print(f"   Token Limit: DISABLED (processing all documents fully)")
        else:
            print(f"   Max Tokens: {args.max_tokens}")
        print(f"   Output File: {args.output}")
        print("="*60)
        
        # Process input
        if input_path.is_file():
            logger.info(f"Processing single file: {input_path}")
            result = chunking_step.process_file(str(input_path))
            chunking_step.save_chunks(result["chunks"], args.output)
            
            print(f"\n✅ Single File Chunking Complete!")
            print(f"📁 File: {input_path.name}")
            print(f"📊 Format: {result['chunking_stats']['format_used']}")
            print(f"🔢 Total Chunks: {result['chunking_stats']['total_chunks']}")
            print(f"💾 Output: {args.output}")
            
        else:
            logger.info(f"Processing directory: {input_path}")
            result = chunking_step.process_directory(str(input_path), args.output)
            
            print(f"\n✅ Directory Chunking Complete!")
            print(f"📁 Directory: {input_path}")
            print(f"📄 Files Processed: {len(result['processed_files'])}")
            print(f"🔢 Total Chunks: {result['total_chunks']}")
            print(f"💾 Output: {args.output}")
            
            if result['processed_files']:
                print(f"\n📋 Processed Files:")
                for file_info in result['processed_files']:
                    filename = Path(file_info['file_path']).name
                    print(f"   • {filename} ({file_info['chunks']} chunks)")
        
        print("="*60)
        print("🎉 Chunking completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Chunking interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Chunking failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)