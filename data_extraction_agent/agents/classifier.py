"""
Batch Classifier Agent for processing all_chunks.jsonl files
Creates mappings.json with file-to-table mappings
"""

import json
import logging
import argparse
import sys
import tiktoken
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from langchain.schema import BaseMessage, HumanMessage, SystemMessage

from .base import BaseAgent
from ..config import config
from ..prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)

# Set up file logging
def setup_file_logging():
    """Set up file logging for classifier agent"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"classifier_agent_{timestamp}.log"
    
    # Configure file handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    # Remove any existing console handlers to prevent terminal logging
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
    
    # Set logger level to INFO for file logging
    logger.setLevel(logging.INFO)
    
    return log_file

class ClassifierAgent(BaseAgent):
    """Batch classifier agent for processing all_chunks.jsonl files"""
    
    def __init__(self, agent_config: Optional[Dict[str, Any]] = None):
        super().__init__("classifier_agent", agent_config)
        
        # Initialize prompt builder
        self.prompt_builder = PromptBuilder()
        
        # Token limits are now managed by the agent's config property
        self.max_tokens_per_prompt = self.config.get("max_tokens", 100000)
        self._tokenizer = None
        
        # Set up file logging
        self.log_file = setup_file_logging()
        logger.info("Classifier Agent initialized")
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process inputs and return results - required by BaseAgent"""
        chunks_file_path = inputs.get("chunks_file_path")
        if not chunks_file_path:
            raise ValueError("chunks_file_path is required in inputs")
        
        return self.process_all_chunks(chunks_file_path)
    
    def process_all_chunks(self, chunks_file_path: str) -> Dict[str, Any]:
        """Process all chunks from all_chunks.jsonl file"""
        
        try:
            logger.info("=" * 80)
            logger.info("STARTING CLASSIFICATION PROCESS")
            logger.info("=" * 80)
            logger.info(f"Chunks file: {chunks_file_path}")
            logger.info(f"Max tokens per prompt: {self.max_tokens_per_prompt}")
            
            # Read all chunks
            chunks = self._read_chunks_file(chunks_file_path)
            logger.info(f"Loaded {len(chunks)} chunks from {chunks_file_path}")
            
            # Group chunks by filename
            file_chunks = self._group_chunks_by_file(chunks)
            logger.info(f"Found {len(file_chunks)} unique files")
            
            # Initialize mappings
            mappings = {"mappings": []}
            
            # Process each file
            for filename, file_chunk_list in file_chunks.items():
                logger.info(f"Processing file: {filename}")
                
                try:
                    # Process file chunks
                    file_mappings = self._process_file_chunks(filename, file_chunk_list)
                    
                    # Add to mappings
                    if file_mappings:
                        mappings["mappings"].extend(file_mappings)
                        logger.info(f"Added {len(file_mappings)} mappings for {filename}")
                    
                except Exception as e:
                    logger.error(f"Failed to process file {filename}: {e}")
                    # Continue with next file as per requirement
            
            # Save mappings.json
            mappings_file = Path(chunks_file_path).parent / "mappings.json"
            self._save_mappings(mappings, mappings_file)
            
            logger.info("=" * 80)
            logger.info("CLASSIFICATION PROCESS COMPLETED")
            logger.info("=" * 80)
            logger.info(f"Files processed: {len(file_chunks)}")
            logger.info(f"Total mappings: {len(mappings['mappings'])}")
            logger.info(f"Mappings file: {mappings_file}")
            logger.info(f"Log file: {self.log_file}")
            
            return {
                "status": "success",
                "files_processed": len(file_chunks),
                "total_mappings": len(mappings["mappings"]),
                "mappings_file": str(mappings_file)
            }
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise
    
    def _read_chunks_file(self, chunks_file_path: str) -> List[Dict[str, Any]]:
        """Read chunks from all_chunks.jsonl file"""
        chunks = []
        
        with open(chunks_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    chunk = json.loads(line.strip())
                    chunks.append(chunk)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num}: {e}")
                    continue
        
        return chunks
    
    def _group_chunks_by_file(self, chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group chunks by filename"""
        file_chunks = {}
        
        for chunk in chunks:
            filename = chunk.get("source_path", "unknown")
            if filename not in file_chunks:
                file_chunks[filename] = []
            file_chunks[filename].append(chunk)
        
        return file_chunks
    
    def _get_tokenizer(self):
        """Get the tokenizer for the current model."""
        if self._tokenizer is None:
            try:
                # Encoding for gpt-4o and similar models
                self._tokenizer = tiktoken.get_encoding("o200k_base")
            except ValueError:
                # Fallback for other models like gpt-4
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
        return self._tokenizer

    def _estimate_tokens(self, data: Any) -> int:
        """Roughly estimate the number of tokens for a given data structure."""
        tokenizer = self._get_tokenizer()
        if not isinstance(data, str):
            text = json.dumps(data, ensure_ascii=False)
        else:
            text = data
        return len(tokenizer.encode(text))

    def _batch_chunks_by_tokens(self, chunks: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Batches chunks to not exceed token limits for the classifier prompt."""
        
        # Estimate the base token count for the prompt template (instructions, etc.)
        base_prompt_tokens = 4000  # A more conservative buffer for the prompt's boilerplate
        
        batches = []
        current_batch = []
        current_tokens = base_prompt_tokens

        for chunk in chunks:
            chunk_tokens = self._estimate_tokens(chunk.get("text", ""))
            
            # If a single chunk is too large, it needs to be handled.
            # For now, we'll log a warning and skip it.
            if chunk_tokens + base_prompt_tokens > self.max_tokens_per_prompt:
                logger.warning(
                    f"Chunk {chunk.get('chunk_id', 'N/A')} is too large ({chunk_tokens} tokens) "
                    f"to fit within the token limit of {self.max_tokens_per_prompt}. Skipping this chunk."
                )
                continue

            if current_batch and (current_tokens + chunk_tokens > self.max_tokens_per_prompt):
                batches.append(current_batch)
                current_batch = []
                current_tokens = base_prompt_tokens

            current_batch.append(chunk)
            current_tokens += chunk_tokens

        if current_batch:
            batches.append(current_batch)
            
        logger.info(f"Created {len(batches)} batches from {len(chunks)} chunks to respect token limits.")
        return batches

    def _process_file_chunks(self, filename: str, file_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process chunks for a single file, with batching to respect token limits."""
        
        logger.info(f"Starting classification for file: {filename}")
        logger.info(f"Total chunks to process: {len(file_chunks)}")
        
        # Batch chunks to respect token limits
        chunk_batches = self._batch_chunks_by_tokens(file_chunks)
        
        all_mappings = []
        
        total_batches = len(chunk_batches)
        for i, batch in enumerate(chunk_batches):
            batch_num = i + 1
            logger.info(f"Processing batch {batch_num}/{total_batches} for file {filename}...")
            
            # Create prompt using prompt builder for the current batch
            logger.info("Building classification prompt for batch...")
            prompt = self.prompt_builder.build_mapper_prompt(batch)
            
            # Log the prompt for debugging
            prompt_tokens = self._estimate_tokens(prompt)
            logger.info(f"=== CLASSIFICATION PROMPT FOR FILE {filename} (BATCH {batch_num}) ===")
            logger.info(f"Estimated prompt tokens: {prompt_tokens}")
            
            # Log complete prompt content
            logger.info(f"Prompt content:\n{prompt}")
            
            logger.info("=== END PROMPT ===")
            
            if prompt_tokens > self.max_tokens_per_prompt:
                logger.warning(
                    f"Prompt for batch {batch_num} of {filename} exceeds token limit "
                    f"({prompt_tokens} > {self.max_tokens_per_prompt}). "
                    "Skipping this batch to avoid API errors."
                )
                continue

            # Send to LLM
            logger.info(f"Sending batch {batch_num} to LLM...")
            response = self._send_to_llm(prompt)
            
            # Parse response and create mappings
            logger.info(f"Parsing LLM response for batch {batch_num}...")
            mappings = self._parse_llm_response(response, filename)
            
            if mappings:
                all_mappings.extend(mappings)
                logger.info(f"Successfully classified {len(mappings)} mappings in batch {batch_num}.")
        
        # In a multi-batch scenario, we might have duplicate mappings.
        # Here we'll merge them, prioritizing higher confidence scores.
        merged_mappings = {}
        for mapping in all_mappings:
            model_name = mapping["model"]
            if model_name not in merged_mappings:
                merged_mappings[model_name] = mapping
            else:
                # If the new mapping has a higher confidence, replace the old one
                if mapping["confidence_score"] > merged_mappings[model_name]["confidence_score"]:
                    merged_mappings[model_name] = mapping

        logger.info(f"Completed classification for file {filename}: {len(merged_mappings)} unique mappings")
        return list(merged_mappings.values())
    
    def _send_to_llm(self, prompt: str) -> str:
        """Send prompt to LLM and get response"""
        try:
            # Create messages
            messages = [
                SystemMessage(content="You are a Senior Customs Data Analysis Specialist."),
                HumanMessage(content=prompt)
            ]
            
            # Get LLM response using base class method
            response = self.invoke_llm(messages)
            
            # Log the response for debugging
            logger.info(f"LLM response received:")
            logger.info(f"Response length: {len(response)} characters")
            
            # Log complete response content
            logger.info(f"Complete response:\n{response}")
            
            return response
                
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            raise
    
    def _parse_llm_response(self, response: str, filename: str) -> List[Dict[str, Any]]:
        """Parse LLM response and create mappings"""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                logger.warning(f"No JSON found in response for {filename}")
                return []
            
            json_str = response[json_start:json_end]
            parsed_response = json.loads(json_str)
            
            # Extract table mappings
            table_mappings = parsed_response.get("table_mappings", [])
            
            # Create mappings in required format
            mappings = []
            for mapping in table_mappings:
                table_name = mapping.get("table_name", "")
                confidence = mapping.get("confidence_score", 0.0)
                
                if table_name:
                    mappings.append({
                        "model": table_name,
                        "filename": [filename],
                        "confidence_score": confidence
                    })
            
            return mappings
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response for {filename}: {e}")
            return []
    
    def _save_mappings(self, mappings: Dict[str, Any], mappings_file: Path) -> None:
        """Save mappings to JSON file"""
        try:
            with open(mappings_file, 'w', encoding='utf-8') as f:
                json.dump(mappings, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Mappings saved to {mappings_file}")
            
        except Exception as e:
            logger.error(f"Failed to save mappings: {e}")
            raise


def main():
    """Main function for standalone execution"""
    parser = argparse.ArgumentParser(
        description="Classifier Agent - Map document chunks to database tables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process chunks from ca_output directory
  python -m data_extraction_agent.agents.classifier --chunks-file ca_output/all_chunks.jsonl
  
  # Process with custom output directory
  python -m data_extraction_agent.agents.classifier --chunks-file ca_output/all_chunks.jsonl --output-dir custom_output
  
  # Process with verbose logging
  python -m data_extraction_agent.agents.classifier --chunks-file ca_output/all_chunks.jsonl --verbose
        """
    )
    
    parser.add_argument(
        "--chunks-file",
        required=True,
        help="Path to all_chunks.jsonl file to process"
    )
    
    parser.add_argument(
        "--output-dir",
        help="Output directory for mappings.json (default: same as chunks file directory)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging to file only
    log_level = getattr(logging, args.log_level)
    if args.verbose:
        log_level = logging.DEBUG
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Setup file handler only (timestamped)
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"classifier_agent_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # Configure root logger with file handler only
    logging.basicConfig(
        level=log_level,
        handlers=[file_handler]
    )
    
    # Validate input file
    chunks_file_path = Path(args.chunks_file)
    if not chunks_file_path.exists():
        logger.error(f"Chunks file not found: {chunks_file_path}")
        sys.exit(1)
    
    if not chunks_file_path.suffix == '.jsonl':
        logger.warning(f"File doesn't have .jsonl extension: {chunks_file_path}")
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        mappings_file = output_dir / "mappings.json"
    else:
        mappings_file = chunks_file_path.parent / "mappings.json"
    
    try:
        # Initialize classifier agent
        logger.info("Initializing Classifier Agent...")
        classifier = ClassifierAgent()
        
        # Process chunks
        logger.info(f"Processing chunks from: {chunks_file_path}")
        result = classifier.process_all_chunks(str(chunks_file_path))
        
        # Move mappings to specified output directory if different
        if args.output_dir:
            temp_mappings_file = chunks_file_path.parent / "mappings.json"
            if temp_mappings_file.exists():
                temp_mappings_file.rename(mappings_file)
                logger.info(f"Moved mappings to: {mappings_file}")
        
        # Print results
        print("\n" + "="*60)
        print("CLASSIFICATION RESULTS")
        print("="*60)
        print(f"Status: {result['status']}")
        print(f"Files processed: {result['files_processed']}")
        print(f"Total mappings: {result['total_mappings']}")
        print(f"Mappings file: {result['mappings_file']}")
        print("="*60)
        
        # Print agent statistics
        stats = classifier.get_stats()
        print(f"\nAgent Statistics:")
        print(f"  Success count: {stats['success_count']}")
        print(f"  Error count: {stats['error_count']}")
        print(f"  Success rate: {stats['success_rate']:.2%}")
        
        if result['status'] == 'success':
            print(f"\n✅ Classification completed successfully!")
            print(f"📁 Mappings saved to: {mappings_file}")
            print(f"📝 Logs saved to: {log_file}")
        else:
            print(f"\n❌ Classification failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Classification interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()