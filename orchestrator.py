#!/usr/bin/env python
"""
LLM Parser Orchestrator

This script orchestrates the complete LLM parsing pipeline:
1. Chunker: Creates chunks from input files
2. Extractor: Extracts structured data from chunks using LLM
3. Validator: Validates extracted records against chunks

Usage:
    python orchestrator.py --input input_files/ --dict data_dictionary.yaml --model deminimis_rules_stg
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('orchestrator.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    """Orchestrates the complete LLM parsing pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure required files exist
        self._validate_config()
    
    def _validate_config(self):
        """Validate that required files and directories exist"""
        required_files = [
            ('input_path', self.config['input_path']),
            ('data_dictionary', self.config['data_dictionary']),
        ]
        
        for name, path in required_files:
            if not Path(path).exists():
                raise FileNotFoundError(f"Required {name} not found: {path}")
        
        # Check if input is file or directory
        input_path = Path(self.config['input_path'])
        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")
        
        logger.info(f"Configuration validated successfully")
        logger.info(f"Input: {input_path}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def run_command(self, cmd: list, step_name: str) -> bool:
        """Run a command and handle errors"""
        logger.info(f"[START] Starting {step_name}...")
        logger.info(f"Command: {' '.join(cmd)}")
        
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,  # Allow real-time output to terminal
                text=True,
                cwd=Path.cwd()
            )
            duration = time.time() - start_time
            logger.info(f"[SUCCESS] {step_name} completed successfully in {duration:.2f}s")
            return True
            
        except subprocess.CalledProcessError as e:
            duration = time.time() - start_time
            logger.error(f"[FAILED] {step_name} failed after {duration:.2f}s")
            logger.error(f"Return code: {e.returncode}")
            # Note: stdout/stderr are not captured when capture_output=False
            return False
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"[FAILED] {step_name} failed after {duration:.2f}s: {e}")
            return False
    
    def run_chunker(self) -> bool:
        """Run the chunker to create chunks from input files"""
        cmd = [
            sys.executable, "chunker.py",
            self.config['input_path'],
            "--out", str(self.output_dir / "all_chunks.jsonl"),
            "--chunk-size", str(self.config.get('chunk_size', 1200)),
            "--chunk-overlap", str(self.config.get('chunk_overlap', 200))
        ]
        
        # Add optional chunker arguments
        if self.config.get('max_csv_rows'):
            cmd.extend(["--max-csv-rows", str(self.config['max_csv_rows'])])
        
        if self.config.get('skip_unstructured'):
            cmd.append("--skip-unstructured")
        
        if self.config.get('verbose'):
            cmd.append("--verbose")
        
        if self.config.get('test_single'):
            cmd.append("--test-single")
        
        return self.run_command(cmd, "Chunker")
    
    def run_extractor(self) -> bool:
        """Run the extractor to extract structured data from chunks"""
        cmd = [
            sys.executable, "extractor.py",
            "--dict", self.config['data_dictionary'],
            "--model", self.config['model'],
            "--chunks", str(self.output_dir / "all_chunks.jsonl"),
            "--out-records", str(self.output_dir / "records.jsonl"),
            "--llm", self.config.get('llm_model', 'gpt-4o-mini'),
            "--max-tokens-per-batch", str(self.config.get('max_tokens_per_batch', 100000))
        ]
        
        # Add optional extractor arguments
        if self.config.get('source_filter'):
            cmd.extend(["--source-filter", self.config['source_filter']])
        
        return self.run_command(cmd, "Extractor")
    
    def run_validator(self) -> bool:
        """Run the validator to validate extracted records"""
        cmd = [
            sys.executable, "validator.py",
            "--dict", self.config['data_dictionary'],
            "--model", self.config['model'],
            "--chunks", str(self.output_dir / "all_chunks.jsonl"),
            "--records", str(self.output_dir / "records.jsonl"),
            "--out", str(self.output_dir / "validation.jsonl"),
            "--out-summary", str(self.output_dir / "validation_summary.json"),
            "--llm", self.config.get('llm_model', 'gpt-4o-mini')
        ]
        
        # Add optional validator arguments
        if self.config.get('no_restrict_chunks_to_file'):
            cmd.append("--no-restrict-chunks-to-file")
        
        return self.run_command(cmd, "Validator")
    
    def run_pipeline(self) -> bool:
        """Run the complete pipeline"""
        logger.info("=" * 80)
        logger.info("[START] Starting LLM Parser Pipeline")
        logger.info("=" * 80)
        
        pipeline_start = time.time()
        
        # Step 1: Chunker
        if not self.run_chunker():
            logger.error("[FAILED] Pipeline failed at chunker step")
            return False
        
        # Step 2: Extractor
        if not self.run_extractor():
            logger.error("[FAILED] Pipeline failed at extractor step")
            return False
        
        # Step 3: Validator
        if not self.run_validator():
            logger.error("[FAILED] Pipeline failed at validator step")
            return False
        
        pipeline_duration = time.time() - pipeline_start
        
        logger.info("=" * 80)
        logger.info("[SUCCESS] Pipeline completed successfully!")
        logger.info(f"Total duration: {pipeline_duration:.2f}s")
        logger.info("=" * 80)
        
        # Print output file locations
        logger.info("[OUTPUT] Output files:")
        logger.info(f"   Chunks: {self.output_dir / 'all_chunks.jsonl'}")
        logger.info(f"   Records: {self.output_dir / 'records.jsonl'}")
        logger.info(f"   Validation: {self.output_dir / 'validation.jsonl'}")
        logger.info(f"   Summary: {self.output_dir / 'validation_summary.json'}")
        
        return True
    
    def print_summary(self):
        """Print a summary of the validation results"""
        summary_file = self.output_dir / "validation_summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                
                logger.info("=" * 80)
                logger.info("[SUMMARY] VALIDATION SUMMARY")
                logger.info("=" * 80)
                logger.info(f"Model: {summary.get('model', 'Unknown')}")
                logger.info(f"Total Records: {summary.get('total', 0)}")
                logger.info(f"Passed: {summary.get('passed', 0)}")
                logger.info(f"Failed: {summary.get('failed', 0)}")
                logger.info(f"Pass Rate: {summary.get('pass_rate', 0):.2%}")
                
                if summary.get('common_field_failures'):
                    logger.info("\nCommon Field Failures:")
                    for failure in summary['common_field_failures'][:5]:  # Top 5
                        logger.info(f"   {failure['field']}: {failure['count']} failures")
                
            except Exception as e:
                logger.warning(f"Could not read validation summary: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="LLM Parser Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python orchestrator.py --input input_files/ --dict data_dictionary.yaml --model deminimis_rules_stg
  
  # With custom output directory
  python orchestrator.py --input input_files/ --dict data_dictionary.yaml --model deminimis_rules_stg --output-dir my_output/
  
  # Test mode (single file)
  python orchestrator.py --input input_files/ --dict data_dictionary.yaml --model deminimis_rules_stg --test-single
  
  # Verbose mode
  python orchestrator.py --input input_files/ --dict data_dictionary.yaml --model deminimis_rules_stg --verbose
        """
    )
    
    # Required arguments
    parser.add_argument("--input", required=True, help="Input file or directory path")
    parser.add_argument("--dict", required=True, help="Path to data_dictionary.yaml")
    parser.add_argument("--model", required=True, help="Target model name from YAML")
    
    # Optional arguments
    parser.add_argument("--output-dir", default="output", help="Output directory (default: output)")
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="LLM model to use (default: gpt-4o-mini)")
    parser.add_argument("--source-filter", help="Override source path filter (comma-separated patterns)")
    parser.add_argument("--max-tokens-per-batch", type=int, default=100000, help="Maximum tokens per batch")
    
    # Chunker options
    parser.add_argument("--chunk-size", type=int, default=1200, help="Target chunk size (default: 1200)")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap (default: 200)")
    parser.add_argument("--max-csv-rows", type=int, help="Limit CSV rows for processing")
    parser.add_argument("--skip-unstructured", action="store_true", help="Skip unstructured processing")
    parser.add_argument("--test-single", action="store_true", help="Test with single file only")
    
    # Validator options
    parser.add_argument("--no-restrict-chunks-to-file", action="store_true", 
                       help="Don't restrict validation evidence to source file")
    
    # General options
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation step")
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Build configuration
    config = {
        'input_path': args.input,
        'data_dictionary': args.dict,
        'model': args.model,
        'output_dir': args.output_dir,
        'llm_model': args.llm_model,
        'source_filter': args.source_filter,
        'max_tokens_per_batch': args.max_tokens_per_batch,
        'chunk_size': args.chunk_size,
        'chunk_overlap': args.chunk_overlap,
        'max_csv_rows': args.max_csv_rows,
        'skip_unstructured': args.skip_unstructured,
        'test_single': args.test_single,
        'no_restrict_chunks_to_file': args.no_restrict_chunks_to_file,
        'verbose': args.verbose,
        'skip_validation': args.skip_validation
    }
    
    try:
        # Create orchestrator and run pipeline
        orchestrator = PipelineOrchestrator(config)
        
        if args.skip_validation:
            # Run only chunker and extractor
            logger.info("Skipping validation step as requested")
            if not orchestrator.run_chunker():
                sys.exit(1)
            if not orchestrator.run_extractor():
                sys.exit(1)
            logger.info("[SUCCESS] Chunker and Extractor completed successfully!")
        else:
            # Run complete pipeline
            if not orchestrator.run_pipeline():
                sys.exit(1)
            
            # Print summary
            orchestrator.print_summary()
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 