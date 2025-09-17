"""
Compatibility layer between agent system and existing pipeline
"""

import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from .pipeline_orchestrator import AgentPipelineOrchestrator
from .config import config

logger = logging.getLogger(__name__)

class AgentCompatibilityLayer:
    """Provides compatibility with existing pipeline interface"""
    
    def __init__(self, output_dir: str = "agent_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize orchestrator
        self.orchestrator = AgentPipelineOrchestrator({
            "output_dir": str(self.output_dir)
        })
        
        logger.info("Agent Compatibility Layer initialized")
    
    def run_chunker_only(self, input_path: str, output_file: str = None) -> Dict[str, Any]:
        """Run only the chunking agent (compatible with existing chunker.py)"""
        
        if output_file is None:
            output_file = self.output_dir / "chunks.jsonl"
        
        # Run chunking step
        result = self.orchestrator.run_step_by_step(
            input_path, 
            {"name": "chunker_only"}, 
            ["chunking"]
        )
        
        # Save chunks in compatible format
        chunks = result["chunking"]["chunks"]
        self._save_chunks_compatible(chunks, output_file)
        
        return {
            "chunks": chunks,
            "output_file": str(output_file),
            "stats": result["chunking"]["chunking_stats"]
        }
    
    def run_extractor_only(self, chunks_file: str, model_spec: Dict[str, Any], output_file: str = None) -> Dict[str, Any]:
        """Run only the extraction agent (compatible with existing extractor.py)"""
        
        if output_file is None:
            output_file = self.output_dir / "records.jsonl"
        
        # Load chunks
        chunks = self._load_chunks(chunks_file)
        
        # Run extraction step
        result = self.orchestrator.run_step_by_step(
            "dummy_path",  # Not used for extraction
            model_spec,
            ["extraction"]
        )
        
        # Save records in compatible format
        records = result["extraction"]["records"]
        self._save_records_compatible(records, output_file)
        
        return {
            "records": records,
            "output_file": str(output_file),
            "stats": result["extraction"]["extraction_stats"]
        }
    
    def run_validator_only(self, records_file: str, chunks_file: str, model_spec: Dict[str, Any], output_file: str = None) -> Dict[str, Any]:
        """Run only the validation agent (compatible with existing validator.py)"""
        
        if output_file is None:
            output_file = self.output_dir / "validation.jsonl"
        
        # Load records and chunks
        records = self._load_records(records_file)
        chunks = self._load_chunks(chunks_file)
        
        # Run validation step
        result = self.orchestrator.run_step_by_step(
            "dummy_path",  # Not used for validation
            model_spec,
            ["validation"]
        )
        
        # Save validation in compatible format
        validations = result["validation"]["validations"]
        self._save_validation_compatible(validations, output_file)
        
        return {
            "validations": validations,
            "output_file": str(output_file),
            "stats": result["validation"]["validation_stats"]
        }
    
    def run_full_pipeline(self, input_path: str, model_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Run complete pipeline (compatible with existing orchestrator.py)"""
        
        result = self.orchestrator.run_full_pipeline(input_path, model_spec)
        
        return {
            "chunks_file": str(self.output_dir / "chunks.jsonl"),
            "records_file": str(self.output_dir / "records.jsonl"),
            "validation_file": str(self.output_dir / "validation.jsonl"),
            "results": result
        }
    
    def _load_chunks(self, chunks_file: str) -> List[Dict[str, Any]]:
        """Load chunks from JSONL file"""
        chunks = []
        with open(chunks_file, 'r') as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        return chunks
    
    def _load_records(self, records_file: str) -> List[Dict[str, Any]]:
        """Load records from JSONL file"""
        records = []
        with open(records_file, 'r') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records
    
    def _save_chunks_compatible(self, chunks: List[Dict[str, Any]], output_file: str) -> None:
        """Save chunks in format compatible with existing pipeline"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    
    def _save_records_compatible(self, records: List[Dict[str, Any]], output_file: str) -> None:
        """Save records in format compatible with existing pipeline"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    def _save_validation_compatible(self, validations: List[Dict[str, Any]], output_file: str) -> None:
        """Save validation in format compatible with existing pipeline"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for validation in validations:
                f.write(json.dumps(validation, ensure_ascii=False) + "\n")

# CLI interface for compatibility
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Agent Compatibility Layer")
    parser.add_argument("--mode", choices=["chunker", "extractor", "validator", "full"], required=True,
                       help="Pipeline mode to run")
    parser.add_argument("--input", required=True, help="Input file or directory")
    parser.add_argument("--output", help="Output file")
    parser.add_argument("--model", help="Model specification")
    parser.add_argument("--chunks", help="Chunks file (for extractor/validator)")
    parser.add_argument("--records", help="Records file (for validator)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    
    # Initialize compatibility layer
    compat = AgentCompatibilityLayer()
    
    # Run based on mode
    if args.mode == "chunker":
        result = compat.run_chunker_only(args.input, args.output)
        print(f"✅ Chunking complete: {result['stats']['total_chunks']} chunks")
        
    elif args.mode == "extractor":
        if not args.chunks or not args.model:
            print("Error: --chunks and --model required for extractor mode")
            sys.exit(1)
        result = compat.run_extractor_only(args.chunks, {"name": args.model}, args.output)
        print(f"✅ Extraction complete: {result['stats']['total_records']} records")
        
    elif args.mode == "validator":
        if not args.records or not args.chunks or not args.model:
            print("Error: --records, --chunks, and --model required for validator mode")
            sys.exit(1)
        result = compat.run_validator_only(args.records, args.chunks, {"name": args.model}, args.output)
        print(f"✅ Validation complete: {result['stats']['total_validations']} validations")
        
    elif args.mode == "full":
        if not args.model:
            print("Error: --model required for full pipeline mode")
            sys.exit(1)
        result = compat.run_full_pipeline(args.input, {"name": args.model})
        print(f"✅ Full pipeline complete")
        print(f"📁 Chunks: {result['chunks_file']}")
        print(f"📁 Records: {result['records_file']}")
        print(f"📁 Validation: {result['validation_file']}")
