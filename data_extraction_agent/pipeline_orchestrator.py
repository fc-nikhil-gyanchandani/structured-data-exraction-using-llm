"""
Agent-based Pipeline Orchestrator
Compatible with existing pipeline while adding intelligent agent capabilities
"""

import json
import logging
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

from .agents.classifier import ClassifierAgent
from .agents.extraction_agent import ExtractionAgent
from .agents.validation_agent import ValidationAgent
from .config import AgentConfig

logger = logging.getLogger(__name__)

class AgentPipelineOrchestrator:
    """Orchestrates the complete agent-based pipeline"""
    
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        self.config = config_override or {}
        self.output_dir = Path(self.config.get('output_dir', 'agent_output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get country from config
        self.country = self.config.get('country', 'CA')
        
        # Initialize config with country
        self.agent_config = AgentConfig(country=self.country)
        
        # Apply overrides from orchestrator config to agent configs
        if "max_tokens" in self.config:
            for agent_name in ["classifier", "extractor", "validator"]:
                agent_cfg = self.agent_config.get_agent_config(agent_name)
                if agent_cfg:
                    agent_cfg["max_tokens"] = self.config["max_tokens"]

        # Initialize agents with country-specific config
        self.classifier_agent = ClassifierAgent(self.agent_config.get_agent_config("classifier"))
        self.extractor_agent = ExtractionAgent(self.agent_config.get_agent_config("extractor"))
        self.validator_agent = ValidationAgent(self.agent_config.get_agent_config("validator"))
        
        # Pipeline state
        self.pipeline_state = {}
        self.processing_stats = {
            "total_files": 0,
            "successful_files": 0,
            "failed_files": 0,
            "total_chunks": 0,
            "total_records": 0,
            "total_validations": 0
        }
        
        logger.info("Agent Pipeline Orchestrator initialized")
    
    def run_full_pipeline(self, 
                         input_path: str, 
                         model_spec: Dict[str, Any],
                         output_prefix: str = "agent") -> Dict[str, Any]:
        """Run the complete agent-based pipeline"""
        
        logger.info("=" * 80)
        logger.info("Starting Agent-Based Pipeline")
        logger.info("=" * 80)
        
        pipeline_start = time.time()
        
        try:
            # Step 1: Chunking
            logger.info("🔄 Step 1: Intelligent Chunking")
            chunking_result = self._run_chunking(input_path)
            
            # Step 2: Classification
            logger.info("🔄 Step 2: Document Classification")
            classification_result = self._run_classification(chunking_result["chunks"])
            
            # Step 3: Extraction
            logger.info("🔄 Step 3: Intelligent Extraction")
            extraction_result = self._run_extraction(
                chunking_result["chunks"], 
                classification_result,
                model_spec
            )
            
            # Step 4: Validation
            logger.info("🔄 Step 4: Agent-Based Validation")
            validation_result = self._run_validation(
                extraction_result["records"],
                chunking_result["chunks"],
                model_spec
            )
            
            # Compile results
            pipeline_duration = time.time() - pipeline_start
            
            results = {
                "pipeline_stats": {
                    "duration_seconds": pipeline_duration,
                    "chunking_stats": chunking_result["chunking_stats"],
                    "classification_stats": classification_result["classification_stats"],
                    "extraction_stats": extraction_result.get("extraction_stats", {}),
                    "validation_stats": validation_result.get("validation_stats", {}),
                    "overall_success": True
                },
                "chunks": chunking_result["chunks"],
                "classification": classification_result,
                "records": extraction_result["records"],
                "validation": validation_result,
                "agent_stats": self._get_agent_stats()
            }
            
            # Save results
            self._save_pipeline_results(results, output_prefix)
            
            logger.info("=" * 80)
            logger.info("✅ Agent Pipeline Completed Successfully!")
            logger.info(f"⏱️  Total Duration: {pipeline_duration:.2f}s")
            logger.info("=" * 80)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise
    
    def run_step_by_step(self, 
                        input_path: str, 
                        model_spec: Dict[str, Any],
                        steps: List[str] = None) -> Dict[str, Any]:
        """Run pipeline steps individually"""
        
        if steps is None:
            steps = ["chunking", "classification", "extraction", "validation"]
        
        results = {}
        
        for step in steps:
            logger.info(f"🔄 Running step: {step}")
            
            if step == "chunking":
                results["chunking"] = self._run_chunking(input_path)
                
            elif step == "classification":
                if "chunking" not in results:
                    raise ValueError("Classification requires chunking to be run first")
                results["classification"] = self._run_classification(results["chunking"]["chunks"])
                
            elif step == "extraction":
                if "chunking" not in results:
                    raise ValueError("Extraction requires chunking to be run first")
                classification = results.get("classification", {})
                results["extraction"] = self._run_extraction(
                    results["chunking"]["chunks"], 
                    classification,
                    model_spec
                )
                
            elif step == "validation":
                if "extraction" not in results:
                    raise ValueError("Validation requires extraction to be run first")
                results["validation"] = self._run_validation(
                    results["extraction"]["records"],
                    results["chunking"]["chunks"],
                    model_spec
                )
            
            logger.info(f"✅ Step {step} completed")
        
        return results
    
    def run_validation_only(self, records_file: str, chunks_file: str, model_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Run validation only on existing records and chunks files"""
        
        logger.info("=" * 80)
        logger.info("Running Validation Only")
        logger.info("=" * 80)
        
        validation_start = time.time()
        
        try:
            # Run validation using the new file-based method
            validation_summary = self.validator_agent.validate_records_file(
                records_file=records_file,
                chunks_file=chunks_file,
                output_dir=str(self.output_dir)
            )
            
            # Load validation results
            validation_file = self.output_dir / "validation.jsonl"
            validation_results = []
            if validation_file.exists():
                with open(validation_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            validation_results.append(json.loads(line))
            
            validation_duration = time.time() - validation_start
            
            results = {
                "validation_summary": validation_summary,
                "validation_results": validation_results,
                "validation_stats": {
                    "duration_seconds": validation_duration,
                    "total_validations": len(validation_results),
                    "validation_rate": validation_summary.get('overall', {}).get('validation_rate', 0),
                    "valid_records": validation_summary.get('overall', {}).get('valid_records', 0),
                    "invalid_records": validation_summary.get('overall', {}).get('invalid_records', 0),
                    "missing_records": validation_summary.get('overall', {}).get('missing_records', 0),
                    "partial_records": validation_summary.get('overall', {}).get('partial_records', 0)
                }
            }
            
            logger.info("=" * 80)
            logger.info("✅ Validation Completed Successfully!")
            logger.info(f"⏱️  Duration: {validation_duration:.2f}s")
            logger.info("=" * 80)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            raise
    
    def _run_chunking(self, input_path: str) -> Dict[str, Any]:
        """Run the chunking step"""
        from .chunking_step import ChunkingStep
        
        chunking_step = ChunkingStep(
            chunk_size=self.config.get("chunk_size", 1200),
            chunk_overlap=self.config.get("chunk_overlap", 200),
            max_tokens=self.config.get("max_tokens", 50000)
        )
        
        input_path_obj = Path(input_path)
        if input_path_obj.is_file():
            chunking_result = chunking_step.process_file(input_path)
        else:
            chunking_result = chunking_step.process_directory(input_path, str(self.output_dir / "chunks.jsonl"))
        
        self.processing_stats["total_chunks"] += chunking_result["chunking_stats"]["total_chunks"]
        
        return chunking_result
    
    def _run_classification(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run the classification agent"""
        classification_result = self.classifier_agent.process({
            "chunks": chunks,
            "document_context": "Trade and customs document processing"
        })
        
        # Save classification
        classification_file = self.output_dir / "classification.json"
        with open(classification_file, 'w') as f:
            json.dump(classification_result, f, indent=2)
        
        return classification_result
    
    def _run_extraction(self, 
                       chunks: List[Dict[str, Any]], 
                       classification: Dict[str, Any],
                       model_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Run the extraction agent"""
        extraction_result = self.extractor_agent.process({
            "chunks": chunks,
            "classification": classification,
            "model_spec": model_spec
        })
        
        # Save records
        records_file = self.output_dir / "records.jsonl"
        with open(records_file, 'w') as f:
            for record in extraction_result["records"]:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        self.processing_stats["total_records"] += len(extraction_result["records"])
        
        return extraction_result
    
    def _run_validation(self, 
                       records: List[Dict[str, Any]], 
                       chunks: List[Dict[str, Any]],
                       model_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Run the validation agent using file-based approach"""
        
        # Save records and chunks to files for validation agent
        records_file = self.output_dir / "records.jsonl"
        chunks_file = self.output_dir / "all_chunks.jsonl"
        
        # Save records in the format expected by validation agent
        with open(records_file, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        # Save chunks in the format expected by validation agent
        with open(chunks_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        
        # Run validation using the new file-based method
        validation_summary = self.validator_agent.validate_records_file(
            records_file=str(records_file),
            chunks_file=str(chunks_file),
            output_dir=str(self.output_dir)
        )
        
        # Load validation results
        validation_file = self.output_dir / "validation.jsonl"
        validation_results = []
        if validation_file.exists():
            with open(validation_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        validation_results.append(json.loads(line))
        
        self.processing_stats["total_validations"] += len(validation_results)
        
        return {
            "validation_summary": validation_summary,
            "validation_results": validation_results,
            "validation_stats": {
                "total_validations": len(validation_results),
                "validation_rate": validation_summary.get('overall', {}).get('validation_rate', 0),
                "valid_records": validation_summary.get('overall', {}).get('valid_records', 0),
                "invalid_records": validation_summary.get('overall', {}).get('invalid_records', 0),
                "missing_records": validation_summary.get('overall', {}).get('missing_records', 0),
                "partial_records": validation_summary.get('overall', {}).get('partial_records', 0)
            }
        }
    
    def _save_pipeline_results(self, results: Dict[str, Any], output_prefix: str) -> None:
        """Save complete pipeline results"""
        
        # Save main results
        results_file = self.output_dir / f"{output_prefix}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save agent stats
        stats_file = self.output_dir / f"{output_prefix}_agent_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self._get_agent_stats(), f, indent=2)
        
        logger.info(f"💾 Results saved to {self.output_dir}")
    
    def _get_agent_stats(self) -> Dict[str, Any]:
        """Get statistics from all agents"""
        return {
            "classifier_agent": self.classifier_agent.get_stats(),
            "extractor_agent": self.extractor_agent.get_stats(),
            "validator_agent": self.validator_agent.get_stats(),
            "pipeline_stats": self.processing_stats
        }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            "pipeline_state": self.pipeline_state,
            "processing_stats": self.processing_stats,
            "agent_stats": self._get_agent_stats(),
            "output_directory": str(self.output_dir)
        }
    
    def reset_pipeline(self) -> None:
        """Reset pipeline state and statistics"""
        self.pipeline_state = {}
        self.processing_stats = {
            "total_files": 0,
            "successful_files": 0,
            "failed_files": 0,
            "total_chunks": 0,
            "total_records": 0,
            "total_validations": 0
        }
        
        # Reset agent stats
        self.classifier_agent.reset_stats()
        self.extractor_agent.reset_stats()
        self.validator_agent.reset_stats()
        
        logger.info("Pipeline state reset")

# CLI interface
if __name__ == "__main__":
    import sys
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Agent-Based Pipeline Orchestrator")
    parser.add_argument("--input", help="Input file or directory (required for full pipeline)")
    parser.add_argument("--model", required=True, help="Model name from data dictionary")
    parser.add_argument("--country", required=True, help="Country code (e.g., CA, US, AU) - determines defaults file")
    parser.add_argument("--dict", default="data_dictionary.yaml", help="Path to data dictionary YAML file")
    parser.add_argument("--output-dir", default="agent_output", help="Output directory")
    parser.add_argument("--steps", nargs="+", choices=["chunking", "classification", "extraction", "validation"],
                       help="Specific steps to run")
    parser.add_argument("--records", help="Path to existing records.jsonl file (for validation-only mode)")
    parser.add_argument("--chunks", help="Path to existing all_chunks.jsonl file (for validation-only mode)")
    parser.add_argument("--chunk-size", type=int, default=1200, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap")
    parser.add_argument("--max-tokens", type=int, default=100000, help="Maximum tokens for validation prompts")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Load data dictionary and get model spec
        with open(args.dict, 'r', encoding='utf-8') as f:
            data_dict = yaml.safe_load(f)
        
        model_spec = data_dict.get('models', {}).get(args.model)
        if not model_spec:
            print(f"Error: Model '{args.model}' not found in data dictionary")
            sys.exit(1)
        
        # Validate input requirements
        if args.records and args.chunks:
            # Validation-only mode - input not required
            if not args.input:
                print("Running validation-only mode...")
        else:
            # Full pipeline mode - input required
            if not args.input:
                print("Error: --input is required for full pipeline mode")
                print("Use --records and --chunks for validation-only mode")
                sys.exit(1)
        
        # Initialize orchestrator
        orchestrator = AgentPipelineOrchestrator({
            "output_dir": args.output_dir,
            "country": args.country,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "max_tokens": args.max_tokens
        })
        
        # Run pipeline
        if args.records and args.chunks:
            # Validation-only mode
            print("Running validation-only mode...")
            results = orchestrator.run_validation_only(args.records, args.chunks, model_spec)
        elif args.steps:
            results = orchestrator.run_step_by_step(args.input, model_spec, args.steps)
        else:
            results = orchestrator.run_full_pipeline(args.input, model_spec)
        
        # Print summary
        print("=" * 60)
        print("AGENT PIPELINE SUMMARY")
        print("=" * 60)
        print(f"Input: {args.input}")
        print(f"Model: {args.model}")
        print(f"Output: {args.output_dir}")
        print(f"Duration: {results['pipeline_stats']['duration_seconds']:.2f}s")
        print(f"Chunks: {results['pipeline_stats']['chunking_stats']['total_chunks']}")
        print(f"Records: {results['pipeline_stats'].get('extraction_stats', {}).get('total_records', 0)}")
        
        # Print validation summary if available
        validation_stats = results.get('validation', {}).get('validation_stats', {})
        if validation_stats:
            print(f"Validation Rate: {validation_stats.get('validation_rate', 0):.2%}")
            print(f"Valid Records: {validation_stats.get('valid_records', 0)}")
            print(f"Invalid Records: {validation_stats.get('invalid_records', 0)}")
            print(f"Missing Records: {validation_stats.get('missing_records', 0)}")
            print(f"Partial Records: {validation_stats.get('partial_records', 0)}")
        
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
