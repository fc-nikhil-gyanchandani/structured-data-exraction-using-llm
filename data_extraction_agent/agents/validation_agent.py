"""
Validation Agent using PromptBuilder for data validation
"""

import json
import logging
import sys
import argparse
from typing import Dict, Any, List, Optional
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from langchain.schema import BaseMessage, HumanMessage, SystemMessage

from .base import BaseAgent
from ..config import config
from ..prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)

# Set up file logging
def setup_file_logging():
    """Set up file logging for validation agent"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"validation_agent_{timestamp}.log"
    
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

class ValidationAgent(BaseAgent):
    """Data validation agent using PromptBuilder"""
    
    def __init__(self, agent_config: Optional[Dict[str, Any]] = None):
        super().__init__("validation_agent", agent_config)
        
        # Get country and defaults from agent config
        self.country = agent_config.get("country", "CA") if agent_config else "CA"
        self.defaults = agent_config.get("defaults", {}) if agent_config else {}
        
        # Initialize prompt builder with country defaults
        self.prompt_builder = PromptBuilder(
            country=self.country,
            defaults=self.defaults
        )
        self.max_tokens = self.config.get("max_tokens", 100000)
        
        # Set up file logging
        self.log_file = setup_file_logging()
        logger.info("Validation Agent initialized")
    
    def validate_records_file(self, records_file: str, chunks_file: str, output_dir: str, model_filter: str = None) -> Dict[str, Any]:
        """
        Validate records from records.jsonl file grouped by model
        
        Args:
            records_file: Path to records.jsonl file
            chunks_file: Path to all_chunks.jsonl file
            output_dir: Output directory for validation results
            model_filter: Optional model name to filter records (validates only this model)
            
        Returns:
            Dictionary with validation summary
        """
        try:
            logger.info("=" * 80)
            logger.info("STARTING VALIDATION PROCESS")
            logger.info("=" * 80)
            logger.info(f"Records file: {records_file}")
            logger.info(f"Chunks file: {chunks_file}")
            logger.info(f"Output directory: {output_dir}")
            logger.info(f"Model filter: {model_filter or 'All models'}")
            logger.info(f"Max tokens: {self.max_tokens}")
            
            # Load records and group by model
            logger.info("Loading and grouping records by model...")
            records_by_model = self._load_and_group_records(records_file)
            logger.info(f"Found {len(records_by_model)} models in records: {list(records_by_model.keys())}")
            
            # Filter by model if specified
            if model_filter:
                if model_filter not in records_by_model:
                    logger.warning(f"Model '{model_filter}' not found in records. Available models: {list(records_by_model.keys())}")
                    return {
                        "overall": {
                            "total_records": 0,
                            "valid_records": 0,
                            "invalid_records": 0,
                            "missing_records": 0,
                            "partial_records": 0,
                            "validation_rate": 0.0
                        },
                        "by_model": {},
                        "models_processed": 0
                    }
                records_by_model = {model_filter: records_by_model[model_filter]}
                logger.info(f"Filtering to model: {model_filter}")
            
            # Load all chunks
            logger.info("Loading evidence chunks...")
            chunks_dict = self._load_chunks(chunks_file)
            logger.info(f"Loaded {len(chunks_dict)} evidence chunks")
            
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            logger.info(f"Created output directory: {output_path}")
            
            all_validation_results = []
            model_summaries = {}
            
            # Process each model
            for model_name, records in records_by_model.items():
                logger.info(f"Processing model: {model_name} with {len(records)} records")
                
                # Get model specification
                model_spec = self.prompt_builder.get_model_spec(model_name)
                if not model_spec:
                    logger.warning(f"Model specification not found for: {model_name}")
                    continue
                
                # Get evidence chunks for this model's records
                evidence_chunks = self._get_evidence_chunks_for_model(records, chunks_dict)
                logger.info(f"Found {len(evidence_chunks)} evidence chunks for model {model_name}")
                
                # Validate records for this model
                model_results = self._validate_model_records(
                    model_name, records, model_spec, evidence_chunks
                )
                
                all_validation_results.extend(model_results)
                model_summaries[model_name] = self._create_model_summary(model_results)
                logger.info(f"Completed validation for model {model_name}: {len(model_results)} results")
            
            # Write validation results
            logger.info("Writing validation results to files...")
            self._write_validation_results(all_validation_results, output_path)
            
            # Write summary
            summary = self._create_overall_summary(model_summaries)
            self._write_validation_summary(summary, output_path)
            
            logger.info("=" * 80)
            logger.info("VALIDATION PROCESS COMPLETED")
            logger.info("=" * 80)
            logger.info(f"Total records processed: {len(all_validation_results)}")
            logger.info(f"Models processed: {len(records_by_model)}")
            logger.info(f"Overall validation rate: {summary['overall']['validation_rate']:.2%}")
            logger.info(f"Log file: {self.log_file}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Records file validation failed: {e}")
            raise
    
    def _load_and_group_records(self, records_file: str) -> Dict[str, List[Dict[str, Any]]]:
        """Load records from JSONL file and group by model"""
        records_by_model = defaultdict(list)
        
        logger.info(f"Loading records from: {records_file}")
        with open(records_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        record = json.loads(line)
                        model_name = record.get('model')
                        if model_name:
                            records_by_model[model_name].append(record)
                        else:
                            logger.warning(f"Record at line {line_num} has no model field")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON at line {line_num}: {e}")
        
        logger.info(f"Loaded records grouped by model: {list(records_by_model.keys())}")
        return dict(records_by_model)
    
    def _load_chunks(self, chunks_file: str) -> Dict[str, Dict[str, Any]]:
        """Load all chunks into a dictionary keyed by chunk_id"""
        chunks_dict = {}
        
        logger.info(f"Loading chunks from: {chunks_file}")
        with open(chunks_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        chunk = json.loads(line)
                        chunk_id = chunk.get('chunk_id')
                        if chunk_id:
                            chunks_dict[chunk_id] = chunk
                        else:
                            logger.warning(f"Chunk at line {line_num} has no chunk_id field")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse chunk JSON at line {line_num}: {e}")
        
        logger.info(f"Loaded {len(chunks_dict)} chunks")
        return chunks_dict
    
    def _estimate_tokens(self, data: Any) -> int:
        """Roughly estimate the number of tokens for a given data structure."""
        return len(json.dumps(data, ensure_ascii=False)) // 4

    def _batch_records_by_tokens(self, records_data: List[Dict[str, Any]], 
                                 evidence_chunks: List[Dict[str, Any]], 
                                 model_spec: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        """Batches records to not exceed token limits for the validation prompt."""
        max_tokens_per_batch = self.max_tokens
        
        # Estimate the base token count for the prompt (template, evidence, etc.)
        # This is a rough estimate and can be refined.
        base_prompt_tokens = self._estimate_tokens({
            "model_spec": model_spec,
            "evidence_chunks": evidence_chunks
        }) + 4000  # More conservative buffer for instructions and formatting

        batches = []
        current_batch = []
        current_tokens = base_prompt_tokens

        for record in records_data:
            record_tokens = self._estimate_tokens(record)
            
            if current_batch and (current_tokens + record_tokens > max_tokens_per_batch):
                batches.append(current_batch)
                current_batch = []
                current_tokens = base_prompt_tokens

            current_batch.append(record)
            current_tokens += record_tokens

        if current_batch:
            batches.append(current_batch)
            
        logger.info(f"Created {len(batches)} batches from {len(records_data)} records to respect token limits.")
        return batches

    def _get_evidence_chunks_for_model(self, records: List[Dict[str, Any]], chunks_dict: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get all unique evidence chunks for a model's records"""
        chunk_ids = set()
        
        for record in records:
            record_chunk_ids = record.get('chunk_ids', [])
            chunk_ids.update(record_chunk_ids)
        
        evidence_chunks = []
        for chunk_id in chunk_ids:
            if chunk_id in chunks_dict:
                chunk = chunks_dict[chunk_id]
                evidence_chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk.get('text', '')
                })
        
        return evidence_chunks
    
    def _validate_model_records(self, model_name: str, records: List[Dict[str, Any]], 
                              model_spec: Dict[str, Any], evidence_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate records for a specific model with batching."""
        try:
            logger.info(f"Starting validation for model: {model_name}")
            logger.info(f"Total records to validate: {len(records)}")
            logger.info(f"Evidence chunks available: {len(evidence_chunks)}")
            
            all_results = []
            
            # Extract just the record data for batching
            records_data = [record.get('record', {}) for record in records]
            
            # Create token-based batches
            record_batches = self._batch_records_by_tokens(records_data, evidence_chunks, model_spec)
            
            total_batches = len(record_batches)
            for i, batch_records in enumerate(record_batches):
                batch_number = i + 1
                logger.info(f"Processing batch {batch_number}/{total_batches} for model {model_name}...")
                
                # Build validation prompt for the current batch
                logger.info("Building validation prompt for batch...")
                prompt = self.prompt_builder.build_validation_prompt_for_model(
                    model_name=model_name,
                    records_data=batch_records,
                    evidence_chunks=evidence_chunks
                )
                
                # Log the prompt
                prompt_tokens = self._estimate_tokens(prompt)
                logger.info(f"=== VALIDATION PROMPT FOR MODEL {model_name} (BATCH {batch_number}) ===")
                logger.info(f"Estimated prompt tokens: {prompt_tokens}")
                
                # Log complete prompt content
                logger.info(f"Prompt content:\n{prompt}")
                
                logger.info("=== END PROMPT ===")

                if prompt_tokens > self.max_tokens:
                    logger.warning(
                        f"Prompt for batch {batch_number} exceeds token limit "
                        f"({prompt_tokens} > {self.max_tokens}). "
                        "Skipping this batch to avoid API errors."
                    )
                    continue
                
                # Send to LLM
                logger.info(f"Sending batch {batch_number} to LLM...")
                response = self._send_to_llm(prompt, model_name)
                
                # Parse response
                logger.info(f"Parsing LLM response for batch {batch_number}...")
                validation_response = self._parse_validation_response(response)
                
                batch_results = validation_response.get('results', [])
                all_results.extend(batch_results)
                logger.info(f"Successfully validated {len(batch_results)} records in batch {batch_number}.")

            # Log processing step for the entire model
            self.log_processing_step(
                "data_validation",
                {"records_count": len(records), "model": model_name},
                {"validation_status": "completed"}
            )
            
            logger.info(f"Successfully validated a total of {len(all_results)} records for model {model_name}")
            
            return all_results
            
        except Exception as e:
            logger.error(f"Model validation failed for {model_name}: {e}")
            # Return empty results for failed model
            return []
    
    def validate_data(self, extracted_data: List[Dict[str, Any]], model_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate extracted data using model specification (legacy method)
        
        Args:
            extracted_data: List of extracted data records
            model_spec: Model specification for validation
            
        Returns:
            Dictionary with validation results and metadata
        """
        try:
            # Build validation prompt using PromptBuilder
            model_name = model_spec.get("model_name", "unknown")
            prompt = self.prompt_builder.build_validation_prompt_for_model(
                model_name=model_name,
                records_data=extracted_data,
                evidence_chunks=[]
            )
            
            # Send to LLM
            response = self._send_to_llm(prompt)
            
            # Parse response
            validation_results = self._parse_validation_response(response)
            
            # Log processing step
            self.log_processing_step(
                "data_validation",
                {"records_count": len(extracted_data), "model": model_spec.get("model_name", "unknown")},
                {"validation_status": validation_results.get("status", "unknown")}
            )
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            raise
    
    def _send_to_llm(self, prompt: str, model_name: str = "unknown") -> str:
        """Send prompt to LLM and get response"""
        try:
            # Create messages
            messages = [
                SystemMessage(content="You are a Senior Customs Data Analysis Specialist."),
                HumanMessage(content=prompt)
            ]
            
            logger.info(f"Sending request to LLM with {len(messages)} messages")
            
            # Get LLM response
            response = self.llm.invoke(messages)
            
            if hasattr(response, 'content'):
                response_content = response.content
            else:
                response_content = str(response)
            
            # Log the response
            logger.info(f"=== LLM RESPONSE FOR MODEL {model_name} ===")
            logger.info(f"Response length: {len(str(response_content))} characters")
            logger.info(f"Response content:\n{response_content}")
            logger.info("=== END RESPONSE ===")
            
            return response_content
                
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            raise
    
    def _parse_validation_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response and extract validation results"""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                logger.warning("No JSON found in validation response")
                return {"status": "error", "message": "No valid JSON response"}
            
            json_str = response[json_start:json_end]
            parsed_response = json.loads(json_str)
            
            return parsed_response
            
        except Exception as e:
            logger.error(f"Failed to parse validation response: {e}")
            return {"status": "error", "message": f"Parse error: {str(e)}"}
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from data dictionary"""
        return self.prompt_builder.get_available_models()
    
    def get_model_spec(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get specification for a specific model"""
        return self.prompt_builder.get_model_spec(model_name)
    
    def _create_model_summary(self, validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary for a single model's validation results"""
        total_records = len(validation_results)
        if total_records == 0:
            return {
                "total_records": 0,
                "valid_records": 0,
                "invalid_records": 0,
                "missing_records": 0,
                "partial_records": 0,
                "validation_rate": 0.0
            }
        
        status_counts = {"V": 0, "I": 0, "M": 0, "P": 0}
        
        for result in validation_results:
            status = result.get("status", "M")
            if status in status_counts:
                status_counts[status] += 1
        
        valid_records = status_counts["V"]
        validation_rate = valid_records / total_records if total_records > 0 else 0.0
        
        return {
            "total_records": total_records,
            "valid_records": valid_records,
            "invalid_records": status_counts["I"],
            "missing_records": status_counts["M"],
            "partial_records": status_counts["P"],
            "validation_rate": validation_rate
        }
    
    def _create_overall_summary(self, model_summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create overall summary across all models"""
        total_records = sum(summary["total_records"] for summary in model_summaries.values())
        total_valid = sum(summary["valid_records"] for summary in model_summaries.values())
        total_invalid = sum(summary["invalid_records"] for summary in model_summaries.values())
        total_missing = sum(summary["missing_records"] for summary in model_summaries.values())
        total_partial = sum(summary["partial_records"] for summary in model_summaries.values())
        
        overall_validation_rate = total_valid / total_records if total_records > 0 else 0.0
        
        return {
            "overall": {
                "total_records": total_records,
                "valid_records": total_valid,
                "invalid_records": total_invalid,
                "missing_records": total_missing,
                "partial_records": total_partial,
                "validation_rate": overall_validation_rate
            },
            "by_model": model_summaries,
            "models_processed": len(model_summaries)
        }
    
    def _write_validation_results(self, validation_results: List[Dict[str, Any]], output_path: Path):
        """Write validation results to validation.jsonl"""
        validation_file = output_path / "validation.jsonl"
        
        with open(validation_file, 'w', encoding='utf-8') as f:
            for result in validation_results:
                f.write(json.dumps(result) + '\n')
        
        logger.info(f"Validation results written to: {validation_file}")
    
    def _write_validation_summary(self, summary: Dict[str, Any], output_path: Path):
        """Write validation summary to validation_summary.json"""
        summary_file = output_path / "validation_summary.json"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Validation summary written to: {summary_file}")
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process method required by BaseAgent abstract class
        
        Args:
            input_data: Dictionary containing 'records', 'chunks', and 'model_spec'
            
        Returns:
            Dictionary with validation results
        """
        try:
            records = input_data.get('records', [])
            chunks = input_data.get('chunks', [])
            model_spec = input_data.get('model_spec', {})
            
            # Create temporary files for validation
            import tempfile
            import os
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Write records to temporary file
                records_file = os.path.join(temp_dir, "temp_records.jsonl")
                with open(records_file, 'w', encoding='utf-8') as f:
                    for record in records:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
                # Write chunks to temporary file
                chunks_file = os.path.join(temp_dir, "temp_chunks.jsonl")
                with open(chunks_file, 'w', encoding='utf-8') as f:
                    for chunk in chunks:
                        f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                
                # Run validation
                summary = self.validate_records_file(
                    records_file=records_file,
                    chunks_file=chunks_file,
                    output_dir=temp_dir
                )
                
                # Load validation results
                validation_file = os.path.join(temp_dir, "validation.jsonl")
                validation_results = []
                if os.path.exists(validation_file):
                    with open(validation_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                validation_results.append(json.loads(line))
                
                return {
                    "validation_summary": summary,
                    "validation_results": validation_results,
                    "validations": validation_results  # For compatibility
                }
                
        except Exception as e:
            logger.error(f"Process method failed: {e}")
            return {
                "validation_summary": {"overall": {"validation_rate": 0.0}},
                "validation_results": [],
                "validations": []
            }

# CLI interface
if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path
    
    def main():
        """CLI entry point for validation agent"""
        parser = argparse.ArgumentParser(
            description="Validation Agent - Validate extracted records against evidence chunks",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Basic validation
  python -m data_extraction_agent.agents.validation_agent --records records.jsonl --chunks all_chunks.jsonl --output validation_output
  
  # With specific model
  python -m data_extraction_agent.agents.validation_agent --records records.jsonl --chunks all_chunks.jsonl --model countries_stg --output validation_output
  
  # With custom token limit
  python -m data_extraction_agent.agents.validation_agent --records records.jsonl --chunks all_chunks.jsonl --max-tokens 50000 --output validation_output
            """
        )
        
        parser.add_argument("--records", "-r", required=True, help="Path to records.jsonl file")
        parser.add_argument("--chunks", "-c", required=True, help="Path to all_chunks.jsonl file")
        parser.add_argument("--output", "-o", required=True, help="Output directory for validation results")
        parser.add_argument("--model", "-m", help="Specific model to validate (optional, validates all models if not specified)")
        parser.add_argument("--max-tokens", type=int, default=100000, help="Maximum tokens for validation prompts (default: 100000)")
        parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
        
        args = parser.parse_args()
        
        # Setup logging - only to file, no console output
        log_level = logging.DEBUG if args.verbose else logging.INFO
        
        # Configure root logger to only use file logging
        logging.basicConfig(
            level=log_level,
            handlers=[],  # No handlers = no console output
            force=True
        )
        
        # Validate input files
        records_path = Path(args.records)
        chunks_path = Path(args.chunks)
        
        if not records_path.exists():
            print(f"Error: Records file not found: {records_path}")
            sys.exit(1)
        
        if not chunks_path.exists():
            print(f"Error: Chunks file not found: {chunks_path}")
            sys.exit(1)
        
        try:
            # Initialize validation agent
            agent_config = config.get_agent_config("validator")
            validation_agent = ValidationAgent(agent_config=agent_config)
            
            # Set custom token limit if provided
            validation_agent.max_tokens = args.max_tokens
            
            # Run validation
            summary = validation_agent.validate_records_file(
                records_file=str(records_path),
                chunks_file=str(chunks_path),
                output_dir=args.output,
                model_filter=args.model
            )
            
            # Print minimal results to console
            overall = summary['overall']
            print(f"Validation completed: {overall['valid_records']}/{overall['total_records']} records valid ({overall['validation_rate']:.2%})")
            print(f"Results: {Path(args.output) / 'validation.jsonl'}")
            print(f"Summary: {Path(args.output) / 'validation_summary.json'}")
            print(f"Log file: {validation_agent.log_file}")
            
        except Exception as e:
            print(f"Validation failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    
    main()
