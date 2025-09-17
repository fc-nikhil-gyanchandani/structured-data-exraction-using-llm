"""
Extraction Agent using PromptBuilder for data extraction
"""

import json
import logging
import sys
import argparse
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
    """Set up file logging for extraction agent"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"extraction_agent_{timestamp}.log"
    
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
    
    return str(log_file)

class ExtractionAgent(BaseAgent):
    """Data extraction agent using PromptBuilder"""
    
    def __init__(self, agent_config: Optional[Dict[str, Any]] = None, mappings_file: Optional[str] = None):
        super().__init__("extraction_agent", agent_config)
        
        # Set up file logging
        self.log_file = setup_file_logging()
        logger.info(f"ExtractionAgent initialized with log file: {self.log_file}")
        
        # Get country and defaults from agent config
        self.country = agent_config.get("country", "CA") if agent_config else "CA"
        self.defaults = agent_config.get("defaults", {}) if agent_config else {}
        
        # Initialize prompt builder with country defaults
        self.prompt_builder = PromptBuilder(
            country=self.country,
            defaults=self.defaults
        )
        
        # Token limits are now managed by the agent's config property
        self.max_tokens_per_prompt = self.config.get("max_tokens", 100000)
        
        # Load mappings file
        self.mappings_file = mappings_file
        self.mappings = self._load_mappings()
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inputs and return results (required by BaseAgent)
        
        Args:
            inputs: Dictionary containing 'file_chunks' and 'model_spec'
            
        Returns:
            Dictionary with extracted data and metadata
        """
        file_chunks = inputs.get("file_chunks", [])
        model_spec = inputs.get("model_spec", {})
        
        if not file_chunks:
            return {"error": "No file chunks provided"}
        
        if not model_spec:
            return {"error": "No model specification provided"}
        
        return self._extract_data_with_batching(file_chunks, model_spec)
    
    def _estimate_tokens(self, data: Any) -> int:
        """Roughly estimate the number of tokens for a given data structure."""
        return len(json.dumps(data, ensure_ascii=False)) // 4

    def _batch_chunks_by_tokens(self, chunks: List[Dict[str, Any]], model_spec: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        """Batches chunks to not exceed token limits for the extraction prompt."""
        
        # Estimate the base token count for the prompt (template, model_spec, etc.)
        base_prompt_tokens = self._estimate_tokens(model_spec) + 4000  # More conservative buffer

        batches = []
        current_batch = []
        current_tokens = base_prompt_tokens

        for chunk in chunks:
            chunk_tokens = self._estimate_tokens(chunk.get("text", ""))
            
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

    def _extract_data_with_batching(self, file_chunks: List[Dict[str, Any]], model_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data with batching to handle large numbers of chunks."""
        try:
            model_name = model_spec.get("model_name", "unknown")

            # Filter chunks based on model-specific source files
            filtered_chunks = self._filter_chunks_by_model(file_chunks, model_spec)
            
            if not filtered_chunks:
                logger.warning(f"No chunks found for model {model_name} after source filtering")
                return {"records": [], "metadata": {"status": "no_matching_chunks"}}

            # Create batches
            chunk_batches = self._batch_chunks_by_tokens(filtered_chunks, model_spec)
            
            all_records = []
            total_batches = len(chunk_batches)

            for i, batch in enumerate(chunk_batches):
                batch_num = i + 1
                logger.info(f"Processing batch {batch_num}/{total_batches} for model {model_name}...")
                
                # Build extraction prompt for the current batch
                prompt = self.prompt_builder.build_extraction_prompt(
                    file_chunks=batch,
                    model_spec=model_spec
                )
                
                prompt_tokens = self._estimate_tokens(prompt)
                logger.info(f"=== EXTRACTION PROMPT FOR MODEL {model_name} (BATCH {batch_num}) ===")
                logger.info(f"Estimated prompt tokens: {prompt_tokens}")
                
                # Log complete prompt content
                logger.info(f"Prompt content:\n{prompt}")
                
                logger.info("=== END PROMPT ===")

                if prompt_tokens > self.max_tokens_per_prompt:
                    logger.warning(
                        f"Prompt for batch {batch_num} of {model_name} exceeds token limit "
                        f"({prompt_tokens} > {self.max_tokens_per_prompt}). Skipping batch."
                    )
                    continue
                
                # Send to LLM
                response = self._send_to_llm(prompt)
                
                # Parse response
                extracted_data = self._parse_extraction_response(response)
                batch_records = extracted_data.get("records", [])
                
                if batch_records:
                    all_records.extend(batch_records)
                
                logger.info(f"Extracted {len(batch_records)} records from batch {batch_num}.")

            # Consolidate results
            final_result = {
                "records": all_records,
                "model_name": model_name,
                "chunk_ids": [chunk.get("chunk_id", "") for chunk in filtered_chunks],
                "metadata": {
                    "status": "success",
                    "batches_processed": total_batches,
                    "total_records": len(all_records)
                }
            }

            self.log_processing_step(
                "data_extraction",
                {"chunks_count": len(filtered_chunks), "model": model_name},
                {"records_extracted": len(all_records)}
            )
            
            return final_result
            
        except Exception as e:
            logger.error(f"Data extraction failed: {e}")
            raise

    def extract_data(self, file_chunks: List[Dict[str, Any]], model_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract data from file chunks using model specification
        
        Args:
            file_chunks: List of chunk dictionaries
            model_spec: Model specification for extraction
            
        Returns:
            Dictionary with extracted data and metadata
        """
        # This method is now a simple wrapper for the batching implementation
        return self._extract_data_with_batching(file_chunks, model_spec)
    
    def _send_to_llm(self, prompt: str) -> str:
        """Send prompt to LLM and get response"""
        try:
            # Ensure prompt is a string
            if isinstance(prompt, dict):
                prompt = json.dumps(prompt, indent=2)
            elif isinstance(prompt, list):
                prompt = json.dumps(prompt, indent=2)
            
            # Create messages
            messages = [
                SystemMessage(content="You are a Senior Customs Data Analysis Specialist."),
                HumanMessage(content=str(prompt))
            ]
            
            logger.info(f"Sending request to LLM with {len(messages)} messages")
            
            # Get LLM response
            response = self.llm.invoke(messages)
            
            if hasattr(response, 'content'):
                response_content = response.content
            else:
                response_content = str(response)
            
            # Log complete response content
            logger.info(f"LLM response received:")
            logger.info(f"Response length: {len(response_content)} characters")
            logger.info(f"Complete response:\n{response_content}")
            
            return response_content
                
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            raise
    
    def _parse_extraction_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response and extract data"""
        try:
            logger.info("Parsing LLM response...")
            
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                logger.warning("No JSON found in extraction response")
                return {"records": [], "metadata": {"status": "no_data"}}
            
            json_str = response[json_start:json_end]
            logger.info(f"Extracted JSON string (length: {len(json_str)})")
            
            # Try to parse JSON
            try:
                parsed_response = json.loads(json_str)
                logger.info(f"Successfully parsed JSON response with {len(parsed_response.get('records', []))} records")
                return parsed_response
            except json.JSONDecodeError as json_error:
                logger.warning(f"JSON parsing failed: {json_error}")
                logger.warning(f"Attempting to fix common JSON issues...")
                
                # Try to fix common JSON issues
                fixed_json = self._fix_json_issues(json_str)
                if fixed_json != json_str:
                    logger.info("Applied JSON fixes, attempting to parse again...")
                    try:
                        parsed_response = json.loads(fixed_json)
                        logger.info(f"Successfully parsed fixed JSON response with {len(parsed_response.get('records', []))} records")
                        return parsed_response
                    except json.JSONDecodeError as fix_error:
                        logger.error(f"JSON parsing still failed after fixes: {fix_error}")
                
                # If all else fails, try to extract partial data
                logger.warning("Attempting to extract partial data from malformed JSON...")
                partial_data = self._extract_partial_data(json_str)
                if partial_data:
                    logger.info(f"Extracted partial data with {len(partial_data.get('records', []))} records")
                    return partial_data
                
                raise json_error
            
        except Exception as e:
            logger.error(f"Failed to parse extraction response: {e}")
            logger.error(f"Response that failed to parse: {response[:1000]}...")
            return {"records": [], "metadata": {"status": "parse_error", "error": str(e)}}
    
    def _fix_json_issues(self, json_str: str) -> str:
        """Fix common JSON issues in LLM responses"""
        import re
        
        # Remove any text before the first {
        json_str = json_str[json_str.find('{'):]
        
        # Fix missing commas between array elements
        json_str = re.sub(r'}\s*{', '},{', json_str)
        
        # Fix missing commas between object properties
        json_str = re.sub(r'"\s*\n\s*"', '",\n"', json_str)
        json_str = re.sub(r'}\s*\n\s*"', '},\n"', json_str)
        
        # Fix trailing commas before closing brackets/braces
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Fix missing quotes around keys
        json_str = re.sub(r'(\w+):', r'"\1":', json_str)
        
        return json_str
    
    def _extract_partial_data(self, json_str: str) -> Dict[str, Any]:
        """Extract partial data from malformed JSON"""
        try:
            # Try to find records array
            records_start = json_str.find('"records"')
            if records_start == -1:
                return {"records": [], "metadata": {"status": "no_records_found"}}
            
            # Find the records array content
            array_start = json_str.find('[', records_start)
            if array_start == -1:
                return {"records": [], "metadata": {"status": "no_array_found"}}
            
            # Try to extract individual record objects
            records = []
            bracket_count = 0
            current_record = ""
            in_string = False
            escape_next = False
            
            for i, char in enumerate(json_str[array_start + 1:], array_start + 1):
                if escape_next:
                    escape_next = False
                    current_record += char
                    continue
                
                if char == '\\':
                    escape_next = True
                    current_record += char
                    continue
                
                if char == '"' and not escape_next:
                    in_string = not in_string
                    current_record += char
                    continue
                
                if not in_string:
                    if char == '{':
                        bracket_count += 1
                        current_record += char
                    elif char == '}':
                        bracket_count -= 1
                        current_record += char
                        
                        if bracket_count == 0 and current_record.strip():
                            # Found a complete record
                            try:
                                record = json.loads(current_record.strip())
                                records.append(record)
                                current_record = ""
                            except:
                                # Skip malformed record
                                current_record = ""
                    elif char == ']' and bracket_count == 0:
                        # End of records array
                        break
                    else:
                        current_record += char
                else:
                    current_record += char
            
            return {
                "records": records,
                "metadata": {
                    "status": "partial_extraction",
                    "total_records": len(records),
                    "note": "Extracted from malformed JSON"
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to extract partial data: {e}")
            return {"records": [], "metadata": {"status": "extraction_failed", "error": str(e)}}
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from data dictionary"""
        return self.prompt_builder.get_available_models()
    
    def get_model_spec(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get specification for a specific model"""
        return self.prompt_builder.get_model_spec(model_name)
    
    def _load_mappings(self) -> Dict[str, Any]:
        """Load mappings from JSON file"""
        if not self.mappings_file:
            logger.warning("No mappings file provided, source filtering will be disabled")
            return {}
        
        try:
            with open(self.mappings_file, 'r', encoding='utf-8') as f:
                mappings_data = json.load(f)
            
            # Convert list of mappings to dictionary for easier lookup
            mappings_dict = {}
            for mapping in mappings_data.get("mappings", []):
                model_name = mapping.get("model")
                if model_name:
                    mappings_dict[model_name] = {
                        "filenames": mapping.get("filename", []),
                        "confidence_score": mapping.get("confidence_score", 0.0)
                    }
            
            logger.info(f"Loaded mappings for {len(mappings_dict)} models from {self.mappings_file}")
            return mappings_dict
            
        except FileNotFoundError:
            logger.error(f"Mappings file not found: {self.mappings_file}")
            return {}
        except Exception as e:
            logger.error(f"Failed to load mappings file: {e}")
            return {}
    
    def _filter_chunks_by_model(self, file_chunks: List[Dict[str, Any]], model_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter chunks based on model-specific source files"""
        model_name = model_spec.get("model_name")
        if not model_name:
            logger.warning("No model name found in model_spec, returning all chunks")
            return file_chunks
        
        # Get filenames for this model from mappings
        model_mapping = self.mappings.get(model_name)
        if not model_mapping:
            logger.warning(f"No mapping found for model {model_name}, returning all chunks")
            return file_chunks
        
        target_filenames = model_mapping.get("filenames", [])
        if not target_filenames:
            logger.warning(f"No filenames found for model {model_name}, returning all chunks")
            return file_chunks
        
        logger.info(f"Filtering chunks for model {model_name} using filenames: {target_filenames}")
        
        # Filter chunks based on source_path containing any of the target filenames
        filtered_chunks = []
        for chunk in file_chunks:
            source_path = chunk.get("source_path", "")
            if any(filename in source_path for filename in target_filenames):
                filtered_chunks.append(chunk)
                logger.debug(f"Chunk {chunk.get('chunk_id', 'unknown')} matches source filter: {source_path}")
        
        logger.info(f"Filtered {len(filtered_chunks)} chunks from {len(file_chunks)} total for model {model_name}")
        return filtered_chunks
    
    def process_all_models(self, file_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process all models from mappings.json"""
        if not self.mappings:
            logger.error("No mappings loaded, cannot process all models")
            return {"error": "No mappings loaded"}
        
        results = {}
        
        for model_name in self.mappings.keys():
            try:
                # Get model specification from data dictionary
                model_spec = self.get_model_spec(model_name)
                if not model_spec:
                    logger.warning(f"Model specification not found for {model_name}, skipping")
                    continue
                
                # Add model name to spec for filtering
                model_spec["model_name"] = model_name
                
                # Extract data for this model
                logger.info(f"Processing model: {model_name}")
                result = self.extract_data(file_chunks, model_spec)
                results[model_name] = result
                
            except Exception as e:
                logger.error(f"Failed to process model {model_name}: {e}")
                results[model_name] = {"error": str(e)}
        
        return results
    
    def get_available_mappings(self) -> List[str]:
        """Get list of available models from mappings file"""
        return list(self.mappings.keys())
    
    def process_single_model(self, file_chunks: List[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
        """Process a single model by name"""
        if not self.mappings:
            logger.error("No mappings loaded, cannot process model")
            return {"error": "No mappings loaded"}
        
        if model_name not in self.mappings:
            logger.error(f"Model {model_name} not found in mappings")
            return {"error": f"Model {model_name} not found in mappings"}
        
        try:
            # Get model specification from data dictionary
            model_spec = self.get_model_spec(model_name)
            if not model_spec:
                logger.error(f"Model specification not found for {model_name}")
                return {"error": f"Model specification not found for {model_name}"}
            
            # Add model name to spec for filtering
            model_spec["model_name"] = model_name
            
            # Extract data for this model
            logger.info(f"Processing single model: {model_name}")
            result = self.extract_data(file_chunks, model_spec)
            return result
            
        except Exception as e:
            logger.error(f"Failed to process model {model_name}: {e}")
            return {"error": str(e)}
    
    def load_chunks_from_file(self, chunks_file: str) -> List[Dict[str, Any]]:
        """Load chunks from JSONL file"""
        chunks = []
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        chunks.append(json.loads(line))
            logger.info(f"Loaded {len(chunks)} chunks from {chunks_file}")
            return chunks
        except Exception as e:
            logger.error(f"Error loading chunks from {chunks_file}: {e}")
            raise
    
    def save_results(self, results: Dict[str, Any], output_file: str) -> None:
        """Save results to JSONL file"""
        try:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                if isinstance(results, dict) and "records" in results:
                    # Single model result
                    self._save_single_model_results(results, f)
                else:
                    # Multiple model results
                    for model_name, model_result in results.items():
                        if isinstance(model_result, dict) and "records" in model_result:
                            self._save_single_model_results(model_result, f, model_name)
                        else:
                            # Error case
                            error_record = {
                                "model": model_name,
                                "file": "unknown",
                                "record": {"error": str(model_result.get("error", "Unknown error"))},
                                "chunk_ids": []
                            }
                            f.write(json.dumps(error_record, ensure_ascii=False) + "\n")
            
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results to {output_file}: {e}")
            raise
    
    def _save_single_model_results(self, model_result: Dict[str, Any], file_handle, model_name: str = None) -> None:
        """Save results for a single model in JSONL format"""
        records = model_result.get("records", [])
        model_name = model_name or model_result.get("model_name", "unknown")
        
        # Get all chunk IDs that were used for this model extraction
        all_chunk_ids = model_result.get("chunk_ids", [])
        
        for record in records:
            # Use all chunk IDs that were sent to the LLM for this extraction
            chunk_ids = all_chunk_ids
            
            # Determine source file from chunk IDs or use default
            source_file = "unknown"
            if chunk_ids:
                # Extract file name from first chunk ID
                first_chunk = chunk_ids[0]
                if "_t" in first_chunk:
                    source_file = first_chunk.split("_t")[0] + ".html"
                elif "_chunk_" in first_chunk:
                    source_file = first_chunk.split("_chunk_")[0] + ".html"
                elif "_" in first_chunk:
                    # Try to extract filename from chunk ID
                    parts = first_chunk.split("_")
                    if len(parts) > 1:
                        source_file = parts[0] + ".html"
            
            # If still unknown, try to get from model mappings
            if source_file == "unknown" and hasattr(self, 'mappings'):
                model_mapping = self.mappings.get(model_name, {})
                filenames = model_mapping.get("filenames", [])
                if filenames:
                    # Use the first filename from mappings
                    source_file = filenames[0].split("\\")[-1]  # Get just the filename
            
            # Create JSONL record
            jsonl_record = {
                "model": model_name,
                "file": f"input_files\\{source_file}",
                "record": record,
                "chunk_ids": chunk_ids
            }
            
            file_handle.write(json.dumps(jsonl_record, ensure_ascii=False) + "\n")


def main():
    """CLI interface for ExtractionAgent"""
    parser = argparse.ArgumentParser(description='Extraction Agent with Source Filtering')
    parser.add_argument('--model', '-m', help='Name of the model to process (optional)')
    parser.add_argument('--chunks', '-c', required=True, help='Path to JSONL file containing chunks')
    parser.add_argument('--mappings', '-p', required=True, help='Path to mappings.json file')
    parser.add_argument('--output', '-o', help='Output file for results (optional)')
    parser.add_argument('--max-tokens', type=int, default=100000, help='Maximum tokens per extraction prompt')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--list-models', action='store_true', help='List available models and exit')
    
    args = parser.parse_args()
    
    # Set up logging - only file logging, no console output
    logging.basicConfig(level=logging.WARNING, handlers=[])
    
    # Validate files exist
    if not Path(args.chunks).exists():
        print(f"Error: Chunks file not found: {args.chunks}")
        sys.exit(1)
    
    if not Path(args.mappings).exists():
        print(f"Error: Mappings file not found: {args.mappings}")
        sys.exit(1)
    
    try:
        # Initialize agent
        print("Initializing extraction agent...")
        agent_config = config.get_agent_config("extractor")
        agent = ExtractionAgent(agent_config=agent_config, mappings_file=args.mappings)
        agent.max_tokens_per_prompt = args.max_tokens  # Set token limit from CLI
        print(f"📝 Logs will be saved to: {agent.log_file}")
        
        # List models if requested
        if args.list_models:
            available_models = agent.get_available_mappings()
            print(f"Available models ({len(available_models)}):")
            for model in available_models:
                print(f"  - {model}")
            return
        
        # Load chunks
        print("Loading chunks...")
        chunks = agent.load_chunks_from_file(args.chunks)
        
        if args.model:
            # Process single model
            print(f"Processing model: {args.model}")
            
            # Check if model exists
            available_models = agent.get_available_mappings()
            if args.model not in available_models:
                print(f"Error: Model '{args.model}' not found in mappings")
                print(f"Available models: {', '.join(available_models)}")
                sys.exit(1)
            
            result = agent.process_single_model(chunks, args.model)
            
            # Check for errors
            if "error" in result:
                print(f"Error processing model: {result['error']}")
                sys.exit(1)
            
            # Display results
            records_count = len(result.get('records', []))
            print(f"✅ Successfully processed model '{args.model}'")
            print(f"📊 Extracted {records_count} records")
            
            # Save results if output file specified
            if args.output:
                agent.save_results(result, args.output)
                print(f"💾 Results saved to: {args.output}")
            
            # Show sample of results if verbose
            if args.verbose and records_count > 0:
                print("\n📋 Sample records:")
                for i, record in enumerate(result.get('records', [])[:3]):
                    print(f"  {i+1}. {record}")
                if records_count > 3:
                    print(f"  ... and {records_count - 3} more records")
        
        else:
            # Process all models
            print("Processing all models...")
            results = agent.process_all_models(chunks)
            
            # Display summary
            total_models = len(results)
            successful_models = sum(1 for r in results.values() if "error" not in r)
            total_records = sum(len(r.get('records', [])) for r in results.values() if "error" not in r)
            
            print(f"✅ Processed {successful_models}/{total_models} models successfully")
            print(f"📊 Total records extracted: {total_records}")
            
            # Show per-model results
            print("\n📋 Per-model results:")
            for model_name, result in results.items():
                if "error" in result:
                    print(f"  ❌ {model_name}: {result['error']}")
                else:
                    records_count = len(result.get('records', []))
                    print(f"  ✅ {model_name}: {records_count} records")
            
            # Save results if output file specified
            if args.output:
                agent.save_results(results, args.output)
                print(f"💾 Results saved to: {args.output}")
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
