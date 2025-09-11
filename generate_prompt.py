#!/usr/bin/env python3
"""
Simple script to generate prompt files for a given model name.
Usage: python generate_prompt.py <model_name> [chunks_file] [source_path_filter]
"""

import sys
import os
import yaml
import json
import math
from prompt_builder import PromptBuilder


def load_data_dictionary(file_path="data_dictionary.yaml"):
    """Load the data dictionary from YAML file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Data dictionary file '{file_path}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML file: {e}")
        sys.exit(1)


def estimate_tokens(text):
    """Rough estimation of token count (approximately 4 characters per token)."""
    return len(text) // 4


def split_chunks_into_batches(chunks, max_tokens_per_batch=1000000):
    """Split chunks into batches that don't exceed the token limit."""
    batches = []
    current_batch = []
    current_tokens = 0
    
    for chunk in chunks:
        chunk_text = chunk.get("text", "")
        chunk_tokens = estimate_tokens(chunk_text)
        
        # If adding this chunk would exceed the limit, start a new batch
        if current_tokens + chunk_tokens > max_tokens_per_batch and current_batch:
            batches.append(current_batch)
            current_batch = [chunk]
            current_tokens = chunk_tokens
        else:
            current_batch.append(chunk)
            current_tokens += chunk_tokens
    
    # Add the last batch if it has chunks
    if current_batch:
        batches.append(current_batch)
    
    return batches


def load_chunks_from_file(chunks_file="output/all_chunks.jsonl"):
    """Load chunks from the JSONL file."""
    chunks = []
    try:
        with open(chunks_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    chunk = json.loads(line.strip())
                    chunks.append(chunk)
        print(f"📄 Loaded {len(chunks)} chunks from {chunks_file}")
        return chunks
    except FileNotFoundError:
        print(f"Warning: Chunks file '{chunks_file}' not found. Using sample chunks.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON in chunks file: {e}")
        return []


def filter_chunks_for_model(chunks, model_spec, custom_source_filter=None):
    """Filter chunks based on model's source path filter or custom filter."""
    # Use custom filter if provided, otherwise use model's filter
    source_filter = custom_source_filter if custom_source_filter is not None else model_spec.get("source_path_filter", [])
    
    if not source_filter:
        print("📝 No source path filter specified, using all chunks")
        return chunks
    
    # Ensure source_filter is a list for consistent handling
    if isinstance(source_filter, str):
        source_filter = [source_filter]
    
    filtered_chunks = []
    for chunk in chunks:
        source_path = chunk.get("source_path", "")
        if any(filter_path in source_path for filter_path in source_filter):
            filtered_chunks.append(chunk)
    
    filter_type = "custom" if custom_source_filter is not None else "model"
    print(f"🔍 Filtered to {len(filtered_chunks)} chunks matching {filter_type} source filter: {source_filter}")
    return filtered_chunks


def generate_prompt_for_model(model_name, data_dict, output_dir="output", chunks_file="output/all_chunks.jsonl", source_path_filter=None, max_tokens_per_batch=1000000):
    """Generate prompt files for the specified model."""
    
    # Check if model exists in data dictionary
    if model_name not in data_dict.get("models", {}):
        available_models = list(data_dict.get("models", {}).keys())
        print(f"Error: Model '{model_name}' not found in data dictionary.")
        print(f"Available models: {', '.join(available_models)}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize prompt builder for extraction
    builder = PromptBuilder(
        model_name=model_name,
        data_dict=data_dict,
        task_type="extraction"
    )
    
    # Load actual chunks from file
    print(f"📚 Loading chunks from {chunks_file}...")
    all_chunks = load_chunks_from_file(chunks_file)
    
    if all_chunks:
        # Filter chunks based on model's source path filter or custom filter
        model_spec = data_dict["models"][model_name]
        filtered_chunks = filter_chunks_for_model(all_chunks, model_spec, source_path_filter)
        
        # Convert chunks to the format expected by PromptBuilder
        formatted_chunks = []
        for chunk in filtered_chunks:
            formatted_chunk = {
                "chunk_id": chunk.get("chunk_id", chunk.get("id", "unknown")),
                "text": chunk.get("text", "")
            }
            # Add metadata if available
            if "metadata" in chunk:
                formatted_chunk["metadata"] = chunk["metadata"]
            formatted_chunks.append(formatted_chunk)
        
        # Split chunks into batches if they exceed token limit
        batches = split_chunks_into_batches(formatted_chunks, max_tokens_per_batch)
        
        if len(batches) > 1:
            print(f"📦 Large dataset detected: splitting into {len(batches)} batches")
            print(f"   - Max tokens per batch: {max_tokens_per_batch:,}")
            print(f"   - Total chunks: {len(formatted_chunks)}")
        
        # Generate prompts for each batch
        for batch_idx, batch_chunks in enumerate(batches):
            if len(batches) > 1:
                print(f"🔨 Building prompt for batch {batch_idx + 1}/{len(batches)} ({len(batch_chunks)} chunks)...")
            
            # Create a new builder for each batch
            batch_builder = PromptBuilder(
                model_name=model_name,
                data_dict=data_dict,
                task_type="extraction"
            )
            batch_builder.add_chunks(batch_chunks)
            
            # Generate JSON schema (same for all batches)
            model_spec = data_dict["models"][model_name]
            fields = model_spec.get("fields", [])
            
            # Build properties for each field
            field_properties = {}
            for field in fields:
                field_name = field["name"]
                field_type = field.get("dtype", "string")
                
                # Map data types to JSON schema types
                type_mapping = {
                    "string": "string",
                    "integer": "integer", 
                    "number": "number",
                    "float": "number",
                    "boolean": "boolean",
                    "array": "array",
                    "object": "object"
                }
                
                json_type = type_mapping.get(field_type, "string")
                field_schema = {"type": json_type}
                
                # Add constraints if specified
                if field.get("enum"):
                    field_schema["enum"] = field["enum"]
                if field.get("regex"):
                    field_schema["pattern"] = field["regex"]
                if field.get("range") and json_type == "number":
                    if len(field["range"]) >= 2:
                        field_schema["minimum"] = field["range"][0]
                        field_schema["maximum"] = field["range"][1]
                
                field_properties[field_name] = field_schema
            
            # Create the complete JSON schema
            json_schema = {
                "type": "object",
                "properties": {
                    "records": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": field_properties,
                            "required": [f["name"] for f in fields if f.get("required", False)]
                        }
                    },
                    "notes": {"type": "string"}
                },
                "required": ["records"]
            }
            
            batch_builder.set_json_schema_for_response(json_schema)
            
            # Build the prompt for this batch
            try:
                prompt = batch_builder.build_prompt()
                
                # Determine output filename
                if len(batches) > 1:
                    output_file = os.path.join(output_dir, f"{model_name}_batch_{batch_idx + 1:03d}.txt")
                else:
                    output_file = os.path.join(output_dir, f"{model_name}.txt")
                
                # Save to file
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(f"# Prompt for Model: {model_name}")
                    if len(batches) > 1:
                        f.write(f" (Batch {batch_idx + 1}/{len(batches)})")
                    f.write("\n")
                    f.write("=" * 50 + "\n\n")
                    
                    # Write system prompt
                    f.write("## System Prompt\n")
                    f.write("-" * 20 + "\n")
                    f.write(prompt[0]['content'])
                    f.write("\n\n")
                    
                    # Write user prompt
                    f.write("## User Prompt\n")
                    f.write("-" * 20 + "\n")
                    f.write(prompt[1]['content'])
                    f.write("\n\n")
                    
                    # Add model information
                    f.write("## Model Information\n")
                    f.write("-" * 20 + "\n")
                    model_info = data_dict["models"][model_name]
                    f.write(f"Description: {model_info.get('description', 'N/A')}\n")
                    f.write(f"Primary Key: {', '.join(model_info.get('primary_key', []))}\n")
                    f.write(f"Field Count: {len(model_info.get('fields', []))}\n")
                    f.write(f"Required Fields: {len([f for f in model_info.get('fields', []) if f.get('required', False)])}\n")
                    
                    if len(batches) > 1:
                        f.write(f"Batch: {batch_idx + 1}/{len(batches)}\n")
                        f.write(f"Chunks in this batch: {len(batch_chunks)}\n")
                
                print(f"✅ Prompt generated successfully: {output_file}")
                
                # Show stats for this batch
                stats = batch_builder.get_stats()
                print(f"📊 Batch {batch_idx + 1} Statistics:")
                print(f"   - Fields: {stats['field_count']}")
                print(f"   - Required fields: {stats['required_field_count']}")
                print(f"   - Chunks: {stats['chunk_count']}")
                print(f"   - Evidence mode: {stats['evidence_mode']}")
                
            except Exception as e:
                print(f"Error generating prompt for batch {batch_idx + 1}: {e}")
                sys.exit(1)
        
        # Summary for multiple batches
        if len(batches) > 1:
            print(f"\n📈 Overall Summary:")
            print(f"   - Total batches: {len(batches)}")
            print(f"   - Total chunks: {len(formatted_chunks)}")
            print(f"   - Model: {model_name}")
            print(f"   - Output directory: {output_dir}")
    else:
        # Fallback to sample chunks if no chunks file found
        print("⚠️  Using sample chunks as fallback")
        sample_chunks = [
            {
                "chunk_id": "sample_1",
                "text": "This is a sample chunk for testing prompt generation. It contains sample data that would typically be extracted from documents."
            },
            {
                "chunk_id": "sample_2", 
                "text": "Another sample chunk with different content to demonstrate how the prompt handles multiple chunks of data."
            }
        ]
        builder.add_chunks(sample_chunks)
    
    # Handle fallback case for sample chunks (single batch)
    if not all_chunks:
        # Generate JSON schema based on model fields
        model_spec = data_dict["models"][model_name]
        fields = model_spec.get("fields", [])
        
        # Build properties for each field
        field_properties = {}
        for field in fields:
            field_name = field["name"]
            field_type = field.get("dtype", "string")
            
            # Map data types to JSON schema types
            type_mapping = {
                "string": "string",
                "integer": "integer", 
                "number": "number",
                "float": "number",
                "boolean": "boolean",
                "array": "array",
                "object": "object"
            }
            
            json_type = type_mapping.get(field_type, "string")
            field_schema = {"type": json_type}
            
            # Add constraints if specified
            if field.get("enum"):
                field_schema["enum"] = field["enum"]
            if field.get("regex"):
                field_schema["pattern"] = field["regex"]
            if field.get("range") and json_type == "number":
                if len(field["range"]) >= 2:
                    field_schema["minimum"] = field["range"][0]
                    field_schema["maximum"] = field["range"][1]
            
            field_properties[field_name] = field_schema
        
        # Create the complete JSON schema
        json_schema = {
            "type": "object",
            "properties": {
                "records": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": field_properties,
                        "required": [f["name"] for f in fields if f.get("required", False)]
                    }
                },
                "notes": {"type": "string"}
            },
            "required": ["records"]
        }
        
        builder.set_json_schema_for_response(json_schema)
        
        # Build the prompt
        try:
            prompt = builder.build_prompt()
            
            # Save to file
            output_file = os.path.join(output_dir, f"{model_name}.txt")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"# Prompt for Model: {model_name}\n")
                f.write("=" * 50 + "\n\n")
                
                # Write system prompt
                f.write("## System Prompt\n")
                f.write("-" * 20 + "\n")
                f.write(prompt[0]['content'])
                f.write("\n\n")
                
                # Write user prompt
                f.write("## User Prompt\n")
                f.write("-" * 20 + "\n")
                f.write(prompt[1]['content'])
                f.write("\n\n")
                
                # Add model information
                f.write("## Model Information\n")
                f.write("-" * 20 + "\n")
                model_info = data_dict["models"][model_name]
                f.write(f"Description: {model_info.get('description', 'N/A')}\n")
                f.write(f"Primary Key: {', '.join(model_info.get('primary_key', []))}\n")
                f.write(f"Field Count: {len(model_info.get('fields', []))}\n")
                f.write(f"Required Fields: {len([f for f in model_info.get('fields', []) if f.get('required', False)])}\n")
            
            print(f"✅ Prompt generated successfully: {output_file}")
            print(f"📊 Model: {model_name}")
            print(f"📁 Output directory: {output_dir}")
            
            # Show some stats
            stats = builder.get_stats()
            print(f"📈 Statistics:")
            print(f"   - Fields: {stats['field_count']}")
            print(f"   - Required fields: {stats['required_field_count']}")
            print(f"   - Chunks: {stats['chunk_count']}")
            print(f"   - Evidence mode: {stats['evidence_mode']}")
            
        except Exception as e:
            print(f"Error generating prompt: {e}")
            sys.exit(1)


def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 2 or len(sys.argv) > 5:
        print("Usage: python generate_prompt.py <model_name> [chunks_file] [source_path_filter] [max_tokens_per_batch]")
        print("\nExamples:")
        print("  python generate_prompt.py canadian_deminimis_thresholds")
        print("  python generate_prompt.py canadian_deminimis_thresholds ca_output/all_chunks.jsonl")
        print("  python generate_prompt.py canadian_deminimis_thresholds ca_output/all_chunks.jsonl canadian_tax_rates")
        print("  python generate_prompt.py provinces_stg ca_output/all_chunks.jsonl countries-pays-eng")
        print("  python generate_prompt.py provinces_stg ca_output/all_chunks.jsonl canadian_tax_rates 500000")
        print("\nParameters:")
        print("  model_name: Name of the model from data dictionary")
        print("  chunks_file: Path to chunks JSONL file (default: output/all_chunks.jsonl)")
        print("  source_path_filter: Filter chunks by source path substring (optional)")
        print("  max_tokens_per_batch: Maximum tokens per batch (default: 1000000)")
        sys.exit(1)
    
    model_name = sys.argv[1]
    chunks_file = sys.argv[2] if len(sys.argv) >= 3 else "output/all_chunks.jsonl"
    source_path_filter = sys.argv[3] if len(sys.argv) >= 4 else None
    max_tokens_per_batch = int(sys.argv[4]) if len(sys.argv) == 5 else 1000000
    
    print(f"🚀 Generating prompt for model: {model_name}")
    print(f"📁 Using chunks file: {chunks_file}")
    if source_path_filter:
        print(f"🔍 Filtering by source path: {source_path_filter}")
    print(f"📊 Max tokens per batch: {max_tokens_per_batch:,}")
    
    # Load data dictionary
    print("📖 Loading data dictionary...")
    data_dict = load_data_dictionary()
    
    # Generate prompt
    print("🔨 Building prompt...")
    generate_prompt_for_model(model_name, data_dict, chunks_file=chunks_file, source_path_filter=source_path_filter, max_tokens_per_batch=max_tokens_per_batch)


if __name__ == "__main__":
    main()
