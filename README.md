# LLM Parser Pipeline

A complete pipeline for extracting structured data from unstructured documents using LLMs.

## Overview

This pipeline consists of three main components:

1. **Chunker** (`chunker.py`) - Breaks down input files (PDF, HTML, CSV, etc.) into manageable chunks
2. **Extractor** (`extractor.py`) - Uses LLM to extract structured data from chunks according to a schema
3. **Validator** (`validator.py`) - Validates extracted records against the original chunks for accuracy

## Quick Start

### Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Basic Usage

Run the complete pipeline using the orchestrator:

```bash
python orchestrator.py --input input_files/ --dict data_dictionary.yaml --model deminimis_rules_stg
```

This will:
1. Process all files in `input_files/` directory
2. Create chunks in `output/all_chunks.jsonl`
3. Extract records using the `deminimis_rules_stg` model
4. Save records to `output/records.jsonl`
5. Validate records and save results to `output/validation.jsonl`
6. Generate a summary in `output/validation_summary.json`

### Instructor Integration (Enhanced)

The pipeline now supports Instructor integration for improved type safety and automatic validation:

```bash
# Extract with Instructor (recommended)
python extractor.py --dict data_dictionary.yaml --model deminimis_rules_stg --chunks output/all_chunks.jsonl --use-instructor

# Validate with Instructor (recommended)
python validator.py --dict data_dictionary.yaml --model deminimis_rules_stg --chunks output/all_chunks.jsonl --records output/records.jsonl --use-instructor
```

**Benefits of Instructor:**
- ✅ **Type Safety**: Pydantic models with automatic validation
- ✅ **Automatic Retries**: No more manual error handling
- ✅ **Better Error Messages**: Clear, actionable validation errors
- ✅ **Backward Compatibility**: Works alongside existing system

## Available Models

The pipeline supports multiple models defined in `data_dictionary.yaml`:

- `deminimis_rules_stg` - De-minimis threshold rules
- `countries_stg` - Country master data
- `currencies_stg` - Currency master data
- `provinces_stg` - Province/state data
- `treatments_stg` - Tariff treatment codes
- `duty_rates_stg` - Duty rates by HS code
- `origin_groups_stg` - Origin group mappings

## Advanced Usage

### Instructor Integration Options

```bash
# Use Instructor for both extraction and validation
python extractor.py --dict data_dictionary.yaml --model deminimis_rules_stg --chunks output/all_chunks.jsonl --use-instructor
python validator.py --dict data_dictionary.yaml --model deminimis_rules_stg --chunks output/all_chunks.jsonl --records output/records.jsonl --use-instructor

# Use legacy approach (default)
python extractor.py --dict data_dictionary.yaml --model deminimis_rules_stg --chunks output/all_chunks.jsonl
python validator.py --dict data_dictionary.yaml --model deminimis_rules_stg --chunks output/all_chunks.jsonl --records output/records.jsonl
```

### Orchestrator Options

```bash
# Test Mode (Single File)
python orchestrator.py --input input_files/ --dict data_dictionary.yaml --model deminimis_rules_stg --test-single

# Custom Output Directory
python orchestrator.py --input input_files/ --dict data_dictionary.yaml --model deminimis_rules_stg --output-dir my_results/

# Skip Validation Step
python orchestrator.py --input input_files/ --dict data_dictionary.yaml --model deminimis_rules_stg --skip-validation

# Verbose Logging
python orchestrator.py --input input_files/ --dict data_dictionary.yaml --model deminimis_rules_stg --verbose

# Custom LLM Model
python orchestrator.py --input input_files/ --dict data_dictionary.yaml --model deminimis_rules_stg --llm-model gpt-4o

# Limit CSV Processing
python orchestrator.py --input input_files/ --dict data_dictionary.yaml --model deminimis_rules_stg --max-csv-rows 100
```

## Output Files

- `all_chunks.jsonl` - Text chunks extracted from input files
- `records.jsonl` - Structured records extracted by the LLM
- `validation.jsonl` - Per-record validation results with evidence
- `validation_summary.json` - Summary statistics and common failures

## Individual Components

### Chunker
```bash
python chunker.py input_files/ --out output/chunks.jsonl --chunk-size 1200 --chunk-overlap 200
```

### Extractor

**Legacy Approach (Default):**
```bash
python extractor.py --dict data_dictionary.yaml --model deminimis_rules_stg --chunks output/chunks.jsonl --out-records output/records.jsonl
```

**Instructor Approach (Recommended):**
```bash
python extractor.py --dict data_dictionary.yaml --model deminimis_rules_stg --chunks output/chunks.jsonl --out-records output/records.jsonl --use-instructor
```

### Validator

**Legacy Approach (Default):**
```bash
python validator.py --dict data_dictionary.yaml --model deminimis_rules_stg --chunks output/chunks.jsonl --records output/records.jsonl --out output/validation.jsonl
```

**Instructor Approach (Recommended):**
```bash
python validator.py --dict data_dictionary.yaml --model deminimis_rules_stg --chunks output/chunks.jsonl --records output/records.jsonl --out output/validation.jsonl --use-instructor
```

## Configuration

The pipeline uses a YAML data dictionary that defines:
- Field schemas and validation rules
- Source file filters
- Constraints and business rules
- Evidence requirements

See `data_dictionary.yaml` for the complete schema definition.

## Troubleshooting

### Common Issues

1. **Unstructured library hangs**: Use `--skip-unstructured` flag
2. **Token limits exceeded**: Reduce `--max-tokens-per-batch`
3. **Memory issues**: Use `--test-single` or limit CSV rows
4. **API rate limits**: Use a different LLM model or add delays

### Logs

Check the following log files for detailed information:
- `orchestrator.log` - Orchestrator execution logs
- `extractor.log` - Chunker and extraction logs
- `test.log` - Test execution logs

## Architecture

```
Input Files → Chunker → Chunks → Extractor → Records → Validator → Validation Results
     ↓           ↓        ↓         ↓         ↓         ↓
  PDF/HTML/   LangChain  JSONL    LLM      JSONL     LLM +     JSONL +
   CSV/TXT    Splitters  Chunks  Extraction Records  Evidence  Summary
                    ↓         ↓         ↓
              PromptBuilder  Instructor  Instructor
              (Text Prompts) (Type Safety) (Auto Retries)
```

### Key Features

- **PromptBuilder**: Sophisticated prompt engineering with evidence tracking
- **Instructor Integration**: Pydantic models with automatic validation and retries
- **Dual Mode**: Legacy and Instructor approaches for gradual migration
- **Evidence Tracking**: Chunk references and snippets for every extracted field
- **Type Safety**: Automatic validation with clear error messages

The pipeline is designed to be:
- **Modular**: Each component can run independently
- **Configurable**: Extensive options for different use cases
- **Robust**: Fallback processing methods and error handling
- **Traceable**: Evidence tracking and validation at each step
- **Type-Safe**: Pydantic validation with automatic retries (Instructor mode) 