# LLM Parser Pipeline

A complete pipeline for extracting structured data from unstructured documents using LLMs with an intelligent agent-based architecture and interactive Streamlit interface.

## Overview

This pipeline consists of two main interfaces:

### 🤖 Agent-Based Pipeline
An intelligent multi-agent system that processes documents through specialized AI agents:

1. **ChunkingStep** - Breaks down input files (PDF, HTML, CSV, etc.) into manageable chunks
2. **ClassifierAgent** - Maps document chunks to appropriate data models using intelligent classification
3. **ExtractionAgent** - Uses LLM to extract structured data from chunks according to a schema
4. **ValidationAgent** - Validates extracted records against the original chunks for accuracy

### 🖥️ Streamlit Web Interface
An interactive web application that provides a user-friendly interface for running the complete pipeline with real-time progress tracking and result visualization.

## Key Features

- **Multi-Agent Architecture**: Specialized AI agents for each processing step
- **Interactive Web Interface**: Streamlit app for easy pipeline execution and monitoring
- **Country-Specific Configuration**: Support for multiple countries (AU, CA, US, UK) with custom defaults
- **Intelligent Classification**: Automatic mapping of documents to appropriate data models
- **Real-time Progress Tracking**: Visual feedback during processing
- **Evidence-Based Validation**: Comprehensive validation with source chunk references
- **Flexible Input Support**: PDF, HTML, CSV, Excel, and other document formats

## Quick Start

### Prerequisites

#### 1. Python Environment
Ensure you have Python 3.8+ installed. We recommend using a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### 2. Install Dependencies
Install all required packages using the consolidated requirements file:

```bash
pip install -r requirements.txt
```

**Note**: This single requirements.txt file contains all dependencies for both the agent-based pipeline and Streamlit interface, including:
- Streamlit for the web interface
- LangChain for AI agent functionality
- Document processing libraries (unstructured, pypdf, etc.)
- Data processing tools (pandas, numpy)
- LLM integration (OpenAI, tiktoken)
- Vector stores (ChromaDB, FAISS)
- Development tools (pytest, black, flake8)

#### 3. Set OpenAI API Key
Set your OpenAI API key as an environment variable:

```bash
# On Windows (Command Prompt):
set OPENAI_API_KEY=your-api-key-here

# On Windows (PowerShell):
$env:OPENAI_API_KEY="your-api-key-here"

# On macOS/Linux:
export OPENAI_API_KEY="your-api-key-here"
```

Alternatively, create a `.env` file in the project root:
```
OPENAI_API_KEY=your-api-key-here
```

### 🖥️ Streamlit Web Interface (Recommended)

The easiest way to use the pipeline is through the interactive Streamlit web interface:

```bash
streamlit run streamlit_app.py
```

This will open a web interface in your browser where you can:

1. **Configure Settings**: Select input/output directories and country code
2. **Step-by-Step Processing**: Run each pipeline step individually with real-time feedback
3. **Interactive File Selection**: Choose specific files to process or process all files
4. **Live Progress Tracking**: See real-time status updates and progress indicators
5. **Result Visualization**: Preview extracted data, validation results, and generated prompts
6. **Prompt Inspection**: View and analyze the prompts sent to each AI agent

#### Streamlit Features:
- **Sidebar Configuration**: Easy setup of input/output directories and country settings
- **Step-by-Step Workflow**: Process documents through chunking → classification → extraction → validation
- **Interactive File Browser**: Select specific files or process entire directories
- **Real-time Feedback**: Live status updates and progress indicators
- **Data Preview**: Interactive preview of chunks, records, and validation results
- **Prompt Analysis**: View generated prompts for transparency and debugging

### 🤖 Command Line Interface

For automated processing or integration into other systems, use the agent-based CLI:

```bash
# Full pipeline
python -m data_extraction_agent.pipeline_orchestrator --input input_files/ --model deminimis_rules_stg --country CA

# Step-by-step processing
python -m data_extraction_agent.pipeline_orchestrator --input input_files/ --model deminimis_rules_stg --country CA --steps chunking classification extraction validation

# Validation only (using existing files)
python -m data_extraction_agent.pipeline_orchestrator --model deminimis_rules_stg --country CA --records output/records.jsonl --chunks output/all_chunks.jsonl
```

This will:
1. Process all files in `input_files/` directory
2. Create chunks in `agent_output/all_chunks.jsonl`
3. Classify chunks and create mappings in `agent_output/mappings.json`
4. Extract records using the specified model
5. Save records to `agent_output/records.jsonl`
6. Validate records and save results to `agent_output/validation.jsonl`
7. Generate a summary in `agent_output/validation_summary.json`

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
- `tax_types_stg` - Tax types by country
- `province_tax_rates_stg` - Province-level tax rates
- `treatments_stg` - Tariff treatment codes
- `origin_groups_stg` - Origin group mappings
- `country_origin_groups_stg` - Country to origin group mappings
- `treatment_eligibilities_stg` - Treatment eligibility mappings
- `duty_rates_stg` - Duty rates by HS code
- `origin_bands_stg` - Origin bands for de-minimis logic
- `hs_codes_stg` - Harmonized System codes

## Agent Components

### 🧠 ClassifierAgent
Intelligently maps document chunks to appropriate data models based on content analysis.

**Features:**
- Content-based classification using AI
- Confidence scoring for mapping decisions
- Support for multiple table mappings per file
- Customizable classification rules

**Usage:**
```python
from data_extraction_agent.agents.classifier import ClassifierAgent
from data_extraction_agent.config import AgentConfig

config = AgentConfig(country="CA")
classifier = ClassifierAgent(config.get_agent_config("classifier"))
result = classifier.process({"chunks": chunks, "document_context": "Trade data"})
```

### 🔍 ExtractionAgent
Extracts structured data from classified chunks using sophisticated prompt engineering.

**Features:**
- Schema-driven extraction with validation
- Evidence tracking for each extracted field
- Batch processing for large datasets
- Support for complex data relationships

**Usage:**
```python
from data_extraction_agent.agents.extraction_agent import ExtractionAgent

extractor = ExtractionAgent(
    agent_config=config.get_agent_config("extractor"),
    mappings_file="mappings.json"
)
result = extractor.process_single_model(chunks, "deminimis_rules_stg")
```

### ✅ ValidationAgent
Validates extracted records against source chunks with comprehensive evidence analysis.

**Features:**
- Evidence-based validation (V/I/M/P status codes)
- Chunk-level evidence citations
- Business rule validation
- Detailed validation reporting

**Usage:**
```python
from data_extraction_agent.agents.validation_agent import ValidationAgent

validator = ValidationAgent(config.get_agent_config("validator"))
summary = validator.validate_records_file(
    records_file="records.jsonl",
    chunks_file="all_chunks.jsonl",
    output_dir="output"
)
```

## Advanced Usage

### Country-Specific Configuration

The pipeline supports country-specific defaults and configurations:

```bash
# Use Canadian defaults
python -m data_extraction_agent.pipeline_orchestrator --country CA --input ca_input_files/

# Use Australian defaults  
python -m data_extraction_agent.pipeline_orchestrator --country AU --input au_input_files/

# Use US defaults
python -m data_extraction_agent.pipeline_orchestrator --country US --input us_input_files/

# Use UK defaults
python -m data_extraction_agent.pipeline_orchestrator --country UK --input uk_input_files/
```

### Agent Pipeline Options

```bash
# Custom Output Directory
python -m data_extraction_agent.pipeline_orchestrator --input input_files/ --model deminimis_rules_stg --country CA --output-dir my_results/

# Specific Steps Only
python -m data_extraction_agent.pipeline_orchestrator --input input_files/ --model deminimis_rules_stg --country CA --steps chunking classification

# Custom Chunking Parameters
python -m data_extraction_agent.pipeline_orchestrator --input input_files/ --model deminimis_rules_stg --country CA --chunk-size 1500 --chunk-overlap 300

# Custom Token Limits
python -m data_extraction_agent.pipeline_orchestrator --input input_files/ --model deminimis_rules_stg --country CA --max-tokens 100000

# Verbose Logging
python -m data_extraction_agent.pipeline_orchestrator --input input_files/ --model deminimis_rules_stg --country CA --verbose
```

### Streamlit Advanced Features

The Streamlit interface supports advanced configuration through the sidebar:

- **Input Directory Selection**: Choose from available input directories
- **Output Directory Configuration**: Set custom output paths
- **Country Code Selection**: Switch between supported countries
- **File-Specific Processing**: Select individual files for processing
- **Model Selection**: Choose specific data models for extraction
- **Real-time Monitoring**: Track processing progress and errors

## Output Files

### Agent Pipeline Outputs
- `all_chunks.jsonl` - Text chunks extracted from input files
- `mappings.json` - File-to-table mappings from ClassifierAgent
- `records.jsonl` - Structured records extracted by ExtractionAgent
- `validation.jsonl` - Per-record validation results with evidence
- `validation_summary.json` - Summary statistics and common failures

### Streamlit Outputs
The Streamlit interface generates the same output files but provides additional features:
- **Interactive Previews**: Real-time preview of chunks, records, and validation results
- **Filtered Views**: Browse data by source file or model
- **Progress Tracking**: Visual indicators for each processing step
- **Error Reporting**: Detailed error messages and troubleshooting guidance

## Individual Agent Usage

### ChunkingStep
```python
from data_extraction_agent.chunking_step import ChunkingStep

chunker = ChunkingStep(chunk_size=1200, chunk_overlap=200)
result = chunker.process_directory("input_files/", "output/chunks.jsonl")
```

### ClassifierAgent
```python
from data_extraction_agent.agents.classifier import ClassifierAgent
from data_extraction_agent.config import AgentConfig

config = AgentConfig(country="CA")
classifier = ClassifierAgent(config.get_agent_config("classifier"))
result = classifier.process({"chunks": chunks, "document_context": "Trade data"})
```

### ExtractionAgent
```python
from data_extraction_agent.agents.extraction_agent import ExtractionAgent

extractor = ExtractionAgent(
    agent_config=config.get_agent_config("extractor"),
    mappings_file="mappings.json"
)
result = extractor.process_single_model(chunks, "deminimis_rules_stg")
```

### ValidationAgent
```python
from data_extraction_agent.agents.validation_agent import ValidationAgent

validator = ValidationAgent(config.get_agent_config("validator"))
summary = validator.validate_records_file(
    records_file="records.jsonl",
    chunks_file="all_chunks.jsonl",
    output_dir="output"
)
```

## Configuration

### Data Dictionary
The pipeline uses a YAML data dictionary (`data_dictionary.yaml`) that defines:
- Field schemas and validation rules
- Source file filters
- Constraints and business rules
- Evidence requirements
- Model specifications and extraction perspectives

### Country-Specific Defaults
The pipeline supports country-specific configurations in the `defaults/` directory:

- `CA.json` - Canadian defaults (CAD currency, en-CA formatting)
- `AU.json` - Australian defaults (AUD currency, en-AU formatting)
- `US.json` - US defaults (USD currency, en-US formatting)
- `UK.json` - UK defaults (GBP currency, en-GB formatting)

Each defaults file contains:
- Country code and currency
- Language and timezone settings
- Date and number formatting preferences
- Model-specific parameters

### Agent Configuration
Each agent can be configured through the `AgentConfig` class:

```python
from data_extraction_agent.config import AgentConfig

# Country-specific configuration
config = AgentConfig(country="CA")

# Get agent-specific settings
classifier_config = config.get_agent_config("classifier")
extractor_config = config.get_agent_config("extractor")
validator_config = config.get_agent_config("validator")
```

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

### Agent-Based Pipeline
```
Input Files → ChunkingStep → Chunks → ClassifierAgent → Mappings → ExtractionAgent → Records → ValidationAgent → Validation Results
     ↓            ↓           ↓           ↓              ↓            ↓              ↓           ↓                    ↓
  PDF/HTML/   LangChain    JSONL     AI Classification  JSON      AI Extraction   JSONL    AI Validation      JSONL +
   CSV/TXT    Splitters    Chunks    + Confidence       Mappings  + Evidence       Records  + Evidence         Summary
                    ↓           ↓              ↓            ↓              ↓           ↓                    ↓
              PromptBuilder  PromptBuilder  PromptBuilder  PromptBuilder  PromptBuilder  PromptBuilder
              (Chunking)    (Classification) (Extraction)  (Validation)   (Evidence)     (Reporting)
```

### Streamlit Interface
```
Web Browser → Streamlit App → Agent Pipeline → Real-time Updates → Interactive Results
     ↓             ↓              ↓                ↓                    ↓
  User Input   Configuration   Step-by-Step    Progress Tracking    Data Preview
  & Control    & File Select   Processing      & Error Handling    & Analysis
```

### Key Features

- **Multi-Agent Architecture**: Specialized AI agents for each processing step
- **Intelligent Classification**: Automatic document-to-model mapping
- **Evidence-Based Processing**: Full traceability from source to output
- **Interactive Web Interface**: User-friendly Streamlit application
- **Country-Specific Configuration**: Support for multiple jurisdictions
- **Real-time Monitoring**: Live progress tracking and error reporting
- **Flexible Processing**: Both automated and interactive modes

### Agent Responsibilities

1. **ChunkingStep**: Document parsing and text chunking
2. **ClassifierAgent**: Content analysis and model mapping
3. **ExtractionAgent**: Structured data extraction with evidence
4. **ValidationAgent**: Quality assurance and validation reporting

The pipeline is designed to be:
- **Modular**: Each agent can run independently
- **Configurable**: Country-specific and model-specific settings
- **Robust**: Comprehensive error handling and fallback methods
- **Traceable**: Full evidence tracking throughout the pipeline
- **User-Friendly**: Both CLI and web interfaces available

## Project Structure

```
LLM Parser/
├── data_extraction_agent/          # Core agent-based pipeline
│   ├── agents/                     # AI agents
│   │   ├── classifier.py          # Document classification agent
│   │   ├── extraction_agent.py    # Data extraction agent
│   │   └── validation_agent.py    # Validation agent
│   ├── chunking_step.py           # Document chunking
│   ├── config.py                  # Configuration management
│   ├── pipeline_orchestrator.py   # Main orchestrator
│   └── prompt_builder.py          # Prompt engineering
├── defaults/                       # Country-specific defaults
│   ├── CA.json                    # Canadian defaults
│   ├── AU.json                    # Australian defaults
│   ├── US.json                    # US defaults
│   └── UK.json                    # UK defaults
├── prompt_templates/               # AI prompt templates
│   ├── extraction.txt             # Extraction prompts
│   ├── validation.txt             # Validation prompts
│   └── mapper.txt                 # Classification prompts
├── *_input_files/                 # Country-specific input data
├── streamlit_app.py               # Web interface
├── data_dictionary.yaml           # Schema definitions
└── requirements.txt               # Consolidated dependencies (all-in-one)
```

## Getting Started with Streamlit

1. **Setup Environment** (if not already done):
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Set API key
   export OPENAI_API_KEY="your-api-key-here"  # On Windows: set OPENAI_API_KEY=your-api-key-here
   ```

2. **Launch Web Interface**:
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Configure and Process**:
   - Select input directory (e.g., `au_input_files/`)
   - Choose output directory (e.g., `au_output/`)
   - Select country code (AU, CA, US, UK)
   - Click "Start Workflow"
   - Follow the step-by-step process

## Getting Started with CLI

1. **Setup Environment** (if not already done):
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Set API key
   export OPENAI_API_KEY="your-api-key-here"  # On Windows: set OPENAI_API_KEY=your-api-key-here
   ```

2. **Basic Pipeline**:
   ```bash
   python -m data_extraction_agent.pipeline_orchestrator --input au_input_files/ --model deminimis_rules_stg --country AU
   ```

3. **Step-by-Step Processing**:
   ```bash
   python -m data_extraction_agent.pipeline_orchestrator --input au_input_files/ --model deminimis_rules_stg --country AU --steps chunking classification extraction validation
   ```

4. **Validation Only**:
   ```bash
   python -m data_extraction_agent.pipeline_orchestrator --model deminimis_rules_stg --country AU --records output/records.jsonl --chunks output/all_chunks.jsonl
   ```

## Installation Troubleshooting

### Common Issues

1. **Unstructured Installation Issues**:
   ```bash
   # If unstructured installation fails, try:
   pip install --upgrade pip
   pip install unstructured[all-docs] --no-cache-dir
   ```

2. **ChromaDB Installation Issues**:
   ```bash
   # If ChromaDB installation fails on Windows:
   pip install chromadb --no-cache-dir
   ```

3. **Memory Issues with Large Documents**:
   ```bash
   # Use smaller chunk sizes
   python -m data_extraction_agent.pipeline_orchestrator --input input_files/ --model deminimis_rules_stg --country CA --chunk-size 800 --chunk-overlap 100
   ```

4. **API Rate Limits**:
   - Use a different OpenAI model: `--llm-model gpt-3.5-turbo`
   - Add delays between requests in the agent configuration
   - Process smaller batches of files 