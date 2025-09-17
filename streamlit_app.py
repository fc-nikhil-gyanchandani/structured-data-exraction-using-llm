import streamlit as st
import json
import sys
from pathlib import Path
import pandas as pd

# Add the project root to the Python path to allow for correct module imports
sys.path.append(str(Path(__file__).parent.resolve()))

from data_extraction_agent.chunking_step import ChunkingStep
from data_extraction_agent.agents.classifier import ClassifierAgent
from data_extraction_agent.agents.extraction_agent import ExtractionAgent
from data_extraction_agent.agents.validation_agent import ValidationAgent
from data_extraction_agent.config import AgentConfig
from data_extraction_agent.prompt_builder import PromptBuilder

# --- Page Configuration ---
st.set_page_config(
    page_title="Data Extraction Workflow",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State Initialization ---
# Initialize session state variables to track the workflow progress
if 'workflow_started' not in st.session_state:
    st.session_state.workflow_started = False
if 'chunker_done' not in st.session_state:
    st.session_state.chunker_done = False
if 'classifier_done' not in st.session_state:
    st.session_state.classifier_done = False
if 'extractor_done' not in st.session_state:
    st.session_state.extractor_done = False
if 'validator_done' not in st.session_state:
    st.session_state.validator_done = False

# To store the paths to the generated files
if 'output_files' not in st.session_state:
    st.session_state.output_files = {
        "chunks": None,
        "mappings": None,
        "records": None,
        "validation_summary": None,
        "validation": None
    }

# To store the generated prompts
if 'prompts' not in st.session_state:
    st.session_state.prompts = {
        "classifier": "Not generated yet.",
        "extractor": "Not generated yet.",
        "validator": "Not generated yet."
    }

# To store selected file for chunking
if 'selected_chunking_file' not in st.session_state:
    st.session_state.selected_chunking_file = None


def show_file_preview(file_path: Path, num_lines: int = 5):
    """Prints a preview of a JSON or JSONL file to the console."""
    st.markdown(f"**📄 Preview of `{file_path.name}`**")
    if not file_path.exists():
        st.warning("File does not exist yet.")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.suffix == ".jsonl":
                records_to_show = []
                for i, line in enumerate(f):
                    if i >= num_lines:
                        break
                    records_to_show.append(json.loads(line))
                
                for i, record in enumerate(records_to_show):
                    with st.expander(f"Record {i + 1}", expanded=(i==0)):
                        st.json(record)
                if len(records_to_show) == num_lines:
                     st.info(f"Showing first {num_lines} records...")

            elif file_path.suffix == ".json":
                content = json.load(f)
                st.json(content)
    except Exception as e:
        st.error(f"Could not read or parse file: {e}")


def show_filtered_chunk_preview(file_path: Path, num_lines: int = 5):
    """Shows a dropdown to filter chunks by source file and previews them."""
    st.markdown(f"**📄 Chunk Browser for `{file_path.name}`**")
    if not file_path.exists():
        st.warning("File does not exist yet.")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            all_chunks = [json.loads(line) for line in f]
        
        source_files = sorted(list(set(chunk.get("source_path") for chunk in all_chunks)))

        if not source_files:
            st.warning("No source files found in the chunks file.")
            return

        selected_file = st.selectbox(
            "Select a source file to view its chunks:",
            options=source_files,
            format_func=lambda x: Path(x).name  # Show only filename in dropdown
        )

        if selected_file:
            filtered_chunks = [chunk for chunk in all_chunks if chunk.get("source_path") == selected_file]
            st.info(f"Displaying {len(filtered_chunks)} chunks for `{Path(selected_file).name}`.")

            # Display preview
            for i, chunk in enumerate(filtered_chunks[:num_lines]):
                with st.expander(f"Chunk {i + 1} (ID: {chunk.get('chunk_id', 'N/A')})", expanded=(i==0)):
                    st.json(chunk)
            
            if len(filtered_chunks) > num_lines:
                st.info(f"Showing first {num_lines} of {len(filtered_chunks)} chunks.")

    except Exception as e:
        st.error(f"Could not read or parse chunks file: {e}")


def show_filtered_records_preview(file_path: Path, num_lines: int = 5):
    """Shows a dropdown to filter records by model name and previews them."""
    st.markdown(f"**📄 Records Browser for `{file_path.name}`**")
    if not file_path.exists():
        st.warning("File does not exist yet.")
        return
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            all_records = [json.loads(line) for line in f]
        
        model_names = sorted(list(set(record.get("model") for record in all_records)))

        if not model_names:
            st.warning("No models found in the records file.")
            return

        selected_model = st.selectbox(
            "Select a model to view its records:",
            options=model_names,
        )

        if selected_model:
            filtered_records = [record for record in all_records if record.get("model") == selected_model]
            st.info(f"Displaying {len(filtered_records)} records for model `{selected_model}`.")

            for i, record_data in enumerate(filtered_records[:num_lines]):
                with st.expander(f"Record {i + 1}", expanded=(i==0)):
                    st.json(record_data)
            
            if len(filtered_records) > num_lines:
                st.info(f"Showing first {num_lines} of {len(filtered_records)} records.")

    except Exception as e:
        st.error(f"Could not read or parse records file: {e}")




def show_filtered_validation_preview(file_path: Path, num_lines: int = 5):
    """Shows a dropdown to filter validation results by model name."""
    st.markdown(f"**📄 Validation Browser for `{file_path.name}`**")
    if not file_path.exists():
        st.warning("File does not exist yet.")
        return
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            all_validations = [json.loads(line) for line in f]
        
        # The model name is nested inside the 'record' dictionary
        model_names = sorted(list(set(v.get("record", {}).get("model") for v in all_validations)))

        if not model_names:
            st.warning("No models found in the validation file.")
            return

        selected_model = st.selectbox(
            "Select a model to view its validation results:",
            options=model_names,
        )

        if selected_model:
            filtered_validations = [v for v in all_validations if v.get("record", {}).get("model") == selected_model]
            st.info(f"Displaying {len(filtered_validations)} validation results for model `{selected_model}`.")

            for i, val_data in enumerate(filtered_validations[:num_lines]):
                with st.expander(f"Validation Result {i + 1}", expanded=(i==0)):
                    st.json(val_data)
            
            if len(filtered_validations) > num_lines:
                st.info(f"Showing first {num_lines} of {len(filtered_validations)} results.")

    except Exception as e:
        st.error(f"Could not read or parse validation file: {e}")


# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    input_dir = st.text_input("Input Directory", "au_input_files")
    output_dir = st.text_input("Output Directory", "au_output")
    country_code = st.selectbox("Country Code", ["AU", "CA", "US", "UK"])

    if st.button("Start Workflow"):
        # Validate input path, but allow output path to be created.
        if not Path(input_dir).is_dir():
            st.error("Please ensure the Input directory exists.")
        else:
            st.session_state.workflow_started = True
            # Reset progress if restarting
            st.session_state.chunker_done = False
            st.session_state.classifier_done = False
            st.session_state.extractor_done = False
            st.session_state.validator_done = False
            st.session_state.prompts = { k: "Not generated yet." for k in st.session_state.prompts }
            st.success("Configuration set. Workflow ready to start.")
            
# --- Main Application ---
st.title("📄 Data Extraction Workflow")
st.markdown("""
Welcome to the Data Extraction Workflow pipeline. This tool helps you process documents, 
extract structured data, and validate the results.
""")
st.markdown("---")

if not st.session_state.workflow_started:
    st.info("Please configure the workflow settings in the sidebar and click 'Start Workflow' to begin.")
else:
    output_path = Path(output_dir)

    # --- Step 1: Document Chunking ---
    st.header("Step 1: Chunking Documents")
    st.markdown("*Breaks down large source documents into smaller pieces for the AI.*")

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Inputs")
        try:
            input_files = [f.name for f in Path(input_dir).iterdir() if f.is_file()]
            
            # File selection dropdown
            if input_files:
                selected_file = st.selectbox(
                    "Select a file to chunk (or leave as 'All Files' to process all):",
                    options=["All Files"] + input_files,
                    key="chunking_file_select"
                )
                st.session_state.selected_chunking_file = selected_file
            else:
                st.error(f"No files found in input directory: {input_dir}")
                selected_file = None
            
            with st.expander("Source Files to be Processed", expanded=True):
                if selected_file == "All Files":
                    st.markdown("**All files will be processed:**")
                    for f in input_files:
                        st.markdown(f"- `{f}`")
                else:
                    st.markdown(f"**Selected file:** `{selected_file}`")
                    
        except FileNotFoundError:
            st.error(f"Input directory not found: {input_dir}")
            input_files = []
            selected_file = None

    with col2:
        st.subheader("Action & Output")
        run_chunker = st.button("▶️ Create Chunks", disabled=st.session_state.chunker_done)

        if run_chunker:
            if not input_files:
                st.error("No input files found to process.")
            else:
                with st.status("Running chunker...", expanded=True) as status:
                    try:
                        chunker = ChunkingStep()
                        chunks_file_path = output_path / "all_chunks.jsonl"
                        st.session_state.output_files['chunks'] = chunks_file_path

                        if selected_file == "All Files":
                            st.write("Processing all files in directory...")
                            result = chunker.process_directory(input_dir, str(chunks_file_path))
                        else:
                            st.write(f"Processing single file: {selected_file}")
                            single_file_path = Path(input_dir) / selected_file
                            # Process single file
                            file_result = chunker.process_file(str(single_file_path))
                            
                            # Save the chunks using the chunker's save method
                            chunker.save_chunks(file_result.get('chunks', []), str(chunks_file_path))
                            
                            result = file_result
                        
                        total_chunks = result.get('chunking_stats', {}).get('total_chunks', len(result.get('chunks', [])))
                        st.write(f"✅ Success! {total_chunks} chunks created.")
                        st.session_state.chunker_done = True
                        status.update(label="Chunking complete!", state="complete")
                        
                    except Exception as e:
                        st.error(f"An error occurred during chunking: {e}")
                        status.update(label="Chunking failed!", state="error")
                
                # Rerun to update the UI state after button press
                st.rerun()

        if st.session_state.chunker_done:
            st.success("Chunking step completed.")
            if st.session_state.output_files['chunks']:
                show_filtered_chunk_preview(st.session_state.output_files['chunks'])

    # --- Step 2: Classification ---
    if st.session_state.chunker_done:
        st.markdown("---")
        st.header("Step 2: Classifying Content")
        st.markdown("*Reads the chunks for a selected file and maps them to known data models.*")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Inputs")
            with st.expander("Input File for Classifier", expanded=True):
                st.markdown(f"- `{st.session_state.output_files['chunks'].name}`")

            # Allow user to select a source file to classify
            try:
                # Read chunks to find available source files
                with open(st.session_state.output_files['chunks'], 'r', encoding='utf-8') as f:
                    all_chunks = [json.loads(line) for line in f]
                
                source_files = sorted(list(set(chunk.get("source_path") for chunk in all_chunks)))
                
                if not source_files:
                    st.warning("No source files found in chunks file.")
                    selected_file_for_classification = None
                else:
                    selected_file_for_classification = st.selectbox(
                        "Select a source file to classify:", 
                        options=source_files
                    )

                # --- Interactive Prompt Preview ---
                if selected_file_for_classification:
                    chunks_for_selected_file = [
                        chunk for chunk in all_chunks if chunk.get("source_path") == selected_file_for_classification
                    ]
                    
                    # For UI preview, use a sample of chunks to keep it fast
                    sample_chunks = chunks_for_selected_file[:10] # e.g., first 10 chunks
                    if len(chunks_for_selected_file) > 20:
                        sample_chunks.extend(chunks_for_selected_file[-5:]) # and last 5 for context

                    prompt_builder = PromptBuilder()
                    prompt = prompt_builder.build_mapper_prompt(sample_chunks)
                    st.session_state.prompts['classifier'] = prompt
                    
                    with st.expander("🔍 View Generated Prompt Preview", expanded=False):
                        st.info("ℹ️ This is a preview using a sample of the chunks. The full set will be processed in batches by the agent.")
                        st.text_area(
                            "Prompt for selected file", 
                            st.session_state.prompts['classifier'], 
                            height=200,
                            key="classifier_prompt_preview"
                        )

            except Exception as e:
                st.error(f"Could not read or parse chunks file: {e}")
                selected_file_for_classification = None

        with col2:
            st.subheader("Action & Output")
            run_classifier = st.button("▶️ Run Classifier Agent", disabled=(not selected_file_for_classification or st.session_state.classifier_done))

            if run_classifier and selected_file_for_classification:
                with st.status(f"Running classifier on '{Path(selected_file_for_classification).name}'...", expanded=True) as status:
                    try:
                        agent_config = AgentConfig(country=country_code)
                        classifier = ClassifierAgent(agent_config.get_agent_config("classifier"))
                        
                        mappings_file_path = output_path / "mappings.json"
                        st.session_state.output_files['mappings'] = mappings_file_path

                        # Filter chunks for the selected file
                        chunks_for_selected_file = [
                            chunk for chunk in all_chunks if chunk.get("source_path") == selected_file_for_classification
                        ]

                        st.write("Using pre-generated prompt...")
                        # The prompt is already in session state from the interactive preview
                        prompt = st.session_state.prompts['classifier']
                        st.write("Sending to agent...")

                        # Process only the chunks for the selected file
                        # We re-use the agent's internal method that takes chunks and returns mappings
                        file_mappings = classifier._process_file_chunks(selected_file_for_classification, chunks_for_selected_file)
                        
                        # Save the result to mappings.json
                        final_mappings = {"mappings": file_mappings}
                        with open(mappings_file_path, 'w', encoding='utf-8') as f:
                            json.dump(final_mappings, f, indent=2)

                        st.write(f"✅ Success! {len(file_mappings)} mappings created.")
                        st.session_state.classifier_done = True
                        status.update(label="Classification complete!", state="complete")
                        
                    except Exception as e:
                        st.error(f"An error occurred during classification: {e}")
                        status.update(label="Classification failed!", state="error")
                
                st.rerun()

            if st.session_state.classifier_done:
                st.success("Classification step completed.")
                if st.session_state.output_files['mappings']:
                    show_file_preview(st.session_state.output_files['mappings'])
                
                with st.expander("🔍 View Prompt Sent to Classifier"):
                    st.text_area("Classifier Prompt", st.session_state.prompts['classifier'], height=300)
                

    # --- Step 3: Extraction ---
    if st.session_state.classifier_done:
        st.markdown("---")
        st.header("Step 3: Extracting Structured Data")
        st.markdown("*Uses a selected data model to extract structured records from the chunks.*")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Inputs")
            with st.expander("Input Files for Extractor", expanded=True):
                if st.session_state.output_files['chunks']:
                    st.markdown(f"- `{st.session_state.output_files['chunks'].name}`")
                else:
                    st.markdown("- `all_chunks.jsonl` (Not available)")
                
                if st.session_state.output_files['mappings']:
                    st.markdown(f"- `{st.session_state.output_files['mappings'].name}`")
                else:
                    st.markdown("- `mappings.json` (Not available)")
            
            # Allow user to select a model from mappings.json
            try:
                if not st.session_state.output_files['mappings']:
                    st.warning("Mappings file not available. Please run the classifier step first.")
                    available_models = []
                else:
                    with open(st.session_state.output_files['mappings']) as f:
                        mappings_data = json.load(f)
                    available_models = [m['model'] for m in mappings_data.get('mappings', [])]
                
                if not available_models:
                    st.warning("No models found in mappings.json.")
                    selected_model = None
                else:
                    selected_model = st.selectbox("Select a model to extract:", options=available_models)
                
                # --- Interactive Prompt Preview ---
                if selected_model:
                    try:
                        agent_config = AgentConfig(country=country_code)
                        # We need a temporary extractor instance to build the prompt
                        temp_extractor = ExtractionAgent(
                            agent_config=agent_config.get_agent_config("extractor"),
                            mappings_file=str(st.session_state.output_files['mappings'])
                        )
                        all_chunks = temp_extractor.load_chunks_from_file(str(st.session_state.output_files['chunks']))
                        model_spec = temp_extractor.prompt_builder.get_model_spec(selected_model)
                        model_spec["model_name"] = selected_model
                        filtered_chunks = temp_extractor._filter_chunks_by_model(all_chunks, model_spec)
                        
                        # Use a sample of chunks for the UI prompt preview
                        sample_chunks = filtered_chunks[:10] # e.g., first 10 chunks

                        prompt = temp_extractor.prompt_builder.build_extraction_prompt(sample_chunks, model_spec)
                        st.session_state.prompts['extractor'] = prompt

                        with st.expander("🔍 View Generated Prompt Preview", expanded=False):
                            st.info("ℹ️ This is a preview using a sample of the relevant chunks. The full set will be processed in batches by the agent.")
                            st.text_area(
                                "Prompt for selected model",
                                st.session_state.prompts['extractor'],
                                height=200,
                                key="extractor_prompt_preview"
                            )
                    except Exception as e:
                        st.warning(f"Could not generate prompt preview: {e}")


            except Exception as e:
                st.error(f"Could not load or parse mappings.json: {e}")
                selected_model = None
        
        with col2:
            st.subheader("Action & Output")
            run_extractor = st.button("▶️ Run Extractor Agent", disabled=(not selected_model or st.session_state.extractor_done))

            if run_extractor and selected_model:
                with st.status(f"Running extractor for model '{selected_model}'...", expanded=True) as status:
                    try:
                        agent_config = AgentConfig(country=country_code)
                        extractor = ExtractionAgent(
                            agent_config=agent_config.get_agent_config("extractor"),
                            mappings_file=str(st.session_state.output_files['mappings'])
                        )
                        
                        records_file_path = output_path / "records.jsonl"
                        st.session_state.output_files['records'] = records_file_path

                        st.write("Loading chunks...")
                        chunks = extractor.load_chunks_from_file(str(st.session_state.output_files['chunks']))
                        
                        st.write("Using pre-generated prompt...")
                        # The prompt is already in session state from the interactive preview
                        st.write("Sending to agent...")
                        
                        st.write(f"Processing model: {selected_model}")
                        result = extractor.process_single_model(chunks, selected_model)
                        
                        extractor.save_results(result, str(records_file_path))
                        
                        records_count = len(result.get('records', []))
                        st.write(f"✅ Success! Extracted {records_count} records.")
                        st.session_state.extractor_done = True
                        status.update(label="Extraction complete!", state="complete")
                        
                    except Exception as e:
                        st.error(f"An error occurred during extraction: {e}")
                        status.update(label="Extraction failed!", state="error")
                
                st.rerun()

            if st.session_state.extractor_done:
                st.success("Extraction step completed.")
                if st.session_state.output_files['records']:
                    show_filtered_records_preview(st.session_state.output_files['records'])

                with st.expander("🔍 View Prompt Sent to Extractor"):
                    st.text_area("Extractor Prompt", st.session_state.prompts['extractor'], height=300)
                

    # --- Step 4: Validation ---
    if st.session_state.extractor_done:
        st.markdown("---")
        st.header("Step 4: Validating Extracted Records")
        st.markdown("*Checks the extracted records for a selected model against the original source chunks for accuracy.*")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Inputs")
            with st.expander("Input Files for Validator", expanded=True):
                if st.session_state.output_files['records']:
                    st.markdown(f"- `{st.session_state.output_files['records'].name}`")
                else:
                    st.markdown("- `records.jsonl` (Not available)")
                
                if st.session_state.output_files['chunks']:
                    st.markdown(f"- `{st.session_state.output_files['chunks'].name}`")
                else:
                    st.markdown("- `all_chunks.jsonl` (Not available)")

            # Allow user to select a model from records.jsonl
            try:
                if not st.session_state.output_files['records']:
                    st.warning("Records file not available. Please run the extraction step first.")
                    available_models = []
                else:
                    with open(st.session_state.output_files['records'], 'r', encoding='utf-8') as f:
                        all_records = [json.loads(line) for line in f]
                    available_models = sorted(list(set(r['model'] for r in all_records)))
                
                if not available_models:
                    st.warning("No models found in records.jsonl.")
                    selected_model_for_validation = None
                else:
                    selected_model_for_validation = st.selectbox(
                        "Select a model to validate:", 
                        options=available_models,
                        key="validation_model_select" # Unique key for this selectbox
                    )
                
                # --- Interactive Prompt Preview ---
                if selected_model_for_validation:
                    try:
                        agent_config = AgentConfig(country=country_code)
                        temp_validator = ValidationAgent(agent_config.get_agent_config("validator"))
                        records_by_model = temp_validator._load_and_group_records(str(st.session_state.output_files['records']))
                        records_for_model = records_by_model.get(selected_model_for_validation, [])
                        records_data = [r.get('record', {}) for r in records_for_model]
                        chunks_dict = temp_validator._load_chunks(str(st.session_state.output_files['chunks']))
                        evidence_chunks = temp_validator._get_evidence_chunks_for_model(records_for_model, chunks_dict)
                        model_spec = temp_validator.prompt_builder.get_model_spec(selected_model_for_validation)

                        # Use a sample of records and evidence for the UI preview
                        sample_records = records_data[:10] # Already using a sample of records
                        sample_evidence_chunks = evidence_chunks[:10]
                        if len(evidence_chunks) > 20:
                            sample_evidence_chunks.extend(evidence_chunks[-5:])
                        
                        prompt = temp_validator.prompt_builder.build_validation_prompt_for_model(
                            model_name=selected_model_for_validation,
                            records_data=sample_records,
                            evidence_chunks=sample_evidence_chunks
                        )
                        st.session_state.prompts['validator'] = prompt

                        with st.expander("🔍 View Generated Prompt Preview", expanded=False):
                            st.info("ℹ️ This is a preview using a sample of the records and evidence chunks. The full set will be processed in batches by the agent.")
                            st.text_area(
                                "Prompt for selected model (first batch)",
                                st.session_state.prompts['validator'],
                                height=200,
                                key="validator_prompt_preview"
                            )
                    except Exception as e:
                        st.warning(f"Could not generate prompt preview: {e}")

            except Exception as e:
                st.error(f"Could not load or parse records.jsonl: {e}")
                selected_model_for_validation = None

        with col2:
            st.subheader("Action & Output")
            run_validator = st.button("▶️ Run Validator Agent", disabled=(not selected_model_for_validation or st.session_state.validator_done))

            if run_validator and selected_model_for_validation:
                with st.status(f"Running validator for model '{selected_model_for_validation}'...", expanded=True) as status:
                    try:
                        agent_config = AgentConfig(country=country_code)
                        validator = ValidationAgent(agent_config.get_agent_config("validator"))

                        validation_summary_path = output_path / "validation_summary.json"
                        st.session_state.output_files['validation_summary'] = validation_summary_path
                        st.session_state.output_files['validation'] = output_path / "validation.jsonl"
                        
                        st.write("Using pre-generated prompt...")
                        # The prompt is already in session state from the interactive preview
                        st.write("Sending to agent...")

                        st.write("Validating records against chunks...")
                        summary = validator.validate_records_file(
                            records_file=str(st.session_state.output_files['records']),
                            chunks_file=str(st.session_state.output_files['chunks']),
                            output_dir=str(output_path),
                            model_filter=selected_model_for_validation # Pass the selected model to the agent
                        )
                        
                        overall_summary = summary.get('overall', {})
                        st.write(f"✅ Success! Validated {overall_summary.get('total_records', 0)} records for model '{selected_model_for_validation}'.")
                        st.session_state.validator_done = True
                        status.update(label="Validation complete!", state="complete")
                        
                    except Exception as e:
                        st.error(f"An error occurred during validation: {e}")
                        status.update(label="Validation failed!", state="error")
                
                st.rerun()
            
            if st.session_state.validator_done:
                st.success("Validation step completed.")
                if st.session_state.output_files['validation_summary']:
                    show_file_preview(st.session_state.output_files['validation_summary'])
                if st.session_state.output_files['validation']:
                    show_filtered_validation_preview(st.session_state.output_files['validation'])
                
                with st.expander("🔍 View Prompt Sent to Validator (First Batch)"):
                    st.text_area("Validator Prompt", st.session_state.prompts['validator'], height=300)
                
    
    # --- Workflow Completion ---
    if st.session_state.validator_done:
        st.markdown("---")
        st.balloons()
        st.success("🎉 **Workflow Finished!**")
        st.info("You have successfully run all the agents. You can review the final outputs in your designated output directory.")
