"""
Configuration for LangChain-based agents
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import BaseMemory
from langchain.memory import ConversationBufferMemory

class AgentConfig:
    """Configuration for all agents"""
    
    def __init__(self, country: Optional[str] = None):
        # Country Configuration
        self.country = country or os.getenv("LLM_DEFAULT_COUNTRY", "CA")
        self.defaults = self._load_country_defaults()
        
        # LLM Configuration
        self.llm_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
        self.llm_max_tokens = int(os.getenv("LLM_MAX_TOKENS", "16000"))
        
        # Embeddings Configuration
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        
        # Chunking Configuration
        self.default_chunk_size = int(os.getenv("DEFAULT_CHUNK_SIZE", "1200"))
        self.default_chunk_overlap = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "200"))
        self.max_chunk_size = int(os.getenv("MAX_CHUNK_SIZE", "2000"))
        self.min_chunk_size = int(os.getenv("MIN_CHUNK_SIZE", "100"))
        
        # Quality Thresholds
        self.min_chunk_quality = float(os.getenv("MIN_CHUNK_QUALITY", "0.7"))
        self.min_extraction_confidence = float(os.getenv("MIN_EXTRACTION_CONFIDENCE", "0.8"))
        self.min_validation_confidence = float(os.getenv("MIN_VALIDATION_CONFIDENCE", "0.8"))
        
        # Memory Configuration
        self.enable_memory = os.getenv("ENABLE_MEMORY", "true").lower() == "true"
        self.memory_max_tokens = int(os.getenv("MEMORY_MAX_TOKENS", "10000"))
        
        # Output Configuration
        self.output_dir = os.getenv("AGENT_OUTPUT_DIR", "agent_output")
        self.enable_verbose = os.getenv("AGENT_VERBOSE", "false").lower() == "true"
        
        # Initialize LLM and embeddings
        self._initialize_llm()
        self._initialize_embeddings()
        self._initialize_memory()
    
    def _load_country_defaults(self) -> Dict[str, Any]:
        """Load country-specific defaults from defaults/{country}.json"""
        try:
            defaults_file = Path(f"defaults/{self.country.upper()}.json")
            if defaults_file.exists():
                with open(defaults_file, 'r', encoding='utf-8') as f:
                    defaults = json.load(f)
                    print(f"Loaded country defaults from {defaults_file}: {defaults}")
                    return defaults
            else:
                print(f"Country defaults file not found: {defaults_file}")
                return {}
        except Exception as e:
            print(f"Failed to load country defaults: {e}")
            return {}
    
    def _initialize_llm(self):
        """Initialize the LLM"""
        self.llm = ChatOpenAI(
            model=self.llm_model,
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens
        )
    
    def _initialize_embeddings(self):
        """Initialize embeddings"""
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model
        )
    
    def _initialize_memory(self):
        """Initialize memory if enabled"""
        if self.enable_memory:
            self.memory = ConversationBufferMemory(
                max_token_limit=self.memory_max_tokens,
                return_messages=True
            )
        else:
            self.memory = None
    
    def get_agent_config(self, agent_type: str) -> Dict[str, Any]:
        """Get configuration for specific agent type"""
        base_config = {
            "llm": self.llm,
            "embeddings": self.embeddings,
            "memory": self.memory,
            "verbose": self.enable_verbose,
            "country": self.country,
            "defaults": self.defaults
        }
        
        if agent_type == "chunker":
            base_config.update({
                "default_chunk_size": self.default_chunk_size,
                "default_chunk_overlap": self.default_chunk_overlap,
                "max_chunk_size": self.max_chunk_size,
                "min_chunk_size": self.min_chunk_size,
                "min_quality": self.min_chunk_quality
            })
        elif agent_type == "classifier":
            base_config.update({
                "max_tokens": 100000
            })
        elif agent_type == "extractor":
            base_config.update({
                "min_confidence": self.min_extraction_confidence,
                "max_tokens": 100000
            })
        elif agent_type == "validator":
            base_config.update({
                "min_confidence": self.min_validation_confidence,
                "max_tokens": 100000
            })
        
        return base_config

# Global configuration instance
config = AgentConfig()
