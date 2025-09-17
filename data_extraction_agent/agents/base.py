"""
Base agent class for all document processing agents
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pathlib import Path
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.schema import Document

# Try different import paths for BaseCallbackHandler based on LangChain version
try:
    from langchain.callbacks import BaseCallbackHandler
except ImportError:
    try:
        from langchain_core.callbacks import BaseCallbackHandler
    except ImportError:
        from langchain.callbacks.base import BaseCallbackHandler

from ..config import config

logger = logging.getLogger(__name__)

class AgentCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for agent monitoring"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.step_count = 0
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        self.step_count += 1
        logger.info(f"[{self.agent_name}] LLM Step {self.step_count}: Processing {len(prompts)} prompts")
    
    def on_llm_end(self, response, **kwargs) -> None:
        logger.info(f"[{self.agent_name}] LLM Step {self.step_count} completed")
    
    def on_llm_error(self, error: Exception, **kwargs) -> None:
        logger.error(f"[{self.agent_name}] LLM Step {self.step_count} failed: {error}")
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        logger.info(f"[{self.agent_name}] Chain started with inputs: {list(inputs.keys())}")
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        logger.info(f"[{self.agent_name}] Chain completed with outputs: {list(outputs.keys())}")
    
    def on_chain_error(self, error: Exception, **kwargs) -> None:
        logger.error(f"[{self.agent_name}] Chain failed: {error}")

class BaseAgent(ABC):
    """Base class for all document processing agents"""
    
    def __init__(self, agent_name: str, agent_config: Optional[Dict[str, Any]] = None):
        self.agent_name = agent_name
        self.config = agent_config or config.get_agent_config(agent_name.split('_')[0])
        self.llm = self.config["llm"]
        self.memory = self.config.get("memory")
        self.verbose = self.config.get("verbose", False)
        
        # Setup callbacks
        self.callback_handler = AgentCallbackHandler(agent_name)
        self.callbacks = [self.callback_handler] if self.verbose else []
        
        # Agent state
        self.processing_history = []
        self.error_count = 0
        self.success_count = 0
        
        logger.info(f"Initialized {agent_name} agent")
    
    @abstractmethod
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process inputs and return results"""
        pass
    
    def invoke_llm(self, messages: List[BaseMessage], **kwargs) -> str:
        """Invoke LLM with error handling and retries"""
        try:
            # Remove callbacks from kwargs to avoid conflicts
            kwargs.pop('callbacks', None)
            
            # Only use callbacks if verbose mode is enabled and callbacks are available
            if self.verbose and self.callbacks:
                response = self.llm.invoke(
                    messages, 
                    callbacks=self.callbacks,
                    **kwargs
                )
            else:
                response = self.llm.invoke(
                    messages, 
                    **kwargs
                )
            
            self.success_count += 1
            return response.content
        except Exception as e:
            self.error_count += 1
            logger.error(f"[{self.agent_name}] LLM invocation failed: {e}")
            raise
    
    def create_system_message(self, content: str) -> SystemMessage:
        """Create a system message"""
        return SystemMessage(content=content)
    
    def create_human_message(self, content: str) -> HumanMessage:
        """Create a human message"""
        return HumanMessage(content=content)
    
    def save_state(self, state: Dict[str, Any], file_path: str) -> None:
        """Save agent state to file"""
        state_data = {
            "agent_name": self.agent_name,
            "processing_history": self.processing_history,
            "error_count": self.error_count,
            "success_count": self.success_count,
            "state": state
        }
        
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(state_data, f, indent=2)
        
        logger.info(f"[{self.agent_name}] State saved to {file_path}")
    
    def load_state(self, file_path: str) -> Dict[str, Any]:
        """Load agent state from file"""
        try:
            with open(file_path, 'r') as f:
                state_data = json.load(f)
            
            self.processing_history = state_data.get("processing_history", [])
            self.error_count = state_data.get("error_count", 0)
            self.success_count = state_data.get("success_count", 0)
            
            logger.info(f"[{self.agent_name}] State loaded from {file_path}")
            return state_data.get("state", {})
        except FileNotFoundError:
            logger.warning(f"[{self.agent_name}] State file not found: {file_path}")
            return {}
        except Exception as e:
            logger.error(f"[{self.agent_name}] Failed to load state: {e}")
            return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        total_operations = self.success_count + self.error_count
        success_rate = self.success_count / total_operations if total_operations > 0 else 0
        
        return {
            "agent_name": self.agent_name,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": success_rate,
            "total_operations": total_operations,
            "processing_history_length": len(self.processing_history)
        }
    
    def reset_stats(self) -> None:
        """Reset agent statistics"""
        self.processing_history = []
        self.error_count = 0
        self.success_count = 0
        logger.info(f"[{self.agent_name}] Statistics reset")
    
    def log_processing_step(self, step_name: str, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Log a processing step"""
        from datetime import datetime
        step_data = {
            "step_name": step_name,
            "timestamp": datetime.now().isoformat(),
            "inputs_keys": list(inputs.keys()),
            "outputs_keys": list(outputs.keys()),
            "success": True
        }
        
        self.processing_history.append(step_data)
        
        if self.verbose:
            logger.info(f"[{self.agent_name}] {step_name}: {len(inputs)} inputs -> {len(outputs)} outputs")
    
    def __str__(self) -> str:
        return f"{self.agent_name}(success={self.success_count}, errors={self.error_count})"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.agent_name}')"
