from typing import List, Dict, Any, Optional, Tuple
import asyncio
import re
import json
import os
import time
import logging
import hydra
import uuid
import wandb
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from together import AsyncTogether

CONFIG_NAME = "llm_judge"
CONFIG_PATH = "./"

GLOBAL_STEP = 0
WANDB_LOGGER = None

import wandb
import json
import time
import re
from typing import Dict, Any, List, Optional

class WandbLogger:
    """
    A class to handle Weights & Biases logging for LLM Judge
    """
    def __init__(self, config: Dict[str, Any], run_name: Optional[str] = None):
        """
        Initialize the WandbLogger
        
        Args:
            config: Configuration dictionary for the experiment
            run_name: Optional name for this wandb run
        """
        # Initialize wandb
        self.run_name = run_name or f"llm-judge-{time.strftime('%Y%m%d-%H%M%S')}"
        self.run = wandb.init(
            project="llm-judge",
            name=self.run_name,
            config=config
        )
        
        # Record start time
        self.start_time = time.time()
        
        # Create the four tables for YES/NO results
        self.tables = {
            "grounding_yes": wandb.Table(columns=["item_id", "state", "content", "response"]),
            "grounding_no": wandb.Table(columns=["item_id", "state", "content", "response", "failure_reason"]),
            "worldmodeling_yes": wandb.Table(columns=["item_id", "state", "content", "response"]),
            "worldmodeling_no": wandb.Table(columns=["item_id", "state", "content", "response", "failure_reason"])
        }
        
        # Initialize metrics with sections but without accuracy
        self.metrics = {
            # Time metrics
            "time/total_seconds": 0,
            "time/avg_per_request_seconds": 0,
            
            # Total section metrics
            "total/requests": 0,
            "total/successful": 0,
            
            # Grounding section metrics
            "grounding/requests": 0,
            "grounding/successful": 0,
            
            # Worldmodeling section metrics
            "worldmodeling/requests": 0,
            "worldmodeling/successful": 0,
            
            # Error metrics
            "errors/timeout_count": 0,
            "errors/parsing_failure_count": 0,
            "errors/other_failure_count": 0,
            "errors/total_failure_count": 0,
            "errors/timeout_rate": 0.0,
            "errors/parsing_failure_rate": 0.0,
            "errors/success_rate": 0.0
        }
        
        # Initialize environment-specific metrics
        self.env_metrics = {}
        
        # Keep track of invalid environment names
        self._invalid_env_names = set()
        
    def _get_env_name_string(self, env_name_obj):
        """
        Convert environment name object to a proper string for metrics
        
        Args:
            env_name_obj: Environment name object
            
        Returns:
            str: Proper string representation of environment name
        """
        if env_name_obj is None:
            return "unknown_env"
        
        # If it's an object with a class, use the class name
        if hasattr(env_name_obj, "__class__") and not isinstance(env_name_obj, str):
            # Get the class name without any module prefix
            class_name = env_name_obj.__class__.__name__
            return class_name.lower()
        
        # If it's already a string, ensure it's not problematic
        if isinstance(env_name_obj, str):
            # Replace any problematic strings
            env_str = env_name_obj.lower()
            if env_str in ["str", "type", "object", "null", "none", "unknown"]:
                return f"env_{env_str}"
            
            # Remove any angle brackets that might be in the string representation
            env_str = re.sub(r'<[^>]+>', '', env_str)
            
            # Remove any special characters that might cause issues in W&B
            env_str = re.sub(r'[^\w\-]', '_', env_str)
            
            # Ensure the name starts with a letter
            if not env_str[0].isalpha():
                env_str = f"env_{env_str}"
                
            return env_str
        
        # For other types, convert to string but clean up the result
        env_str = str(env_name_obj).lower()
        
        # Remove any problematic prefixes or patterns
        env_str = re.sub(r'<[^>]+>', '', env_str)
        env_str = re.sub(r'[^\w\-]', '_', env_str)
        
        # Ensure the name is valid
        if env_str in ["str", "type", "object", "null", "none", "unknown"]:
            return f"env_{env_str}"
            
        # Ensure the name starts with a letter
        if not env_str or not env_str[0].isalpha():
            env_str = f"env_{env_str}"
            
        return env_str
    
    def _format_state(self, state: Dict[str, Any]) -> str:
        """
        Format state dictionary into a human-readable single line string
        
        Args:
            state: State dictionary
            
        Returns:
            str: Human-readable state string
        """
        # Use compact JSON formatting with no indentation
        compact_json = json.dumps(state, separators=(',', ':'))
        
        # Add spaces after commas and colons for better readability
        readable_json = re.sub(r'[:,]', lambda m: m.group(0) + ' ', compact_json)
        
        # Add some minimal formatting to make arrays more readable
        readable_json = readable_json.replace('[', '[ ').replace(']', ' ]')
        
        # Fix any double spaces
        readable_json = re.sub(r'\s+', ' ', readable_json)
        
        return readable_json
        
    def log_step(self, step: int):
        """
        Log current metrics for this step
        
        Args:
            step: Current step number
        """
        # Calculate time metrics
        current_time = time.time()
        self.metrics["time/total_seconds"] = current_time - self.start_time
        
        # Calculate average time per request if any requests were made
        if self.metrics["total/requests"] > 0:
            self.metrics["time/avg_per_request_seconds"] = self.metrics["time/total_seconds"] / self.metrics["total/requests"]
            
            # Calculate error rates
            total_failures = (self.metrics["errors/timeout_count"] + 
                             self.metrics["errors/parsing_failure_count"] + 
                             self.metrics["errors/other_failure_count"])
            
            self.metrics["errors/total_failure_count"] = total_failures
            self.metrics["errors/timeout_rate"] = self.metrics["errors/timeout_count"] / self.metrics["total/requests"]
            self.metrics["errors/parsing_failure_rate"] = self.metrics["errors/parsing_failure_count"] / self.metrics["total/requests"]
            self.metrics["errors/success_rate"] = 1.0 - (total_failures / self.metrics["total/requests"])
        
        # Log aggregated metrics to wandb (not by environment)
        wandb.log(self.metrics, step=step)
        
        # Log environment-specific metrics
        env_combined_metrics = {}
        
        for env_name, env_data in self.env_metrics.items():
            # Make sure we've sanitized the environment name
            env_key = self._get_env_name_string(env_name)
            
            # Total section for this environment
            env_combined_metrics[f"env/{env_key}/total/requests"] = env_data["total"]
            env_combined_metrics[f"env/{env_key}/total/successful"] = env_data["correct"]
            
            # Grounding section for this environment
            env_combined_metrics[f"env/{env_key}/grounding/requests"] = env_data["grounding_total"]
            env_combined_metrics[f"env/{env_key}/grounding/successful"] = env_data["grounding_correct"]
            
            # Worldmodeling section for this environment
            env_combined_metrics[f"env/{env_key}/worldmodeling/requests"] = env_data["worldmodeling_total"]
            env_combined_metrics[f"env/{env_key}/worldmodeling/successful"] = env_data["worldmodeling_correct"]
            
            # Error metrics for this environment
            env_combined_metrics[f"env/{env_key}/errors/timeout_count"] = env_data.get("timeout_count", 0)
            env_combined_metrics[f"env/{env_key}/errors/parsing_failures"] = env_data.get("parsing_failures", 0)
        
        # Log all environment metrics at once
        if env_combined_metrics:
            wandb.log(env_combined_metrics, step=step)
    
    def add_result(self, item: Dict[str, Any], result: float, response: str, failure_reason: Optional[str] = None):
        """
        Add a result to the appropriate table and update metrics
        
        Args:
            item: The input item dictionary
            result: The result score (1.0 for YES, 0.0 for NO)
            response: The model's response text
            failure_reason: The reason for failure if result is 0.0 (timeout, parsing_failure, or None for actual NO)
        """
        # Extract item details
        item_id = item.get("id", "unknown")
        item_type = item.get("type", "unknown")
        
        # Get environment name and ensure it's a proper string
        env_name_obj = item.get("env_name")
        env_name = self._get_env_name_string(env_name_obj)
        
        content = item.get("content", "")
        state = item.get("state", {})
        
        # Format state in a human-readable way
        state_str = self._format_state(state)
        
        # Determine if this was a correct response (YES)
        is_correct = result > 0.5
        
        # Update total section metrics
        self.metrics["total/requests"] += 1
        if is_correct:
            self.metrics["total/successful"] += 1
        else:
            # Track failure reasons
            if failure_reason == "timeout":
                self.metrics["errors/timeout_count"] += 1
            elif failure_reason == "parsing_failure":
                self.metrics["errors/parsing_failure_count"] += 1
            elif failure_reason:  # Any other failure type
                self.metrics["errors/other_failure_count"] += 1
        
        # Update type-specific section metrics
        if item_type == "observation":
            self.metrics["grounding/requests"] += 1
            if is_correct:
                self.metrics["grounding/successful"] += 1
            
            # Add to appropriate table
            table_key = "grounding_yes" if is_correct else "grounding_no"
            if is_correct:
                self.tables[table_key].add_data(item_id, state_str, content, response)
            else:
                self.tables[table_key].add_data(item_id, state_str, content, response, failure_reason or "actual_no")
            
        elif item_type == "prediction":
            self.metrics["worldmodeling/requests"] += 1
            if is_correct:
                self.metrics["worldmodeling/successful"] += 1
                
            # Add to appropriate table
            table_key = "worldmodeling_yes" if is_correct else "worldmodeling_no"
            if is_correct:
                self.tables[table_key].add_data(item_id, state_str, content, response)
            else:
                self.tables[table_key].add_data(item_id, state_str, content, response, failure_reason or "actual_no")
        
        # Update environment-specific metrics
        if env_name not in self.env_metrics:
            self.env_metrics[env_name] = {
                "total": 0,
                "correct": 0,
                "grounding_total": 0,
                "grounding_correct": 0,
                "worldmodeling_total": 0,
                "worldmodeling_correct": 0,
                "timeout_count": 0,
                "parsing_failures": 0
            }
        
        env_metrics = self.env_metrics[env_name]
        env_metrics["total"] += 1
        
        if is_correct:
            env_metrics["correct"] += 1
        else:
            # Track environment-specific failure reasons
            if failure_reason == "timeout":
                env_metrics["timeout_count"] = env_metrics.get("timeout_count", 0) + 1
            elif failure_reason == "parsing_failure":
                env_metrics["parsing_failures"] = env_metrics.get("parsing_failures", 0) + 1
        
        if item_type == "observation":
            env_metrics["grounding_total"] += 1
            if is_correct:
                env_metrics["grounding_correct"] += 1
        elif item_type == "prediction":
            env_metrics["worldmodeling_total"] += 1
            if is_correct:
                env_metrics["worldmodeling_correct"] += 1
    
    def finish(self):
        """
        Finish logging and upload tables to wandb
        """
        # Calculate final time metrics before finishing
        current_time = time.time()
        self.metrics["time/total_seconds"] = current_time - self.start_time
        
        if self.metrics["total/requests"] > 0:
            self.metrics["time/avg_per_request_seconds"] = self.metrics["time/total_seconds"] / self.metrics["total/requests"]
            
            # Calculate final error rates
            total_failures = (self.metrics["errors/timeout_count"] + 
                             self.metrics["errors/parsing_failure_count"] + 
                             self.metrics["errors/other_failure_count"])
            
            self.metrics["errors/total_failure_count"] = total_failures
            self.metrics["errors/timeout_rate"] = self.metrics["errors/timeout_count"] / self.metrics["total/requests"]
            self.metrics["errors/parsing_failure_rate"] = self.metrics["errors/parsing_failure_count"] / self.metrics["total/requests"]
            self.metrics["errors/success_rate"] = 1.0 - (total_failures / self.metrics["total/requests"])
        
        # Log final metrics
        wandb.log(self.metrics)
        
        # Log final tables
        for table_name, table in self.tables.items():
            self.run.log({table_name: table})
        
        # Close wandb run
        wandb.finish()
            
def setup_wandb_logger(config: Dict[str, Any]) -> WandbLogger:
    """
    Set up and return a WandbLogger instance
    
    Args:
        config: Configuration dictionary
        
    Returns:
        WandbLogger: Initialized wandb logger
    """
    return WandbLogger(config)


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Set up logging for the LLM judge module"""
    log_dir = config.get("log_dir", "./logs/llm_judge")
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"llm_judge_{timestamp}.log")
    
    # Configure logging
    logger = logging.getLogger("llm_judge")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplicate logging
    if logger.handlers:
        logger.handlers.clear()
    
    # File handler for logging to file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging to {log_file}")
    
    return logger

def extract_answer(response_text: str) -> tuple[str, Optional[str]]:
    """
    Extract YES/NO answer from LLM response and identify any parsing failures
    
    Args:
        response_text: Response text from the LLM
        
    Returns:
        tuple: (answer, failure_reason) where answer is "YES" or "NO" and 
               failure_reason is None, "parsing_failure", or other error message
    """
    # Check for timeout or error messages
    if response_text.startswith("[TIMEOUT"):
        return "NO", "timeout"
    elif response_text.startswith("[ERROR"):
        return "NO", "api_error"
    
    # Look for <answer> tag
    match = re.search(r"<answer>(YES|NO)</answer>", response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper(), None  # Properly parsed response
    
    # Fallback: Check if YES or NO appears in the text
    if "YES" in response_text.upper() and "NO" not in response_text.upper():
        return "YES", None  # Found YES without tag
    elif "NO" in response_text.upper() and "YES" not in response_text.upper():
        return "NO", None  # Found NO without tag
    
    # If we reach here, the model didn't provide a clear answer in expected format
    return "NO", "parsing_failure"  # Default to NO but mark as parsing failure


async def async_judge_single_item(
    prompt: str, 
    config: Dict[str, Any], 
    logger: logging.Logger, 
    original_id: Any, 
    item_uuid: str,
    item_content: str,
    item_state: Dict[str, Any]
) -> Tuple[float, Dict[str, Any]]:
    """
    Judge a single item asynchronously and return both result and logging data
    
    Args:
        prompt: The prompt to send to the model
        config: Configuration dictionary
        logger: Logger instance
        original_id: Original ID of the item (for logging)
        item_uuid: UUID assigned to the item (for tracking)
        item_content: Original content of the item
        item_state: State of the item
        
    Returns:
        Tuple containing the score (1.0 for YES, 0.0 for NO) and logging data
    """
    # Extract parameters from config
    model = config.get("name", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")
    temperature = config.get("temperature", 0.1)
    max_tokens = config.get("max_tokens", 200)
    max_retries = config.get("max_retries", 3)
    request_timeout = config.get("request_timeout", 15)  # Get timeout from config, default 15 seconds
    
    # Log the request being sent
    logger.debug(f"Sending request for item {original_id} to model: {model} with timeout: {request_timeout}s")
    
    # Initialize async client
    async_client = AsyncTogether(api_key=os.getenv("TOGETHER_API_KEY"), max_retries=max_retries)
    
    messages = [{"role": "user", "content": prompt}]
    
    # Track start time for performance logging
    start_time = time.time()
    
    failure_reason = None
    
    try:
        # Add timeout control to prevent hanging requests
        response = await asyncio.wait_for(
            async_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            ),
            timeout=request_timeout
        )
        
        # Calculate response time
        response_time = time.time() - start_time
        
        response_text = response.choices[0].message.content
        answer, failure_reason = extract_answer(response_text)
        
    except asyncio.TimeoutError:
        # Handle timeout case
        logger.warning(f"Request for item {original_id} timed out after {request_timeout} seconds")
        response_time = time.time() - start_time
        response_text = f"[TIMEOUT AFTER {request_timeout} SECONDS]"
        answer = "NO"  # Default to NO for timeouts
        failure_reason = "timeout"
    except Exception as e:
        # Handle other errors
        logger.error(f"Error in API call for item {original_id}: {str(e)}")
        response_time = time.time() - start_time
        response_text = f"[ERROR: {str(e)}]"
        answer = "NO"  # Default to NO for errors
        failure_reason = "api_error"
    
    # Create detailed log data with only necessary information
    log_data = {
        "original_id": original_id,
        "uuid": item_uuid,
        "model": model,
        "content": item_content,
        "state": item_state,  # Include state for logging
        "response": response_text,
        "answer": answer,
        "failure_reason": failure_reason
    }
    
    return (1.0 if answer == "YES" else 0.0), log_data


def extract_answer(response_text: str) -> tuple[str, Optional[str]]:
    """
    Extract YES/NO answer from LLM response and identify any parsing failures
    
    Args:
        response_text: Response text from the LLM
        
    Returns:
        tuple: (answer, failure_reason) where answer is "YES" or "NO" and 
               failure_reason is None, "parsing_failure", or other error message
    """
    # Check for timeout or error messages
    if response_text.startswith("[TIMEOUT"):
        return "NO", "timeout"
    elif response_text.startswith("[ERROR"):
        return "NO", "api_error"
    
    # Look for <answer> tag
    match = re.search(r"<answer>(YES|NO)</answer>", response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper(), None  # Properly parsed response
    
    # Fallback: Check if YES or NO appears in the text
    if "YES" in response_text.upper() and "NO" not in response_text.upper():
        return "YES", None  # Found YES without tag
    elif "NO" in response_text.upper() and "YES" not in response_text.upper():
        return "NO", None  # Found NO without tag
    
    # If we reach here, the model didn't provide a clear answer in expected format
    return "NO", "parsing_failure"  # Default to NO but mark as parsing failure


async def async_judge_single_item(
    prompt: str, 
    config: Dict[str, Any], 
    logger: logging.Logger, 
    original_id: Any, 
    item_uuid: str,
    item_content: str,
    item_state: Dict[str, Any]
) -> Tuple[float, Dict[str, Any]]:
    """
    Judge a single item asynchronously and return both result and logging data
    
    Args:
        prompt: The prompt to send to the model
        config: Configuration dictionary
        logger: Logger instance
        original_id: Original ID of the item (for logging)
        item_uuid: UUID assigned to the item (for tracking)
        item_content: Original content of the item
        item_state: State of the item
        
    Returns:
        Tuple containing the score (1.0 for YES, 0.0 for NO) and logging data
    """
    # Extract parameters from config
    model = config.get("name", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")
    temperature = config.get("temperature", 0.1)
    max_tokens = config.get("max_tokens", 200)
    max_retries = config.get("max_retries", 3)
    request_timeout = config.get("request_timeout", 15)  # Get timeout from config, default 15 seconds
    
    # Log the request being sent
    logger.debug(f"Sending request for item {original_id} to model: {model} with timeout: {request_timeout}s")
    
    # Initialize async client
    async_client = AsyncTogether(api_key=os.getenv("TOGETHER_API_KEY"), max_retries=max_retries)
    
    messages = [{"role": "user", "content": prompt}]
    
    # Track start time for performance logging
    start_time = time.time()
    
    failure_reason = None
    
    try:
        # Add timeout control to prevent hanging requests
        response = await asyncio.wait_for(
            async_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            ),
            timeout=request_timeout
        )
        
        # Calculate response time
        response_time = time.time() - start_time
        
        response_text = response.choices[0].message.content
        answer, failure_reason = extract_answer(response_text)
        
    except asyncio.TimeoutError:
        # Handle timeout case
        logger.warning(f"Request for item {original_id} timed out after {request_timeout} seconds")
        response_time = time.time() - start_time
        response_text = f"[TIMEOUT AFTER {request_timeout} SECONDS]"
        answer = "NO"  # Default to NO for timeouts
        failure_reason = "timeout"
    except Exception as e:
        # Handle other errors
        logger.error(f"Error in API call for item {original_id}: {str(e)}")
        response_time = time.time() - start_time
        response_text = f"[ERROR: {str(e)}]"
        answer = "NO"  # Default to NO for errors
        failure_reason = "api_error"
    
    # Create detailed log data with only necessary information
    log_data = {
        "original_id": original_id,
        "uuid": item_uuid,
        "model": model,
        "content": item_content,
        "state": item_state,  # Include state for logging
        "response": response_text,
        "answer": answer,
        "failure_reason": failure_reason
    }
    
    return (1.0 if answer == "YES" else 0.0), log_data


def run_llm_judge(inputs: List[Dict[str, Any]]) -> List[float]:
    """
    Run LLM judge synchronously by handling asyncio internally.
    This is the main function to call from other modules.
    
    Args:
        inputs: A list of dicts: {"id": id, "content": content, "state": state, "type": "observation"/"prediction"}
        
    Returns:
        list: A list of scores (0.0 or 1.0) for each input.
    """
    if not inputs:
        return []
    
    # Track overall start time
    overall_start_time = time.time()
    
    # Load configuration
    config = load_config()
    
    # Setup logging
    logger = setup_logging(config)
    logger.info(f"Starting LLM judge with {len(inputs)} inputs")
    
    # Set up wandb logger
    wandb_logger = setup_wandb_logger(config)
    
    # Set up batch log file path - use a fixed name for appending across multiple batches
    log_dir = config.get("log_dir", "./logs/llm_judge")
    # Use a date-based filename rather than timestamp to group all batches from the same day
    current_date = datetime.now().strftime("%Y%m%d")
    batch_log_file = os.path.join(log_dir, f"batch_summary_{current_date}.json")
    
    # Initialize batch log data
    batch_log = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "batch_size": len(inputs),
        "model": config.get("name", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"),
        "temperature": config.get("temperature", 0.1),
        "max_tokens": config.get("max_tokens", 200),
        "items": [],
        "success_rate": 0,
        "total_time_sec": 0,
        "avg_time_per_item_sec": 0,
        "timeout_count": 0,
        "parsing_failure_count": 0
    }
    
    # Generate UUIDs for each input item and create a mapping to track original positions
    uuid_to_position = {}
    position_to_uuid = {}
    for i, item in enumerate(inputs):
        # Generate a UUID for this item
        item_uuid = str(uuid.uuid4())
        uuid_to_position[item_uuid] = i
        position_to_uuid[i] = item_uuid
    
    # Initialize results array
    results = [0.0] * len(inputs)
    
    # Create formatted prompt inputs
    max_tokens = config.get("max_tokens", 200)
    
    eval_items = []
    for i, item in enumerate(inputs):
        # Find the matching environment template
        env_name = None
        templates = config.get("prompt_templates")
        
        # Get string representation of environment name for template matching
        env_name_obj = item.get("env_name")
        env_name_str = str(env_name_obj)
        
        for template_env in templates.keys():
            if template_env.lower() in env_name_str.lower():
                env_name = template_env
                break
        
        if not env_name:
            # Default to frozenlake if no match
            env_name = "frozenlake"
            logger.warning(f"No matching template for env_name '{env_name_str}', using default: {env_name}")
        
        # Get template type based on observation or prediction
        template_type = 'grounding' if item["type"] == 'observation' else 'worldmodeling'
        
        # Get the appropriate template
        template = templates.get(env_name, {}).get(template_type, '')
        if not template:
            logger.error(f"Template not found for environment '{env_name}' and type '{template_type}'")
            continue
        
        prompt = template.format(
            state_information_dict=json.dumps(item["state"], indent=2),
            natural_language_description=item["content"],
            max_tokens=max_tokens
        )
        
        # Use the generated UUID for tracking instead of memory ID
        item_uuid = position_to_uuid[i]
        
        eval_items.append({
            "uuid": item_uuid,  # Use UUID for internal tracking
            "original_id": item.get("id", "unknown"),  # Store original ID for logging
            "prompt": prompt,
            "content": item["content"],  # Store original content for logging
            "state": item["state"],  # Store state for logging
            "type": item["type"],  # Store type for wandb logging
            "env_name": item["env_name"]  # Store env_name for wandb logging
        })
    
    # Process in batches using asyncio
    current_batch = eval_items.copy()
    max_retries = config.get("max_retries", 3)
    retry_count = 0
    
    while current_batch and retry_count < max_retries:
        batch_start_time = time.time()
        
        async def process_batch():
            batch_results = []
            failures = []
            
            # Process items in parallel with concurrency control
            semaphore = asyncio.Semaphore(config.get("max_parallel_requests", 10))
            
            async def process_item(item):
                try:
                    async with semaphore:
                        # Pass complete item information to avoid lookup later
                        result, log_data = await async_judge_single_item(
                            item["prompt"], 
                            config, 
                            logger, 
                            original_id=item["original_id"],
                            item_uuid=item["uuid"],
                            item_content=item["content"],
                            item_state=item["state"]  # Pass state to the function
                        )
                        return {
                            "uuid": item["uuid"],  # Use UUID for tracking
                            "result": result, 
                            "success": True, 
                            "log_data": log_data,
                            "type": item["type"],  # Pass type for wandb
                            "env_name": item["env_name"]  # Pass env_name for wandb
                        }
                except Exception as e:
                    error_msg = f"Error processing item {item['original_id']}: {str(e)}"
                    logger.error(error_msg)
                    return {"uuid": item["uuid"], "success": False, "error": str(e)}
            
            tasks = [process_item(item) for item in current_batch]
            task_results = await asyncio.gather(*tasks)
            
            for result in task_results:
                if result["success"]:
                    batch_results.append(result)
                    
                    # Log successful judgment
                    if "log_data" in result:
                        log_entry = result["log_data"]
                        failure_info = ""
                        if log_entry.get("failure_reason"):
                            failure_info = f" (Failure: {log_entry['failure_reason']})"
                        logger.info(f"Item {log_entry['original_id']} - Answer: {log_entry['answer']}{failure_info}")
                else:
                    failures.append(next(item for item in current_batch if item["uuid"] == result["uuid"]))
                    logger.warning(f"Failed item with UUID {result['uuid']}: {result.get('error', 'Unknown error')}")
            
            return batch_results, failures
        
        # Run in a new event loop
        try:
            loop = asyncio.get_event_loop()
            # Check if loop is closed
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        batch_results, next_batch = loop.run_until_complete(process_batch())
        
        # Calculate batch processing time
        batch_time = time.time() - batch_start_time
        
        # Update results and collect log data for batch summary
        for item in batch_results:
            # Get original position using UUID
            if item["uuid"] in uuid_to_position:
                original_pos = uuid_to_position[item["uuid"]]
                results[original_pos] = item["result"]
                
                # Add to wandb logger
                if "log_data" in item:
                    # Create a properly structured item for add_result
                    original_item = {
                        "id": item["log_data"]["original_id"],
                        "content": item["log_data"]["content"],
                        "state": item["log_data"]["state"],
                        "type": item["type"],
                        "env_name": item["env_name"]
                    }
                    # Pass failure_reason to add_result if it exists
                    failure_reason = item["log_data"].get("failure_reason")
                    wandb_logger.add_result(
                        original_item,
                        item["result"],
                        item["log_data"]["response"],
                        failure_reason
                    )
                    
                    # Count failure types for batch summary
                    if failure_reason == "timeout":
                        batch_log["timeout_count"] = batch_log.get("timeout_count", 0) + 1
                    elif failure_reason == "parsing_failure":
                        batch_log["parsing_failure_count"] = batch_log.get("parsing_failure_count", 0) + 1
                
                # Add to batch log - only include necessary fields
                if "log_data" in item:
                    batch_item = {
                        "id": item["log_data"]["original_id"],
                        "content": item["log_data"]["content"],
                        "state": item["log_data"]["state"],
                        "response": item["log_data"]["response"],
                        "answer": item["log_data"]["answer"],
                    }
                    
                    # Add failure_reason if it exists
                    if "failure_reason" in item["log_data"] and item["log_data"]["failure_reason"]:
                        batch_item["failure_reason"] = item["log_data"]["failure_reason"]
                        
                    batch_log["items"].append(batch_item)
            else:
                logger.warning(f"UUID not found in position map: {item['uuid']}")
        
        # Log batch completion time
        logger.info(f"Batch processed in {batch_time:.2f} seconds ({len(batch_results)} items)")
        
        # Prepare for retry if needed
        if next_batch:
            retry_count += 1
            logger.info(f"Retrying {len(next_batch)} failed items (attempt {retry_count}/{max_retries})")
            current_batch = next_batch
            time.sleep(1)  # Small delay before retry
        else:
            break
    
    # Calculate overall metrics
    overall_time = time.time() - overall_start_time
    success_count = sum(1 for r in results if r > 0.5)
    success_rate = success_count / len(results) if results else 0
    
    # Update batch log with final metrics
    batch_log["success_rate"] = success_rate
    batch_log["total_time_sec"] = overall_time
    batch_log["avg_time_per_item_sec"] = overall_time / len(inputs) if inputs else 0
    
    # Log step to wandb
    wandb_logger.log_step(1)  # Using 1 since it's a single batch
    
    # Read existing batches if file exists
    all_batches = []
    if os.path.exists(batch_log_file):
        try:
            with open(batch_log_file, 'r') as f:
                all_batches = json.load(f)
                # If the file just has one batch (not in a list), convert to list
                if not isinstance(all_batches, list):
                    all_batches = [all_batches]
        except json.JSONDecodeError:
            # If file is corrupted, start fresh
            logger.warning(f"Could not read existing batch log file: {batch_log_file}. Starting fresh.")
            all_batches = []
    
    # Add current batch to all batches
    all_batches.append(batch_log)
    
    # Write all batches back to file
    with open(batch_log_file, 'w') as f:
        json.dump(all_batches, f, indent=2)
    
    # Calculate error statistics
    timeout_count = batch_log.get("timeout_count", 0)
    parsing_failure_count = batch_log.get("parsing_failure_count", 0)
    
    # Log overall summary
    logger.info(f"LLM judge completed: {success_count}/{len(results)} passed ({success_rate:.2%})")
    logger.info(f"Timeout errors: {timeout_count}, Parsing failures: {parsing_failure_count}")
    logger.info(f"Total time: {overall_time:.2f} seconds, Average: {batch_log['avg_time_per_item_sec']:.2f} seconds per item")
    logger.info(f"Batch appended to log file: {batch_log_file}")
    
    # Finish wandb logging
    wandb_logger.finish()
    
    return results

async def llm_judge(inputs: List[Dict[str, Any]]) -> List[float]:
    """
    Original async version - maintained for backwards compatibility.
    New code should use run_llm_judge instead.
    
    Args:
        inputs: A list of dicts with eval inputs
        
    Returns:
        list: A list of scores (0.0 or 1.0) for each input.
    """
    return run_llm_judge(inputs)

def load_config() -> Dict[str, Any]:
    """Load Hydra configuration safely"""
    try:
        # Clear existing Hydra instance if it exists
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        
        with hydra.initialize(version_base=None, config_path=CONFIG_PATH):
            config = hydra.compose(config_name=CONFIG_NAME)
        return OmegaConf.to_container(config, resolve=True)
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        return get_default_config()

def get_default_config() -> Dict[str, Any]:
    """Return default configuration when Hydra is not available"""
    return {
        "name": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "max_parallel_requests": 10,
        "temperature": 0.1,
        "max_tokens": 200,
        "max_retries": 3,
        "request_timeout": 15,  # Default timeout in seconds
        "log_dir": "./logs/llm_judge"
    }