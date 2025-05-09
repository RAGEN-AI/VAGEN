from typing import List, Dict, Any, Optional, Tuple
import asyncio
import re
import json
import os
import time
import logging
import hydra
import uuid
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from together import AsyncTogether

# Import wandb logger
from .llm_judge_utils import WandbLogger

CONFIG_NAME = "llm_judge"
CONFIG_PATH = "./"

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
        "log_dir": "./logs/llm_judge",
        "wandb": {
            "enabled": True,
            "project_name": "llm-judge",
            "run_name": None  # Will be auto-generated if None
        }
    }

def setup_logging(config):
    """Set up logging for the LLM judge module"""
    import logging
    import os
    from datetime import datetime
    
    log_dir = config.get("log_dir", "./logs/llm_judge")
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"llm_judge_{timestamp}.log")
    
    # Configure logging
    logger = logging.getLogger("llm_judge")
    logger.setLevel(logging.INFO)
    
    # Check if handlers already exist to avoid duplicates
    if not logger.handlers:
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

def run_llm_judge(
    inputs: List[Dict[str, Any]], 
    wandb_config: Optional[Dict[str, Any]] = None
) -> List[float]:
    """
    Run LLM judge synchronously by handling asyncio internally.
    This is the main function to call from other modules.
    
    Args:
        inputs: A list of dicts: {"id": id, "content": content, "state": state, "type": "observation"/"prediction"}
        wandb_config: Optional configuration for wandb logging
        
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
    
    # Initialize wandb logger if enabled
    wandb_logger = None
    if config.get("wandb", {}).get("enabled", False) or wandb_config:
        try:
            import wandb
            # Merge wandb_config with config.wandb
            if wandb_config is None:
                wandb_config = {}
            merged_wandb_config = {
                **config.get("wandb", {}),
                **wandb_config
            }
            
            # Initialize wandb logger
            wandb_logger = WandbLogger(
                project_name=merged_wandb_config.get("project_name", "llm-judge"),
                run_name=merged_wandb_config.get("run_name"),
                config=config,
                log_dir=config.get("log_dir", "./logs/llm_judge"),
                resume=merged_wandb_config.get("resume", False)
            )
            logger.info(f"WandB logging enabled: project={merged_wandb_config.get('project_name')}, run={merged_wandb_config.get('run_name')}")
        except ImportError:
            logger.warning("WandB package not found. Install with 'pip install wandb' to enable logging.")
        except Exception as e:
            logger.error(f"Error initializing WandB: {str(e)}")
    
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
        "avg_time_per_item_sec": 0
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
        for template_env in templates.keys():
            if template_env.lower() in str(item["env_name"]).lower():
                env_name = template_env
                break
        
        if not env_name:
            # Default to frozenlake if no match
            env_name = "frozenlake"
            logger.warning(f"No matching template for env_name '{item['env_name']}', using default: {env_name}")
        
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
            "type": item["type"]  # Store type for wandb categorization
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
                            item_state=item["state"],  # Pass state to the function
                            item_type=item["type"]  # Pass type for categorization
                        )
                        return {
                            "uuid": item["uuid"],  # Use UUID for tracking
                            "result": result, 
                            "success": True, 
                            "log_data": log_data
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
                        logger.info(f"Item {log_entry['original_id']} - Answer: {log_entry['answer']}")
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
                
                # Add to batch log - only include necessary fields
                if "log_data" in item:
                    batch_log["items"].append({
                        "id": item["log_data"]["original_id"],  # Use original ID for logging
                        "content": item["log_data"]["content"],
                        "state": item["log_data"]["state"],
                        "response": item["log_data"]["response"],
                        "answer": item["log_data"]["answer"],
                        "type": item["log_data"]["type"]  # Include type for wandb categorization
                    })
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
    
    # Log to wandb if enabled
    if wandb_logger:
        wandb_logger.log_batch(batch_log["items"], overall_time)
    
    # Log overall summary
    logger.info(f"LLM judge completed: {success_count}/{len(results)} passed ({success_rate:.2%})")
    logger.info(f"Total time: {overall_time:.2f} seconds, Average: {batch_log['avg_time_per_item_sec']:.2f} seconds per item")
    logger.info(f"Batch appended to log file: {batch_log_file}")
    
    return results

async def async_judge_single_item(
    prompt: str, 
    config: Dict[str, Any], 
    logger: logging.Logger, 
    original_id: Any, 
    item_uuid: str,
    item_content: str,
    item_state: Dict[str, Any],  # Add state parameter
    item_type: str = "observation"  # Add type parameter
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
        item_state: State of the item for logging
        item_type: Type of the item (observation/prediction)
        
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
        answer = extract_answer(response_text)
        is_parsing_failure = not bool(re.search(r"<answer>(YES|NO)</answer>", response_text, re.IGNORECASE))
        
    except asyncio.TimeoutError:
        # Handle timeout case
        logger.warning(f"Request for item {original_id} timed out after {request_timeout} seconds")
        response_time = time.time() - start_time
        response_text = f"[TIMEOUT AFTER {request_timeout} SECONDS]"
        answer = "NO"  # Default to NO for timeouts
        is_parsing_failure = False  # Not a parsing failure, it's a timeout
    except Exception as e:
        # Handle other errors
        logger.error(f"Error in API call for item {original_id}: {str(e)}")
        response_time = time.time() - start_time
        response_text = f"[ERROR: {str(e)}]"
        answer = "NO"  # Default to NO for errors
        is_parsing_failure = False  # Not a parsing failure, it's an error
    
    # Create detailed log data with all necessary information
    log_data = {
        "original_id": original_id,
        "uuid": item_uuid,
        "model": model,
        "content": item_content,
        "state": item_state,  # Include state for logging
        "response": response_text,
        "answer": answer,
        "type": item_type,  # Include type for categorization in wandb
        "response_time": response_time,
        "is_timeout": "[TIMEOUT" in response_text,
        "is_parsing_failure": is_parsing_failure
    }
    
    return (1.0 if answer == "YES" else 0.0), log_data