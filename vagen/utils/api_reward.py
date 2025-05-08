from typing import Dict, Tuple, Any
import asyncio
import re
import json
import os
import time
from together import AsyncTogether

PROMPT_TEMPLATE = """
Compare the natural language description with the state information dictionary.
Answer YES if the description accurately matches the state, or NO if it doesn't.
Think step by step and end with your answer in <answer>YES</answer> or <answer>NO</answer> format.

State Information:
{state_information_dict}

Description:
"{natural_language_description}"

Your answer should be within {api_max_tokens} tokens and MUST end with <answer>YES</answer> or <answer>NO</answer>.
"""


def extract_answer(response_text: str) -> str:
    """Extract YES/NO answer from LLM response"""
    match = re.search(r"<answer>(YES|NO)</answer>", response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Fallback: Check if YES or NO appears in the text
    if "YES" in response_text.upper() and "NO" not in response_text.upper():
        return "YES"
    elif "NO" in response_text.upper() and "YES" not in response_text.upper():
        return "NO"
    
    return None

async def get_api_scores_batch(
    step_batch_result: Dict[Any, Tuple[Dict, float, bool, Dict]], 
    model: str,
    max_parallel_requests: int = 3  # 添加并发控制参数
) -> Dict[Any, Tuple[Dict, float, bool, Dict]]:
    """Process API evaluation requests in batch"""
    eval_items = []
    
    # Using SDK's built-in retry mechanism with exponential backoff
    async_client = AsyncTogether(api_key=os.getenv("TOGETHER_API_KEY"), max_retries=3)
    
    for env_id, (_, _, _, info) in step_batch_result.items():
        if "api_evaluation" in info and info["api_evaluation"]:
            if "state_information" in info and "natural_language_description" in info:
                prompt = PROMPT_TEMPLATE.format(
                    state_information_dict=json.dumps(info["state_information"], indent=2),
                    natural_language_description=info["natural_language_description"],
                    api_max_tokens=info.get("api_max_tokens", 200)
                )
                
                eval_items.append({
                    "env_id": env_id,
                    "prompt": prompt,
                    "weight": info.get("api_evaluation_weight", 1.0),
                    "api_temperature": info.get("api_temperature", 0.1),
                    "api_max_tokens": info.get("api_max_tokens", 200)
                })
    
    if not eval_items:
        return step_batch_result
    
    # Create modifiable copy of results
    new_step_batch_result = {env_id: list(step_data) for env_id, step_data in step_batch_result.items()}
    
    # Process items in parallel using Together SDK
    async def process_item(item):
        try:
            messages = [{"role": "user", "content": item["prompt"]}]
            
            response = await async_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=item["api_temperature"],
                max_tokens=item["api_max_tokens"]
            )
            
            return {
                "env_id": item["env_id"],
                "response": response.choices[0].message.content,
                "answer": extract_answer(response.choices[0].message.content),
                "weight": item["weight"]
            }
        except Exception as e:
            print(f"Error in API call: {str(e)}")
            return {"env_id": item["env_id"], "answer": None}
    
    # 添加信号量来限制并发请求数
    semaphore = asyncio.Semaphore(max_parallel_requests)
    
    async def process_with_semaphore(item):
        async with semaphore:
            return await process_item(item)
    
    # 使用semaphore控制并发
    eval_results = await asyncio.gather(*[process_with_semaphore(item) for item in eval_items])
    
    # Update rewards and metrics based on results
    for result in eval_results:
        env_id = result["env_id"]
        answer = result.get("answer")
        
        if answer:
            info = new_step_batch_result[env_id][3]
            
            # Initialize metrics if needed
            if "metrics" not in info:
                info["metrics"] = {}
            if "turn_metrics" not in info["metrics"]:
                info["metrics"]["turn_metrics"] = {}
            
            # Save response and result
            info["api_evaluation_response"] = result.get("response", "")
            info["api_evaluation_result"] = answer
            info["metrics"]["turn_metrics"]["api_evaluation"] = 1.0 if answer == "YES" else 0.0
            
            # Update reward
            if answer == "YES":
                new_step_batch_result[env_id][1] += 1.0 * result["weight"]
    
    # Convert back to tuples
    return {env_id: tuple(step_data) for env_id, step_data in new_step_batch_result.items()}

if __name__ == "__main__":
    from vagen.env.frozenlake import FrozenLakeEnv, FrozenLakeEnvConfig
    
    model = "Qwen/Qwen3-235B-A22B-fp8"
    
    # Create environments for testing
    base_configs = [
        FrozenLakeEnvConfig(size=5, is_slippery=False), 
        FrozenLakeEnvConfig(size=5, is_slippery=True)
    ]
    environments = [FrozenLakeEnv(config) for config in base_configs]
    
    # Setup and run environments
    step_batch_result = {}
    for i, env in enumerate(environments):
        obs, _ = env.reset(seed=42+i)
        
        # Take step with simple action
        action = "<answer>Right</answer>" if i == 0 else "<answer>Down</answer>"
        obs, reward, done, info = env.step(action)
        
        # Add api evaluation fields
        player_pos = env._get_player_position()
        row, col = player_pos
        
        info["api_evaluation"] = True
        info["api_evaluation_weight"] = 1.0
        info["state_information"] = {
            "player_position": {"row": int(row), "col": int(col)},
            "map_size": env.config.size,
            "is_slippery": env.config.is_slippery,
            "map_description": [[cell.decode('utf-8') for cell in row] for row in env.gym_env.desc]
        }
        info["api_temperature"] = 0.7
        info["api_max_tokens"] = 300
        info["natural_language_description"] = (
            f"The player is at position ({row},{col}) on a {env.config.size}x{env.config.size} grid." 
            if i == 0 else 
            f"The player is at position ({row+1},{col-1}) on a {env.config.size}x{env.config.size} grid."
        )
        
        step_batch_result[i] = (obs, reward, done, info)
    
    # Run api evaluation
    print("Starting API evaluation...")
    print(f"Using up to 3 concurrent API requests")
    start_time = time.time()
    results = asyncio.run(get_api_scores_batch(step_batch_result, model, max_parallel_requests=3))
    print(f"Completed in {time.time() - start_time:.2f} seconds")
    
    # Print results
    for env_id, (_, reward, _, info) in results.items():
        print(f"\nEnvironment {env_id}:")
        print(f"Reward: {reward}")
        print(f"API Result: {info.get('api_evaluation_result', 'N/A')}")
    
    # Clean up
    for env in environments:
        env.close()