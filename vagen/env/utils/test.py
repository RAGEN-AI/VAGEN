import time
import os
from together_batch_request import run_together_request


def test_rate_limited_timing():
    """Test the rate-limited version with 256 prompts and 40 parallel requests"""
    
    # Create realistic prompt for grounding task
    base_prompt = """Compare the state and description. Answer YES if they match exactly or NO if they don't.

State: {
    "player_position": {"row": 1, "col": 2},
    "map_size": 5,
    "is_slippery": false,
    "map_grid": [
        ["S", "F", "F", "F", "F"],
        ["F", "F", "F", "F", "H"],
        ["F", "F", "F", "F", "F"],
        ["H", "F", "F", "F", "F"],
        ["F", "F", "G", "F", "F"]
    ]
}

Description: "The player is currently on the third tile from the left in the second row. There is a hole directly above and another hole two tiles to the left in the bottom row. The goal is located directly below the player."

<answer>"""
    
    # Create 256 prompts
    prompts = [base_prompt] * 256
    
    # Configuration with rate limiting settings
    config = {
        "name": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "temperature": 0.1,
        "max_tokens": 200,
        "max_retries": 3,
        "batch_size": 200,  # Process in batches
        
        # Rate limiting configuration
        "qps_limit": 70,    # Queries per second
        "rpm_limit": 4000,  # Requests per minute
        "tps_limit": 15000, # Tokens per second
        "retry_delay": 1,   # Base retry delay
        "request_timeout": 120,
    }
    
    print(f"Starting test with {len(prompts)} prompts...")
    print(f"Model: {config['name']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Rate limits: QPS={config['qps_limit']}, RPM={config['rpm_limit']}, TPS={config['tps_limit']}")
    print("-" * 60)
    
    # Run and time the test
    start_time = time.time()
    results = run_together_request(prompts, config)
    end_time = time.time()
    
    # Calculate metrics
    execution_time = end_time - start_time
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    print(f"\nExecution time: {execution_time:.2f} seconds")
    print(f"Successful requests: {successful}/{len(results)}")
    print(f"Failed requests: {failed}")
    print(f"Requests per second: {len(results)/execution_time:.2f}")
    
    # Show retry statistics
    retry_counts = {}
    for result in results:
        retries = result.get("retries", 0)
        retry_counts[retries] = retry_counts.get(retries, 0) + 1
    
    print(f"\nRetry statistics:")
    for retry_count, count in sorted(retry_counts.items()):
        print(f"  {retry_count} retries: {count} requests")
    
    # Show sample responses
    if successful > 0:
        print("\nSample responses:")
        shown = 0
        for i, result in enumerate(results):
            if result["success"] and shown < 3:
                response = result["response"].strip()
                print(f"  Response {shown + 1}: {response[:100]}...")
                shown += 1
    
    # Show any errors
    if failed > 0:
        print("\nError types:")
        error_types = {}
        for result in results:
            if not result["success"]:
                error_type = result.get("error", "Unknown")
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        for error, count in error_types.items():
            print(f"  {error}: {count} occurrences")
            
        # Show first error detail
        print("\nFirst error detail:")
        for result in results:
            if not result["success"]:
                print(f"  {result.get('error', 'Unknown error')}")
                break


def test_rate_limiting_behavior():
    """Test to specifically observe rate limiting behavior"""
    
    # Simple prompt
    simple_prompt = "Is 2+2=4? Answer YES or NO."
    
    # Run with high QPS to trigger rate limiting
    config = {
        "name": "Qwen/Qwen3-235B-A22B-fp8-tput",
        "temperature": 0.1,
        "max_tokens": 50,
        "max_retries": 3,
        "batch_size": 100,  # Large batch size
        
        # More restrictive rate limits to observe behavior
        "qps_limit": 30,   # Lower QPS limit
        "rpm_limit": 1000, # Lower RPM limit
        "tps_limit": 8000, # Lower TPS limit
        "retry_delay": 1,
        "request_timeout": 120,
    }
    
    print("\n" + "="*60)
    print("Testing rate limiting behavior with 100 requests...")
    print(f"QPS limit: {config['qps_limit']}")
    print(f"RPM limit: {config['rpm_limit']}")
    print(f"TPS limit: {config['tps_limit']}")
    print("-" * 60)
    
    prompts = [simple_prompt] * 100
    
    start_time = time.time()
    results = run_together_request(prompts, config)
    end_time = time.time()
    
    execution_time = end_time - start_time
    successful = sum(1 for r in results if r["success"])
    
    print(f"\nExecution time: {execution_time:.2f} seconds")
    print(f"Successful requests: {successful}/{len(results)}")
    print(f"Actual QPS: {successful/execution_time:.2f}")
    
    # Show rate-limited requests
    rate_limited = sum(1 for r in results if "rate_limit" in str(r.get("error", "")).lower())
    print(f"Rate-limited requests: {rate_limited}")


if __name__ == "__main__":
    # Run the main timing test
    test_rate_limited_timing()
    
    # Run rate limiting behavior test
    test_rate_limiting_behavior()