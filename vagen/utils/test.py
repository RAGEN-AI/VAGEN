import asyncio
import json
import os
import time
import numpy as np
from typing import Dict, List, Tuple, Any

# 导入您的llm_judge函数
from vagen.env.utils.llm_judge import llm_judge  # 请将your_module替换为您实际的模块名

# 导入环境注册表
from vagen.env import REGISTERED_ENV

# 自定义JSON编码器处理numpy类型
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tuple) and any(isinstance(x, (np.integer, np.floating)) for x in obj):
            return tuple(int(x) if isinstance(x, np.integer) else 
                        float(x) if isinstance(x, np.floating) else x 
                        for x in obj)
        return super(NumpyEncoder, self).default(obj)

def convert_numpy_types(obj):
    """递归转换numpy类型为Python原生类型以便JSON序列化"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, tuple) and any(isinstance(x, (np.integer, np.floating)) for x in obj):
        return tuple(int(x) if isinstance(x, np.integer) else 
                    float(x) if isinstance(x, np.floating) else x 
                    for x in obj)
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

def ensure_json_serializable(data):
    """确保所有数据都可JSON序列化，转换任何有问题的类型"""
    if isinstance(data, dict):
        return {str(k): ensure_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [ensure_json_serializable(item) for item in data]
    elif isinstance(data, tuple):
        return list(ensure_json_serializable(item) for item in data)
    elif isinstance(data, (int, float, str, bool, type(None))):
        return data
    else:
        # 将任何其他类型转换为字符串
        return str(data)

def create_custom_test_cases(env_name, state):
    """创建用于测试的自定义匹配和不匹配描述"""
    test_cases = []
    
    if env_name == "frozenlake":
        # 为frozenlake创建用户定义的描述
        matching_desc = (
            "The player is positioned near the center of the map, with the goal tucked away in "
            "the upper left corner. There are two dangerous holes scattered on the small 4x4 grid "
            "that must be carefully avoided during navigation."
        )
        test_cases.append((matching_desc, True))
        
        non_matching_desc = (
            "The player is on the bottom edge of the map and the goal is directly above. "
            "There are seven holes surrounding the path, making it extremely dangerous to "
            "navigate the massive 10x10 grid."
        )
        test_cases.append((non_matching_desc, False))
    
    elif env_name == "sokoban":
        # 为sokoban创建用户定义的描述
        matching_desc = (
            "######\n"
            "#PX_O#\n"
            "#____#\n"
            "#____#\n"
            "#____#\n"
            "######"
        )
        test_cases.append((matching_desc, True))
        
        non_matching_desc = (
            "###########\n"
            "#____O____#\n"
            "#_________#\n"
            "#___P_____#\n"
            "#_________#\n"
            "#XX_______#\n"
            "###########"
        )
        test_cases.append((non_matching_desc, False))
    
    elif env_name == "navigation":
        # 为navigation创建用户定义的描述
        matching_desc = (
            "There's a Box ahead about 2.65 meters away. To reach it efficiently, we should move "
            "forward and slightly to the right. The path appears to be clear without any immediate "
            "obstacles, so we can proceed directly toward the target."
        )
        test_cases.append((matching_desc, True))
        
        non_matching_desc = (
            "There's a Chair very far away, approximately 15 meters in the distance. To reach it, "
            "we need to navigate around a large table directly in front of us, then turn left at "
            "the bookshelf and proceed past three potted plants."
        )
        test_cases.append((non_matching_desc, False))
    
    elif env_name == "primitive_skill":
        # 为primitive_skill创建用户定义的描述
        matching_desc = (
            "The red cube is positioned in the upper area of the workspace, while the green cube "
            "is located below it toward the bottom of the workspace. The red cube is closer to "
            "the left side than the green cube."
        )
        test_cases.append((matching_desc, True))
        
        non_matching_desc = (
            "The blue cube is balanced precariously on the edge of the yellow cylinder. The red "
            "cube is nowhere to be seen, while the green cube has been split into two smaller "
            "triangular prisms at opposite corners of the workspace."
        )
        test_cases.append((non_matching_desc, False))
    
    # 如果没有创建特定的测试用例，使用通用方法（后备）
    if not test_cases:
        print(f"警告：未为{env_name}定义自定义描述。使用后备描述。")
        # 环境未知时的后备描述
        matching_desc = f"The {env_name} environment is in its current state."
        test_cases.append((matching_desc, True))
        
        non_matching_desc = f"The {env_name} environment is in a completely different state."
        test_cases.append((non_matching_desc, False))
    
    return test_cases

async def test_llm_judge():
    """使用环境get_env_state方法测试LLM评判"""
    # 检查API密钥
    if not os.getenv("TOGETHER_API_KEY"):
        print("错误：未设置TOGETHER_API_KEY环境变量")
        return
    
    # 要使用的模型
    model = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
    print(f"使用模型测试LLM评判: {model}")
    
    # 定义要测试的环境
    env_names = ["sokoban", "frozenlake", "navigation", "primitive_skill"]
    
    # 存储结果以供报告
    all_results = {}
    
    # 处理每个环境
    for env_name in env_names:
        print(f"\n--- 测试环境: {env_name} ---")
        
        # 如果环境不可用则跳过
        if env_name not in REGISTERED_ENV:
            print(f"在已注册环境中未找到环境 {env_name}")
            continue
        
        try:
            # 创建环境实例
            env_cls = REGISTERED_ENV[env_name]['env_cls']
            config_cls = REGISTERED_ENV[env_name]['config_cls']
            env = env_cls(config_cls())
            
            # 使用固定种子重置环境以便可重现
            obs, _ = env.reset(seed=42)
            
            # 使用提供的get_env_state方法获取环境状态
            state = env.get_env_state()
            print(f"成功使用get_env_state()从{env_name}提取状态")
            
            # 将状态中的numpy类型转换为Python原生类型以便JSON序列化
            state = convert_numpy_types(state)
            # 确保状态完全可JSON序列化
            state = ensure_json_serializable(state)
            
            # 创建自定义测试用例（匹配和不匹配描述）
            test_cases = create_custom_test_cases(env_name, state)
            
            # 如果无法创建测试用例则跳过
            if not test_cases:
                print(f"跳过 {env_name} - 无法创建合适的测试用例")
                continue
            
            # 为llm_judge准备输入
            inputs = []
            
            for i, (desc, should_match) in enumerate(test_cases):
                # 创建输入字典
                input_item = {
                    "id": f"{env_name}_{i}",
                    "content": desc,
                    "state": state,
                    "env_name": env_name
                }
                
                # 添加到输入列表
                inputs.append(input_item)
            
            # 运行LLM评判
            print(f"为{len(test_cases)}个测试用例运行LLM评判...")
            start_time = time.time()
            scores = await llm_judge(inputs=inputs)
            elapsed = time.time() - start_time
            print(f"在{elapsed:.2f}秒内完成")
            
            # 处理并存储结果
            env_results = []
            for i, ((desc, should_match), score) in enumerate(zip(test_cases, scores)):
                # 确定API结果是YES还是NO
                api_result = "YES" if score > 0.5 else "NO"
                is_correct = (api_result == "YES") == should_match
                
                # 存储仅所需的字段
                case_result = {
                    "natural_language_description": desc,
                    "api_result": api_result,
                    "score": score,
                    "should_match": should_match,
                    "is_correct": is_correct
                }
                env_results.append(case_result)
                
                # 打印结果摘要
                match_status = "匹配" if should_match else "不匹配"
                correct_str = "✓" if is_correct else "✗"
                print(f"  用例 {i+1} ({match_status}): LLM判断 {api_result} {correct_str}")
                print(f"    得分: {score}")
            
            # 存储此环境的所有结果
            all_results[env_name] = env_results
            
            # 清理
            env.close()
            
        except Exception as e:
            print(f"测试{env_name}时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 打印最终报告
    print("\n=== 最终报告 ===")
    for env_name, results in all_results.items():
        correct_count = sum(1 for r in results if r["is_correct"])
        accuracy = correct_count / len(results) if results else 0
        print(f"{env_name}: {correct_count}/{len(results)} 正确 ({accuracy*100:.1f}%)")
    
    # 在保存前验证结果是完全可JSON序列化的
    try:
        # 测试序列化
        json.dumps(all_results, cls=NumpyEncoder)
        print("已验证结果可JSON序列化")
    except (TypeError, ValueError) as e:
        print(f"错误：结果不可JSON序列化: {str(e)}")
        print("尝试修复序列化问题...")
        all_results = ensure_json_serializable(all_results)
    
    # 将详细结果保存到文件，仅包含所需字段
    with open("llm_judge_results.json", "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print("\n详细结果已保存到llm_judge_results.json")

if __name__ == "__main__":
    print("开始在各环境中测试LLM评判...")
    asyncio.run(test_llm_judge())