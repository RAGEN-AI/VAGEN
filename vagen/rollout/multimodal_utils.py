from typing import List, Dict, Tuple, Optional
import PIL.Image
import torch
import re


def get_multimodal_handler(model_type: str):
    """Factory function to get the appropriate multimodal handler based on model type.
    
    Args:
        model_type: Model type, either 'qwen' or 'internvl'
        
    Returns:
        Handler function for the specified model type
    """
    if model_type.lower() == 'qwen':
        return handle_qwen_multimodal_data
    elif model_type.lower() == 'internvl':
        return handle_internvl_multimodal_data
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: 'qwen', 'internvl'")


def handle_qwen_multimodal_data(
    prompt_template: str,
    row_dict: Dict,
    image_data: List[PIL.Image.Image],
    processor,
    do_embedding: bool = True,
) -> Tuple[str, Dict, Optional[torch.Tensor], str]:
    """Handle multi-modal data for Qwen models.
    
    Args:
        prompt_template: Template string with <image> placeholders
        row_dict: Dictionary to store processed data
        image_data: List of PIL images
        processor: Qwen processor
        do_embedding: Whether to do embedding (True) or prepare for vllm (False)
        
    Returns:
        Tuple of (prompt_template, row_dict, image_grid_thw, raw_prompt)
    """
    assert len(image_data) == prompt_template.count('<image>'), \
        'Number of images does not match number of <image> in the prompt template'
    
    raw_prompt = prompt_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
    row_dict['multi_modal_data'] = {'image': image_data}
    image_grid_thw = None
    
    if do_embedding:
        image_inputs = processor.image_processor(image_data, return_tensors='pt')
        image_grid_thw = image_inputs['image_grid_thw']
        row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
    
    if image_grid_thw is not None:
        merge_length = processor.image_processor.merge_size**2
        index = 0
        while '<image>' in prompt_template:
            prompt_template = prompt_template.replace(
                '<image>',
                '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                '<|vision_end|>',
                1,
            )
            index += 1
        
        prompt_template = prompt_template.replace('<|placeholder|>', processor.image_token)
    
    return prompt_template, row_dict, image_grid_thw, raw_prompt


def handle_internvl_multimodal_data(
    prompt_template: str,
    row_dict: Dict,
    image_data: List[PIL.Image.Image],
    processor,
    do_embedding: bool = True,
) -> Tuple[str, Dict, Optional[torch.Tensor], str]:
    """Handle multi-modal data for InternVL models for huggingface version.
    
    Args:
        prompt_template: Template string with <image> placeholders
        row_dict: Dictionary to store processed data
        image_data: List of PIL images
        processor: InternVL processor
        do_embedding: Whether to do embedding (True) or prepare for vllm (False)
        
    Returns:
        Tuple of (prompt_template, row_dict, image_grid_thw, raw_prompt)
    """
    assert len(image_data) == prompt_template.count('<image>'), \
        'Number of images does not match number of <image> in the prompt template'
    
    # For InternVL, raw_prompt uses the standard image tokens
    # raw_prompt = prompt_template.replace('<image>', 
    #     f'{processor.tokenizer.start_image_token}{processor.tokenizer.context_image_token}{processor.tokenizer.end_image_token}')
    # raw_prompt = prompt_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
    raw_prompt = prompt_template
    row_dict['multi_modal_data'] = {'image': image_data} # vllm can automatically handle the data with <image>
    
    if do_embedding:
        proxy_text = [f'<IMG_CONTEXT><SPLIT>' * len(image_data)]
        model_inputs = processor(text=proxy_text, images=image_data, return_tensors='pt')
        input_ids = model_inputs['input_ids'] # 1 * D
        decoded_text = processor.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        image_tokens = decoded_text.split('<SPLIT>')
        for image_token in image_tokens:
            prompt_template = prompt_template.replace('<image>', image_token, 1)
        row_dict['multi_modal_inputs'] = {'pixel_values': model_inputs['pixel_values']}
    
    # InternVL doesn't use image_grid_thw, so return None
    return prompt_template, row_dict, None, raw_prompt


def handle_kimivl_multimodal_data(
    prompt_template: str,
    row_dict: Dict,
    image_data: List[PIL.Image.Image],
    processor,
    do_embedding: bool = True,
) -> Tuple[str, Dict, Optional[torch.Tensor], str]:
    """Handle multi-modal data for KimiVL models.
    
    Args:
        prompt_template: Template string with <image> placeholders
        row_dict: Dictionary to store processed data
        image_data: List of PIL images
        processor: InternVL processor
        do_embedding: Whether to do embedding (True) or prepare for vllm (False)
        
    Returns:
        Tuple of (prompt_template, row_dict, image_grid_thw, raw_prompt)
    """
    assert len(image_data) == prompt_template.count('<image>'), \
        'Number of images does not match number of <image> in the prompt template'
    
    raw_prompt = prompt_template
    row_dict['multi_modal_data'] = {'image': image_data} # vllm can automatically handle the data with <image>
    image_grid_hws = None
    
    if do_embedding:
        image_inputs = processor.image_processor(image_data, return_tensors='pt')
        image_grid_hws = image_inputs['image_grid_hws']
        row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
    
    if image_grid_hws is not None:
        merge_length = processor.image_processor.merge_kernel_size[0] * processor.image_processor.merge_kernel_size[1]
        index = 0
        while '<image>' in prompt_template:
            prompt_template = prompt_template.replace(
                '<image>',
                '<|media_start|>' + '<|media_content|>' * (image_grid_hws[index].prod() // merge_length) +
                '<|media_end|>',
                1,
            )
            index += 1
        

    return prompt_template, row_dict, None, raw_prompt


def detect_model_type(processor) -> str:
    """Detect model type based on processor characteristics.
    
    Args:
        processor: Model processor
        
    Returns:
        Model type string ('qwen' or 'internvl')
    """
    # Check if it's InternVL by looking for specific attributes
    if 'qwen' in processor.__class__.__name__.lower():
        return 'qwen'
    elif 'intern' in processor.__class__.__name__.lower():
        return 'internvl'
    elif 'kimi' in processor.__class__.__name__.lower():
        return 'kimivl'
    else:
        raise ValueError(f"Unsupported model type: {processor.__class__.__name__}. Supported types: 'qwen', 'internvl', 'kimivl'")