"""
The file contains pure QA class for evaluation and training
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple
import re


def extract_elements(
    pred: str,
    expected_type: type = str,
    clean_pattern: Optional[str] = None,
) -> Optional[Union[List[Any], Tuple[Any, Any]]]:
    """
    Extract elements from a raw input string, supporting both pairs and lists.
    
    Handles multiple input formats:
    - Parentheses format: (a, b) or (a, b, c, d)
    - Brackets format: [a, b] or [a, b, c, d]
    - Comma-separated list: a, b, c, d
    
    Args:
        pred (str): The raw input string
        expected_type (type): The expected type of elements (str, int, etc.)
        clean_pattern (Optional[str]): Optional regex pattern to remove from values before conversion
            
    Returns:
        Optional[Union[List[Any], Tuple[Any, Any]]]:
        - For pairs (force_pair=True): Tuple of (item1, item2) or None if extraction fails
        - For lists: List of extracted elements or None if extraction fails
    """
    if not pred or not isinstance(pred, str):
        return None
    
    try:
        content = pred
        
        # Check for parentheses format
        if '(' in pred and ')' in pred:
            match = re.search(r'\((.*?)\)', pred)
            if match:
                content = match.group(1)
        
        # Check for brackets format
        elif '[' in pred and ']' in pred:
            match = re.search(r'\[(.*?)\]', pred)
            if match:
                content = match.group(1)
        
        # Parse the elements
        items = []
        for item in content.split(','):
            # Strip whitespace and quotes
            cleaned = item.strip().strip("'\"")
            
            if not cleaned:  # Skip empty items
                continue
            
            # Apply additional cleaning if pattern provided
            if clean_pattern:
                cleaned = re.sub(clean_pattern, '', cleaned, flags=re.IGNORECASE)
            
            # Convert to the appropriate type
            if expected_type is int:
                try:
                    items.append(int(cleaned))
                except ValueError:
                    return None
            elif expected_type is float:
                try:
                    items.append(float(cleaned))
                except ValueError:
                    return None
            else:  # Default to string or other type
                items.append(expected_type(cleaned) if expected_type != str else cleaned)
        
        if not items:
            return None
        return items
        
    except Exception:
        # Handle any parsing errors gracefully
        return None




def exp_evaluate_fn(
        pred: str,
        relationships_to_query: List[Dict],
) -> bool:
    """
    Evaluate if the predicted string matches the ground truth relationships to query
    
    Args:
        pred (str): the raw predicted string
            1. relationship query:
                - Parentheses: (object1, object2)
                - Brackets: [object1, object2]
            2. termination:
                - "terminate"
        relationships_to_query (List[Dict]): the ground truth relationships to query
            - key: object1, object2, direction
        
    Returns:
        bool, True if the predicted string matches the ground truth relationships to query, False otherwise
    """
    
    if not pred or not isinstance(pred, str):
            return False
        
    # 1. First check if the answer is about termination
    if not relationships_to_query: # empty relationship_to_query means no unknown pairs for current room
        return "terminate" in pred.lower()
    
    # 2. Then check if the answer is about a relationship
    # 2.1 Extract the predicted pair
    pred_pair = extract_elements(pred, expected_type=str)
    if not pred_pair or len(pred_pair) != 2:
        return False
    
    pred_obj1, pred_obj2 = pred_pair
    
    # 2.2 Check if the predicted objects match any of the relationships to query
    for rel in relationships_to_query:
        gt_obj1 = rel['object1']
        gt_obj2 = rel['object2']
        
        # Check if the objects match (in either order)
        if ((pred_obj1.lower() == gt_obj1.lower() and pred_obj2.lower() == gt_obj2.lower()) or
            (pred_obj1.lower() == gt_obj2.lower() and pred_obj2.lower() == gt_obj1.lower())):
            return True
    
    return False




def direction_evaluate_fn(
        pred: str,
        direction: str,
) -> bool:
    """
    Evaluate if the predicted direction matches the ground truth direction
    - ground truth direction: (H, V)
    - predicted direction: raw string, (H, V), [H, V], ...
    """
    if not pred or not isinstance(pred, str):
        return False
    pair = extract_elements(pred, expected_type=str)
    
    if not pair or len(pair) != 2:
        return False
        
    pred_h, pred_v = pair
    return f"({pred_h}, {pred_v})".lower() == direction.lower()
    

def object_sequence_evaluate_fn(
        pred: str,
        gt_object_sequence: List[str],
) -> bool:
    """
    Evaluate if the predicted object sequence matches the ground truth.
    
    Handles multiple input formats:
    - Array-like format: [a, b, c, d]
    - Comma-separated list: a, b, c, d
    - JSON-formatted array: ["a", "b", "c", "d"]
    
    Returns:
        bool: True if the predicted sequence matches the ground truth, False otherwise.
    """
    if not pred or not isinstance(pred, str):
        return False
    
    pred_objects = extract_elements(pred, str)
    
    if not pred_objects:
        return False
        
    # Quick length check
    if len(pred_objects) != len(gt_object_sequence):
        return False
    
    # Compare items (case-insensitive)
    for pred_obj, gt_obj in zip(pred_objects, gt_object_sequence):
        if pred_obj.lower() != gt_obj.lower():
            return False
            
    return True

def degree_sequence_evaluate_fn(
        pred: str,
        gt_degree_sequence: List[int],
) -> bool:
    """
    Evaluate if the predicted sequence of degrees matches the ground truth.
    
    Handles multiple input formats:
    - Array-like format: [90, -45, 180, 30]
    - Comma-separated list: 90, -45, 180, 30
    - JSON-formatted array: [90, -45, 180, 30]
    
    Returns:
        bool: True if the predicted degree sequence matches the ground truth, False otherwise.
    """
    if not pred or not isinstance(pred, str):
        return False
    
    pred_degrees = extract_elements(pred, int, clean_pattern=r'[°degrees\s]')
    
    if not pred_degrees:
        return False
        
    # Check if the sequences match in length and values
    if len(pred_degrees) != len(gt_degree_sequence):
        return False

    for pred_deg, gt_deg in zip(pred_degrees, gt_degree_sequence):
        if pred_deg != gt_deg:
            return False
    
    return True



@dataclass
class QA:
    """
    A class for question and answer pair
    
    Attributes:
        dataset_name (str): name of the dataset (stepgame, spatial_qa, ...)
        question_type (str): type of the question (exploration, direction, rotation, ego2allo, allo2ego)
        type (str): type of the SpatialQA (EgoSS, EgoSD, ...)
    """
    question: str
    answer: str | List[str] | List[int]
    question_type: str
    type: str
    extra_data: Dict[str, Any] = field(default_factory=dict)
    dataset_name: str = field(default='spatial_qa', init=False)
    

    def __post_init__(self):
        if self.question_type not in ['exploration', 'direction', 'rotation', 'ego2allo', 'allo2ego']:
            raise ValueError(f"Invalid question type: {self.question_type}")

    def to_dict(self) -> Dict:
        return {
            'question': self.question,
            'answer': self.answer,
            'question_type': self.question_type,
            'dataset_name': self.dataset_name,
            'type': self.type,
            'extra_data': self.extra_data,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SpatialQA':
        return cls(
            question=data['question'],
            answer=data['answer'],
            extra_data=data['extra_data'],
            question_type=data['question_type'],
            type=data['type'],
        )

    def evaluate(self, pred: str) -> bool:
        """
        Evaluate if the answer is correct
        """
        if self.question_type == 'exploration':
            return exp_evaluate_fn(pred, self.answer)
        elif self.question_type == 'direction':
            return direction_evaluate_fn(pred, self.answer)
        elif self.question_type == 'rotation' or self.question_type == 'ego2allo':
            return object_sequence_evaluate_fn(pred, self.answer)
        elif self.question_type == 'allo2ego':
            return degree_sequence_evaluate_fn(pred, self.answer)