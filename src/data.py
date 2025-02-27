import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import torch
from tqdm import tqdm

def extract_arithmetic_answer(text: str) -> Optional[float]:
    """
    Extract the final answer from a solution text for an arithmetic problem.
    
    Args:
        text: Solution text
        
    Returns:
        Extracted answer as a float, or None if no answer found
    """
    # Look for patterns like "= 42" or "answer is 42" at the end of the text
    answer_patterns = [
        r"=\s*(-?\d+\.?\d*)\s*$",  # = 42
        r"=\s*(-?\d+\.?\d*)[^\d]*$",  # = 42.
        r"answer\s+is\s+(-?\d+\.?\d*)[^\d]*$",  # answer is 42
        r"result\s+is\s+(-?\d+\.?\d*)[^\d]*$",  # result is 42
        r"equals\s+(-?\d+\.?\d*)[^\d]*$",  # equals 42
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    
    # If no match found with the patterns above, try to find the last number in the text
    numbers = re.findall(r"(-?\d+\.?\d*)", text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    
    return None

def extract_intermediate_calculations(text: str) -> List[Tuple[str, float]]:
    """
    Extract intermediate calculations from a solution text.
    
    Args:
        text: Solution text
        
    Returns:
        List of (expression, result) tuples
    """
    # Look for patterns like "2 + 2 = 4" or "2 * 3 = 6"
    calculation_pattern = r"([\d\s\+\-\*\/\(\)]+)\s*=\s*(-?\d+\.?\d*)"
    
    calculations = []
    for match in re.finditer(calculation_pattern, text):
        expression = match.group(1).strip()
        result_str = match.group(2)
        try:
            result = float(result_str)
            calculations.append((expression, result))
        except ValueError:
            continue
    
    return calculations

def create_hidden_variable_dataset(
    problems: List[str],
    solutions: List[str],
    activations: np.ndarray
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Create a dataset with hidden variables extracted from solutions.
    
    Args:
        problems: List of problem statements
        solutions: List of solution texts
        activations: Activation matrix (n_tokens, hidden_dim)
        
    Returns:
        Tuple of (activations, hidden_variables_dict)
    """
    # Extract problem parameters
    problem_params = []
    for problem in problems:
        # Extract numbers and operation from problems like "Calculate 23 + 45."
        match = re.search(r"Calculate\s+(\d+)\s*([+\-*/])\s*(\d+)", problem)
        if match:
            a = int(match.group(1))
            op = match.group(2)
            b = int(match.group(3))
            problem_params.append((a, op, b))
        else:
            problem_params.append((None, None, None))
    
    # Extract answers and intermediate calculations
    final_answers = []
    intermediate_calcs = []
    
    for solution in solutions:
        answer = extract_arithmetic_answer(solution)
        final_answers.append(answer if answer is not None else float('nan'))
        
        calcs = extract_intermediate_calculations(solution)
        intermediate_calcs.append(calcs)
    
    # Create hidden variables
    hidden_vars = {}
    
    # 1. Final answer
    hidden_vars['final_answer'] = np.array(final_answers)
    
    # 2. Correct answer (ground truth)
    correct_answers = []
    for a, op, b in problem_params:
        if a is not None and op is not None and b is not None:
            if op == '+':
                correct_answers.append(a + b)
            elif op == '-':
                correct_answers.append(a - b)
            elif op == '*':
                correct_answers.append(a * b)
            elif op == '/':
                correct_answers.append(a / b if b != 0 else float('nan'))
        else:
            correct_answers.append(float('nan'))
    hidden_vars['correct_answer'] = np.array(correct_answers)
    
    # 3. Is the answer correct?
    is_correct = np.isclose(hidden_vars['final_answer'], hidden_vars['correct_answer'])
    hidden_vars['is_correct'] = is_correct.astype(float)
    
    # 4. Number of intermediate calculations
    num_calcs = np.array([len(calcs) for calcs in intermediate_calcs])
    hidden_vars['num_calculations'] = num_calcs
    
    # 5. First operand
    first_operands = np.array([a if a is not None else float('nan') for a, _, _ in problem_params])
    hidden_vars['first_operand'] = first_operands
    
    # 6. Second operand
    second_operands = np.array([b if b is not None else float('nan') for _, _, b in problem_params])
    hidden_vars['second_operand'] = second_operands
    
    # 7. Operation type (one-hot encoded)
    op_types = []
    for _, op, _ in problem_params:
        if op == '+':
            op_types.append([1, 0, 0, 0])
        elif op == '-':
            op_types.append([0, 1, 0, 0])
        elif op == '*':
            op_types.append([0, 0, 1, 0])
        elif op == '/':
            op_types.append([0, 0, 0, 1])
        else:
            op_types.append([0, 0, 0, 0])
    hidden_vars['operation_type'] = np.array(op_types)
    
    return activations, hidden_vars

def create_token_level_dataset(
    solutions: List[str],
    activations_list: List[np.ndarray],
    tokenizer
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Create a dataset at the token level with token-specific hidden variables.
    
    Args:
        solutions: List of solution texts
        activations_list: List of activation matrices for each solution
        tokenizer: Tokenizer to convert text to tokens
        
    Returns:
        Tuple of (activations, hidden_variables_dict)
    """
    all_activations = []
    token_texts = []
    token_positions = []
    is_number = []
    is_operator = []
    is_equals = []
    
    for i, solution in enumerate(solutions):
        # Tokenize the solution
        tokens = tokenizer.encode(solution)
        token_strings = [tokenizer.decode([t]) for t in tokens]
        
        # Get activations for this solution
        solution_acts = activations_list[i]
        
        # Make sure dimensions match
        if len(tokens) != solution_acts.shape[0]:
            print(f"Warning: Token count mismatch for solution {i}. Skipping.")
            continue
        
        # Add to dataset
        all_activations.append(solution_acts)
        token_texts.extend(token_strings)
        token_positions.extend(list(range(len(tokens))))
        
        # Token type features
        for t in token_strings:
            is_number.append(1.0 if re.match(r"^\s*\d+\s*$", t) else 0.0)
            is_operator.append(1.0 if re.match(r"^\s*[+\-*/]\s*$", t) else 0.0)
            is_equals.append(1.0 if re.match(r"^\s*=\s*$", t) else 0.0)
    
    # Stack all activations
    activations = np.vstack(all_activations)
    
    # Create hidden variables dictionary
    hidden_vars = {
        'token_position': np.array(token_positions),
        'is_number': np.array(is_number),
        'is_operator': np.array(is_operator),
        'is_equals': np.array(is_equals),
    }
    
    return activations, hidden_vars, token_texts 