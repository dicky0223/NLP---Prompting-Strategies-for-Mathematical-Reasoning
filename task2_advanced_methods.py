"""
Task 2: Advanced Prompting Methods
Chain-of-Thought (CoT) and Self-Verification for GSM8K math reasoning

Methods:
1. Chain-of-Thought (CoT): Decompose reasoning step by step
2. Self-Verification: Generate multiple solutions and verify consistency

Author: Assignment 1
Date: 2025
"""

import json
import time
import re
from typing import List, Dict, Tuple
from collections import Counter
from tqdm import tqdm
from api_client import PoeAPIClient
from config import initialize_api_client, DEFAULT_CONFIG
from data.GSM8K.evaluation import extract_ans_from_response


# CoT System Prompt
SYSTEM_PROMPT_COT = """You are an expert math problem solver. Your task is to solve math word problems using chain-of-thought reasoning.

IMPORTANT: You MUST break down the problem into clear, logical steps:
1. Identify what we know (given information)
2. Identify what we need to find
3. Break down the solution into intermediate steps
4. Show all calculations clearly
5. Verify your answer makes sense

Format your response exactly as follows:
Given: [list what is given in the problem]
Find: [what the problem asks]
Step 1: [first reasoning step with calculation]
Step 2: [second reasoning step with calculation]
Step 3: [continue as needed]
...
Verification: [briefly check if answer makes sense]
#### [final numerical answer]

Always end with #### [answer] where [answer] is just the number."""

# Self-Verification System Prompt
SYSTEM_PROMPT_SELF_VERIFICATION = """You are an expert math problem solver and validator. Your task is to:
1. Generate a solution to a math problem
2. Verify your own solution by re-reading the problem
3. Check if your answer logically follows from the problem statement

Generate your solution step by step, then provide verification:

Solution:
[Your step-by-step solution]

Verification Check:
- Does the answer match what the problem asks?
- Are all calculations correct?
- Does the answer make logical sense?

#### [final numerical answer]"""


# CoT Examples for few-shot
COT_EXAMPLES = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "cot_reasoning": """Given: 15 trees initially, 21 trees after planting
Find: Number of trees planted
Step 1: Identify the change in number of trees
  Before planting: 15 trees
  After planting: 21 trees
Step 2: Calculate trees planted
  Trees planted = After - Before = 21 - 15 = 6 trees
Verification: If we start with 15 and add 6, we get 21. ✓
#### 6"""
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "cot_reasoning": """Given: Leah has 32 chocolates, sister has 42, they ate 35 total
Find: Total chocolates remaining
Step 1: Calculate total chocolates initially
  Total = 32 + 42 = 74 chocolates
Step 2: Subtract chocolates eaten
  Remaining = 74 - 35 = 39 chocolates
Verification: 39 + 35 = 74, and 32 + 42 = 74. ✓
#### 39"""
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "cot_reasoning": """Given: Shawn initially has 5 toys, gets 2 from mom and 2 from dad
Find: Total toys now
Step 1: Calculate toys received from parents
  From mom: 2 toys
  From dad: 2 toys
  Total received: 2 + 2 = 4 toys
Step 2: Calculate total toys
  Total = Initial + Received = 5 + 4 = 9 toys
Verification: 5 initial toys + 4 new toys = 9 total. ✓
#### 9"""
    }
]


def build_cot_messages(question: str, use_examples: bool = True) -> List[Dict[str, str]]:
    """
    Build message list for Chain-of-Thought prompting
    
    Args:
        question: The math problem question
        use_examples: Whether to include few-shot examples
        
    Returns:
        List of message dictionaries
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_COT}
    ]
    
    if use_examples:
        # Add CoT examples
        for example in COT_EXAMPLES:
            messages.append({
                "role": "user",
                "content": f"Question: {example['question']}"
            })
            messages.append({
                "role": "assistant",
                "content": example['cot_reasoning']
            })
    
    # Add the actual question
    messages.append({
        "role": "user",
        "content": f"Question: {question}"
    })
    
    return messages


def build_self_verification_messages(question: str) -> List[Dict[str, str]]:
    """
    Build message list for Self-Verification prompting
    
    Args:
        question: The math problem question
        
    Returns:
        List of message dictionaries
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_SELF_VERIFICATION},
        {"role": "user", "content": f"Question: {question}"}
    ]
    
    return messages


def extract_multiple_answers(response_text: str, num_attempts: int = 1) -> List[str]:
    """
    Extract answer from response, handling multiple possible answer formats
    
    Args:
        response_text: The response text from the model
        num_attempts: Number of answer extraction attempts
        
    Returns:
        List of extracted answers
    """
    answers = []
    
    # Try to extract answer from #### format
    if "####" in response_text:
        answer = extract_ans_from_response(response_text)
        if answer is not None:
            answers.append(str(answer))
    
    # If no answer found, try extracting numbers from the response
    if not answers:
        # Find all numbers that could be answers
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response_text)
        if numbers:
            # Usually the last number or the one closest to the end is the answer
            answers.append(numbers[-1])
    
    return answers if answers else [None]


def run_cot_method(client: PoeAPIClient, questions: List[str], answers: List[str],
                   max_problems: int = None, output_file: str = "cot.jsonl") -> Dict:
    """
    Run Chain-of-Thought method on GSM8K
    
    Args:
        client: PoeAPIClient instance
        questions: List of questions
        answers: List of ground truth answers
        max_problems: Maximum number of problems to evaluate
        output_file: Output JSONL file path
        
    Returns:
        Dictionary with accuracy and timing metrics
    """
    print(f"\n{'='*60}")
    print("TASK 2 METHOD 1: CHAIN-OF-THOUGHT (CoT)")
    print(f"{'='*60}")
    
    max_problems = max_problems or len(questions)
    num_correct = 0
    total_tokens_generated = 0
    start_time = time.time()
    
    results = []
    
    with tqdm(total=max_problems, desc="Chain-of-Thought") as pbar:
        for idx in range(min(max_problems, len(questions))):
            question = questions[idx]
            ground_truth = answers[idx]
            
            # Build messages with CoT prompting
            messages = build_cot_messages(question, use_examples=True)
            
            # Get response from API
            try:
                response_data = client.query_claude_sonnet(
                    messages=messages,
                    max_tokens=DEFAULT_CONFIG['max_tokens'],
                    temperature=DEFAULT_CONFIG['temperature']
                )
                
                response_text = response_data['content']
                tokens_used = response_data['usage']['completion_tokens']
                total_tokens_generated += tokens_used
                
                # Extract answer
                predicted_answer = extract_ans_from_response(response_text)
                
                # Check correctness
                try:
                    ground_truth_num = int(float(ground_truth))
                    predicted_num = int(float(str(predicted_answer)))
                    is_correct = ground_truth_num == predicted_num
                except (ValueError, TypeError):
                    is_correct = str(predicted_answer).strip() == str(ground_truth).strip()
                
                if is_correct:
                    num_correct += 1
                
                results.append({
                    "question_id": idx,
                    "question": question,
                    "predicted_answer": str(predicted_answer),
                    "ground_truth": ground_truth,
                    "is_correct": is_correct,
                    "reasoning": response_text,
                    "tokens_used": tokens_used
                })
                
            except Exception as e:
                print(f"Error processing question {idx}: {e}")
                results.append({
                    "question_id": idx,
                    "question": question,
                    "predicted_answer": None,
                    "ground_truth": ground_truth,
                    "is_correct": False,
                    "reasoning": None,
                    "tokens_used": 0,
                    "error": str(e)
                })
            
            pbar.update(1)
    
    end_time = time.time()
    wall_clock_time = end_time - start_time
    
    # Calculate metrics
    accuracy = (num_correct / max_problems) * 100
    avg_tokens_per_problem = total_tokens_generated / max_problems if max_problems > 0 else 0
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    metrics = {
        "method": "chain-of-thought",
        "accuracy": accuracy,
        "num_correct": num_correct,
        "total_problems": max_problems,
        "wall_clock_time": wall_clock_time,
        "avg_tokens_per_problem": avg_tokens_per_problem,
        "total_tokens": total_tokens_generated,
        "output_file": output_file
    }
    
    print(f"\nChain-of-Thought Results:")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Correct: {num_correct}/{max_problems}")
    print(f"  Wall-clock time: {wall_clock_time:.2f}s")
    print(f"  Avg tokens per problem: {avg_tokens_per_problem:.1f}")
    print(f"  Total tokens: {total_tokens_generated}")
    print(f"  Results saved to: {output_file}")
    
    return metrics


def run_self_verification_method(client: PoeAPIClient, questions: List[str], answers: List[str],
                                 num_attempts: int = 3, max_problems: int = None,
                                 output_file: str = "self_verification.jsonl") -> Dict:
    """
    Run Self-Verification method on GSM8K
    
    Args:
        client: PoeAPIClient instance
        questions: List of questions
        answers: List of ground truth answers
        num_attempts: Number of solution attempts for verification
        max_problems: Maximum number of problems to evaluate
        output_file: Output JSONL file path
        
    Returns:
        Dictionary with accuracy and timing metrics
    """
    print(f"\n{'='*60}")
    print(f"TASK 2 METHOD 2: SELF-VERIFICATION (with {num_attempts} attempts)")
    print(f"{'='*60}")
    
    max_problems = max_problems or len(questions)
    num_correct = 0
    total_tokens_generated = 0
    start_time = time.time()
    
    results = []
    
    with tqdm(total=max_problems, desc="Self-Verification") as pbar:
        for idx in range(min(max_problems, len(questions))):
            question = questions[idx]
            ground_truth = answers[idx]
            
            # Generate multiple solutions
            solutions = []
            extracted_answers = []
            attempt_tokens = 0
            
            for attempt in range(num_attempts):
                # Build messages for verification
                messages = build_self_verification_messages(question)
                
                # Get response from API
                try:
                    response_data = client.query_claude_sonnet(
                        messages=messages,
                        max_tokens=DEFAULT_CONFIG['max_tokens'],
                        temperature=0.7 + (attempt * 0.1)  # Vary temperature across attempts
                    )
                    
                    response_text = response_data['content']
                    tokens_used = response_data['usage']['completion_tokens']
                    attempt_tokens += tokens_used
                    total_tokens_generated += tokens_used
                    
                    solutions.append(response_text)
                    
                    # Extract answer from this attempt
                    answer = extract_ans_from_response(response_text)
                    if answer is not None:
                        extracted_answers.append(str(answer))
                    
                except Exception as e:
                    print(f"Error in attempt {attempt + 1} for question {idx}: {e}")
            
            # Determine final answer by voting/consensus
            if extracted_answers:
                # Count occurrences of each answer
                answer_counts = Counter(extracted_answers)
                # Select the most common answer
                most_common_answer = answer_counts.most_common(1)[0][0]
                consensus_count = answer_counts.most_common(1)[0][1]
                predicted_answer = most_common_answer
            else:
                predicted_answer = None
                consensus_count = 0
            
            # Check correctness
            try:
                ground_truth_num = int(float(ground_truth))
                predicted_num = int(float(str(predicted_answer)))
                is_correct = ground_truth_num == predicted_num
            except (ValueError, TypeError):
                is_correct = str(predicted_answer).strip() == str(ground_truth).strip()
            
            if is_correct:
                num_correct += 1
            
            results.append({
                "question_id": idx,
                "question": question,
                "predicted_answer": str(predicted_answer),
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "num_attempts": num_attempts,
                "extracted_answers": extracted_answers,
                "consensus_count": consensus_count,
                "solutions": solutions,
                "tokens_used": attempt_tokens
            })
            
            pbar.update(1)
    
    end_time = time.time()
    wall_clock_time = end_time - start_time
    
    # Calculate metrics
    accuracy = (num_correct / max_problems) * 100
    avg_tokens_per_problem = total_tokens_generated / max_problems if max_problems > 0 else 0
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    metrics = {
        "method": f"self-verification-{num_attempts}",
        "accuracy": accuracy,
        "num_correct": num_correct,
        "total_problems": max_problems,
        "wall_clock_time": wall_clock_time,
        "avg_tokens_per_problem": avg_tokens_per_problem,
        "total_tokens": total_tokens_generated,
        "num_attempts": num_attempts,
        "output_file": output_file
    }
    
    print(f"\nSelf-Verification Results ({num_attempts} attempts):")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Correct: {num_correct}/{max_problems}")
    print(f"  Wall-clock time: {wall_clock_time:.2f}s")
    print(f"  Avg tokens per problem: {avg_tokens_per_problem:.1f}")
    print(f"  Total tokens: {total_tokens_generated}")
    print(f"  Results saved to: {output_file}")
    
    return metrics


def main():
    """Main function to run Task 2 advanced methods"""
    
    print("\n" + "="*60)
    print("TASK 2: ADVANCED PROMPTING METHODS FOR GSM8K")
    print("="*60)
    
    # Initialize API client
    print("\nInitializing API client...")
    client = initialize_api_client()
    
    # Load dataset
    dataset_path = "data/GSM8K/test.jsonl"
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    questions = [item['question'] for item in data]
    answers = [item['answer'].split('#### ')[-1].strip() for item in data]
    print(f"Loaded {len(questions)} questions")
    
    max_test_problems = DEFAULT_CONFIG['max_problems']
    
    # Method 1: Chain-of-Thought
    cot_metrics = run_cot_method(
        client, questions, answers,
        max_problems=max_test_problems,
        output_file="cot.jsonl"
    )
    
    # Method 2: Self-Verification
    sv_metrics = run_self_verification_method(
        client, questions, answers,
        num_attempts=3,
        max_problems=max_test_problems,
        output_file="self_verification.jsonl"
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("TASK 2 SUMMARY")
    print(f"{'='*60}")
    print(f"Chain-of-Thought accuracy: {cot_metrics['accuracy']:.2f}%")
    print(f"Self-Verification accuracy: {sv_metrics['accuracy']:.2f}%")
    

if __name__ == "__main__":
    main()
