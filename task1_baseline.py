"""
Task 1: Baseline Implementation
Zero-shot and Few-shot prompting for GSM8K math reasoning

Author: Assignment 1
Date: 2025
"""

import json
import time
import os
from typing import List, Dict, Tuple
from tqdm import tqdm
from api_client import PoeAPIClient
from config import initialize_api_client, DEFAULT_CONFIG
from data.GSM8K.evaluation import extract_ans_from_response


# Few-shot examples for GSM8K
FEW_SHOT_EXAMPLES = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "reasoning": "There are 15 trees originally. Then there were 21 trees after the Grove workers planted some more. So there must have been 21 - 15 = 6 trees that were planted.",
        "answer": "6"
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "reasoning": "There are originally 3 cars. Then 2 more cars arrive. Now 3 + 2 = 5 cars are in the parking lot.",
        "answer": "5"
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "reasoning": "Originally, Leah had 32 chocolates and her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39 pieces left in total.",
        "answer": "39"
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "reasoning": "Jason had 20 lollipops originally. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8 lollipops.",
        "answer": "8"
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "reasoning": "Shawn started with 5 toys. He then got 2 toys each from his mom and dad. So he got 2 * 2 = 4 more toys. Now he has 5 + 4 = 9 toys.",
        "answer": "9"
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "reasoning": "There were originally 9 computers. For each day from monday to thursday, 5 more computers were installed. So 4 * 5 = 20 computers were added. Now 9 + 20 = 29 computers are now in the server room.",
        "answer": "29"
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "reasoning": "Michael started with 58 golf balls. He lost 23 on Tuesday, and lost 2 more on wednesday. So he had 58 - 23 = 35 at the end of Tuesday, and 35 - 2 = 33 at the end of wednesday.",
        "answer": "33"
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "reasoning": "Olivia had 23 dollars. She bought 5 bagels for 3 dollars each. So she spent 5 * 3 = 15 dollars. Now she has 23 - 15 = 8 dollars left.",
        "answer": "8"
    }
]

SYSTEM_PROMPT_ZERO_SHOT = """You are an expert math problem solver. Your task is to solve math word problems step by step.

When solving:
1. Read the problem carefully
2. Identify what is being asked
3. Show your reasoning process clearly
4. Provide the final numerical answer

Always end your response with: #### [answer]
where [answer] is the numerical result."""

SYSTEM_PROMPT_FEW_SHOT = """You are an expert math problem solver. Your task is to solve math word problems step by step.

When solving:
1. Read the problem carefully
2. Identify what is being asked
3. Show your reasoning process clearly
4. Provide the final numerical answer

Always end your response with: #### [answer]
where [answer] is the numerical result.

Study the following examples carefully to understand the solution format."""


def load_gsm8k_dataset(dataset_path: str) -> Tuple[List[str], List[str]]:
    """
    Load GSM8K dataset from JSONL file
    
    Args:
        dataset_path: Path to the JSONL file
        
    Returns:
        Tuple of (questions, answers) lists
    """
    questions = []
    answers = []
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            questions.append(data['question'])
            # Extract answer after "#### "
            answer = data['answer'].split('#### ')[-1].strip()
            answers.append(answer)
    
    return questions, answers


def build_zero_shot_messages(question: str) -> List[Dict[str, str]]:
    """
    Build message list for zero-shot prompting
    
    Args:
        question: The math problem question
        
    Returns:
        List of message dictionaries
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT_ZERO_SHOT},
        {"role": "user", "content": f"Question: {question}\n\nPlease solve this step by step."}
    ]


def build_few_shot_messages(question: str, num_examples: int = 5) -> List[Dict[str, str]]:
    """
    Build message list for few-shot prompting
    
    Args:
        question: The math problem question
        num_examples: Number of demonstration examples to include
        
    Returns:
        List of message dictionaries
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_FEW_SHOT}
    ]
    
    # Add demonstration examples
    examples_to_use = FEW_SHOT_EXAMPLES[:num_examples]
    for example in examples_to_use:
        messages.append({
            "role": "user",
            "content": f"Question: {example['question']}"
        })
        messages.append({
            "role": "assistant",
            "content": f"{example['reasoning']}\n\n#### {example['answer']}"
        })
    
    # Add the actual question
    messages.append({
        "role": "user",
        "content": f"Question: {question}\n\nPlease solve this step by step."
    })
    
    return messages


def run_zero_shot_baseline(client: PoeAPIClient, questions: List[str], answers: List[str], 
                           max_problems: int = None, output_file: str = "zeroshot.baseline.jsonl") -> Dict:
    """
    Run zero-shot baseline on GSM8K
    
    Args:
        client: PoeAPIClient instance
        questions: List of questions
        answers: List of ground truth answers
        max_problems: Maximum number of problems to evaluate (None for all)
        output_file: Output JSONL file path
        
    Returns:
        Dictionary with accuracy and timing metrics
    """
    print(f"\n{'='*60}")
    print("TASK 1: ZERO-SHOT BASELINE")
    print(f"{'='*60}")
    
    max_problems = max_problems or len(questions)
    num_correct = 0
    total_tokens_generated = 0
    start_time = time.time()
    
    results = []
    
    with tqdm(total=max_problems, desc="Zero-shot Baseline") as pbar:
        for idx in range(min(max_problems, len(questions))):
            question = questions[idx]
            ground_truth = answers[idx]
            
            # Build messages
            messages = build_zero_shot_messages(question)
            
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
                
                # Convert to string for comparison
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
                    "response": response_text,
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
                    "response": None,
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
        "method": "zero-shot",
        "accuracy": accuracy,
        "num_correct": num_correct,
        "total_problems": max_problems,
        "wall_clock_time": wall_clock_time,
        "avg_tokens_per_problem": avg_tokens_per_problem,
        "total_tokens": total_tokens_generated,
        "output_file": output_file
    }
    
    print(f"\nZero-shot Results:")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Correct: {num_correct}/{max_problems}")
    print(f"  Wall-clock time: {wall_clock_time:.2f}s")
    print(f"  Avg tokens per problem: {avg_tokens_per_problem:.1f}")
    print(f"  Total tokens: {total_tokens_generated}")
    print(f"  Results saved to: {output_file}")
    
    return metrics


def run_few_shot_baseline(client: PoeAPIClient, questions: List[str], answers: List[str], 
                          num_examples: int = 5, max_problems: int = None, 
                          output_file: str = "fewshot.baseline.jsonl") -> Dict:
    """
    Run few-shot baseline on GSM8K
    
    Args:
        client: PoeAPIClient instance
        questions: List of questions
        answers: List of ground truth answers
        num_examples: Number of demonstration examples
        max_problems: Maximum number of problems to evaluate (None for all)
        output_file: Output JSONL file path
        
    Returns:
        Dictionary with accuracy and timing metrics
    """
    print(f"\n{'='*60}")
    print(f"TASK 1: FEW-SHOT BASELINE ({num_examples}-shot)")
    print(f"{'='*60}")
    
    max_problems = max_problems or len(questions)
    num_correct = 0
    total_tokens_generated = 0
    start_time = time.time()
    
    results = []
    
    with tqdm(total=max_problems, desc=f"Few-shot Baseline ({num_examples}-shot)") as pbar:
        for idx in range(min(max_problems, len(questions))):
            question = questions[idx]
            ground_truth = answers[idx]
            
            # Build messages
            messages = build_few_shot_messages(question, num_examples=num_examples)
            
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
                
                # Convert to string for comparison
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
                    "response": response_text,
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
                    "response": None,
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
        "method": f"few-shot-{num_examples}",
        "accuracy": accuracy,
        "num_correct": num_correct,
        "total_problems": max_problems,
        "wall_clock_time": wall_clock_time,
        "avg_tokens_per_problem": avg_tokens_per_problem,
        "total_tokens": total_tokens_generated,
        "output_file": output_file
    }
    
    print(f"\nFew-shot Results ({num_examples}-shot):")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Correct: {num_correct}/{max_problems}")
    print(f"  Wall-clock time: {wall_clock_time:.2f}s")
    print(f"  Avg tokens per problem: {avg_tokens_per_problem:.1f}")
    print(f"  Total tokens: {total_tokens_generated}")
    print(f"  Results saved to: {output_file}")
    
    return metrics


def main():
    """Main function to run Task 1 baselines"""
    
    print("\n" + "="*60)
    print("TASK 1: BASELINE IMPLEMENTATION FOR GSM8K")
    print("="*60)
    
    # Initialize API client
    print("\nInitializing API client...")
    client = initialize_api_client()
    
    # Load dataset
    dataset_path = "data/GSM8K/test.jsonl"
    print(f"Loading dataset from {dataset_path}...")
    questions, answers = load_gsm8k_dataset(dataset_path)
    print(f"Loaded {len(questions)} questions")
    
    # Run baselines
    max_test_problems = DEFAULT_CONFIG['max_problems']  # Start with fewer problems for testing
    
    # Zero-shot baseline
    zero_shot_metrics = run_zero_shot_baseline(
        client, questions, answers,
        max_problems=max_test_problems,
        output_file="zeroshot.baseline.jsonl"
    )
    
    # Few-shot baseline
    few_shot_metrics = run_few_shot_baseline(
        client, questions, answers,
        num_examples=5,
        max_problems=max_test_problems,
        output_file="fewshot.baseline.jsonl"
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("TASK 1 SUMMARY")
    print(f"{'='*60}")
    print(f"Zero-shot accuracy: {zero_shot_metrics['accuracy']:.2f}%")
    print(f"Few-shot accuracy: {few_shot_metrics['accuracy']:.2f}%")
    print(f"Improvement: {few_shot_metrics['accuracy'] - zero_shot_metrics['accuracy']:.2f}%")
    

if __name__ == "__main__":
    main()
