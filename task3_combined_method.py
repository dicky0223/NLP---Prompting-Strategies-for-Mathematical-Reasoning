"""
Task 3: Combined Method - CoT + Self-Verification
Combines Chain-of-Thought reasoning with Self-Verification to achieve better performance

The idea:
1. Generate multiple CoT solutions (with different temperatures)
2. Extract and verify answers using self-verification logic
3. Select the most consistent answer across attempts

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


# Combined CoT + Verification System Prompt
SYSTEM_PROMPT_COMBINED = """You are an expert math problem solver with exceptional verification skills.

Your task:
1. Solve the problem using step-by-step chain-of-thought reasoning
2. Verify each step is correct
3. Double-check the final answer

Format your response EXACTLY as:

Given: [What we know from the problem]
Find: [What the problem asks for]

Solution Steps:
Step 1: [First reasoning step]
Step 2: [Second reasoning step]
...

Verification:
- Check all calculations are correct
- Verify the logic of each step
- Confirm the answer addresses what was asked

#### [FINAL ANSWER - just the number]

CRITICAL: Always end with #### followed by ONLY the numerical answer."""


COMBINED_EXAMPLES = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "response": """Given: Initially 15 trees, finally 21 trees
Find: How many trees were planted

Solution Steps:
Step 1: Identify what changed
  - We started with 15 trees
  - We ended with 21 trees
  - The difference is what was planted

Step 2: Calculate the difference
  - Trees planted = 21 - 15 = 6 trees

Verification:
- Does this make sense? 15 + 6 = 21 ✓
- Is the calculation correct? 21 - 15 = 6 ✓
- Does this answer the question? Yes, 6 trees were planted ✓

#### 6"""
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "response": """Given: Leah has 32, sister has 42, they ate 35 total
Find: Total chocolates remaining

Solution Steps:
Step 1: Find total chocolates initially
  - Leah: 32 chocolates
  - Sister: 42 chocolates
  - Total: 32 + 42 = 74 chocolates

Step 2: Subtract what they ate
  - Remaining: 74 - 35 = 39 chocolates

Verification:
- Total calculation correct? 32 + 42 = 74 ✓
- Subtraction correct? 74 - 35 = 39 ✓
- Does it make sense? They had 74, ate 35, left with 39 ✓

#### 39"""
    }
]


def build_combined_messages(question: str, use_examples: bool = True) -> List[Dict[str, str]]:
    """
    Build message list for combined CoT + Self-Verification
    
    Args:
        question: The math problem question
        use_examples: Whether to include few-shot examples
        
    Returns:
        List of message dictionaries
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_COMBINED}
    ]
    
    if use_examples:
        # Add examples
        for example in COMBINED_EXAMPLES:
            messages.append({
                "role": "user",
                "content": f"Question: {example['question']}"
            })
            messages.append({
                "role": "assistant",
                "content": example['response']
            })
    
    # Add the actual question
    messages.append({
        "role": "user",
        "content": f"Question: {question}"
    })
    
    return messages


def run_combined_method(client: PoeAPIClient, questions: List[str], answers: List[str],
                        num_attempts: int = 3, max_problems: int = None,
                        output_file: str = "combined_cot_verification.jsonl") -> Dict:
    """
    Run combined CoT + Self-Verification method on GSM8K
    
    Args:
        client: PoeAPIClient instance
        questions: List of questions
        answers: List of ground truth answers
        num_attempts: Number of solution attempts
        max_problems: Maximum number of problems to evaluate
        output_file: Output JSONL file path
        
    Returns:
        Dictionary with accuracy and timing metrics
    """
    print(f"\n{'='*60}")
    print(f"TASK 3: COMBINED METHOD - CoT + Self-Verification")
    print(f"({num_attempts} attempts with temperature variation)")
    print(f"{'='*60}")
    
    max_problems = max_problems or len(questions)
    num_correct = 0
    total_tokens_generated = 0
    start_time = time.time()
    
    results = []
    
    with tqdm(total=max_problems, desc="CoT + Self-Verification") as pbar:
        for idx in range(min(max_problems, len(questions))):
            question = questions[idx]
            ground_truth = answers[idx]
            
            # Generate multiple solutions with varying temperatures
            solutions = []
            extracted_answers = []
            attempt_tokens = 0
            temperatures = [0.0, 0.3, 0.5]  # Conservative, moderate, exploratory
            
            for attempt in range(num_attempts):
                # Use different temperatures for diversity
                temp = temperatures[attempt] if attempt < len(temperatures) else 0.7
                
                # Build messages
                messages = build_combined_messages(question, use_examples=True)
                
                # Get response from API
                try:
                    response_data = client.query_claude_sonnet(
                        messages=messages,
                        max_tokens=DEFAULT_CONFIG['max_tokens'],
                        temperature=temp
                    )
                    
                    response_text = response_data['content']
                    tokens_used = response_data['usage']['completion_tokens']
                    attempt_tokens += tokens_used
                    total_tokens_generated += tokens_used
                    
                    solutions.append({
                        'response': response_text,
                        'temperature': temp,
                        'tokens': tokens_used
                    })
                    
                    # Extract answer
                    answer = extract_ans_from_response(response_text)
                    if answer is not None:
                        extracted_answers.append(str(answer))
                    
                except Exception as e:
                    print(f"Error in attempt {attempt + 1} for question {idx}: {e}")
            
            # Determine final answer by consensus
            if extracted_answers:
                # Count occurrences of each answer
                answer_counts = Counter(extracted_answers)
                # Get the most common answer
                most_common = answer_counts.most_common(1)[0]
                predicted_answer = most_common[0]
                consensus_score = most_common[1]
                all_same = consensus_score == len(extracted_answers)
            else:
                predicted_answer = None
                consensus_score = 0
                all_same = False
            
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
                "consensus_score": consensus_score,
                "all_same_answer": all_same,
                "solutions": [
                    {
                        'temperature': s['temperature'],
                        'tokens': s['tokens'],
                        'has_response': True
                    } for s in solutions
                ],
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
        "method": f"combined-cot-verification-{num_attempts}",
        "accuracy": accuracy,
        "num_correct": num_correct,
        "total_problems": max_problems,
        "wall_clock_time": wall_clock_time,
        "avg_tokens_per_problem": avg_tokens_per_problem,
        "total_tokens": total_tokens_generated,
        "num_attempts": num_attempts,
        "output_file": output_file
    }
    
    print(f"\nCombined CoT + Self-Verification Results:")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Correct: {num_correct}/{max_problems}")
    print(f"  Wall-clock time: {wall_clock_time:.2f}s")
    print(f"  Avg tokens per problem: {avg_tokens_per_problem:.1f}")
    print(f"  Total tokens: {total_tokens_generated}")
    print(f"  Results saved to: {output_file}")
    
    return metrics


def main():
    """Main function to run Task 3 combined method"""
    
    print("\n" + "="*60)
    print("TASK 3: COMBINED METHOD FOR GSM8K")
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
    
    # Run combined method with 3 attempts
    combined_metrics = run_combined_method(
        client, questions, answers,
        num_attempts=3,
        max_problems=max_test_problems,
        output_file="combined_cot_verification.jsonl"
    )
    
    print(f"\n{'='*60}")
    print("TASK 3 COMPLETED")
    print(f"{'='*60}")
    print(f"Combined method accuracy: {combined_metrics['accuracy']:.2f}%")
    

if __name__ == "__main__":
    main()
