"""
Task 3: Combined Method - Self-Verification + Self-Refinement
Enhanced Math Reasoning through Sequential Verification and Refinement

Strategy: Sequential Verification → Refinement
1. Generate initial solution (CoT)
2. Verify the solution through backward reasoning
3. If verification fails, provide feedback for refinement
4. Iteratively refine until correct or max iterations reached

The combination is synergistic because:
- Verification catches errors that the initial generation misses
- Refinement fixes the errors that verification detects
- Together they create a robust error-correction loop

Author: Assignment 3
Date: 2025
"""

import json
import time
import os
import re
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from api_client import PoeAPIClient
from config import initialize_api_client, DEFAULT_CONFIG
from data.GSM8K.evaluation import extract_ans_from_response


# System prompts for Chain-of-Thought generation
SYSTEM_PROMPT_GENERATE = """You are an expert math problem solver. Your task is to solve math word problems step by step.

When solving:
1. Read the problem carefully
2. Identify what is being asked
3. Show your reasoning process clearly
4. Provide the final numerical answer

Always end your response with: #### [answer]
where [answer] is the numerical result."""

# System prompt for verification
SYSTEM_PROMPT_VERIFIER = """You are an expert math problem verifier. Your task is to verify if a given answer is correct by checking if it satisfies the problem conditions.

When verifying:
1. Understand the problem and the proposed answer
2. Work backwards from the answer to check if it satisfies all conditions
3. Show your verification process clearly
4. Determine if the answer is correct or incorrect
5. Provide specific reasons for your verification result

End your response with either:
- "VERIFICATION_PASSED" if the answer is correct
- "VERIFICATION_FAILED" if the answer is incorrect, followed by explanation of the error"""

# System prompt for feedback generation
SYSTEM_PROMPT_FEEDBACK = """You are an expert math teacher providing constructive feedback on solutions.

When reviewing a solution:
1. Identify the specific errors in the mathematical reasoning
2. Explain why the approach or calculation is incorrect
3. Provide specific guidance on what needs to be fixed
4. Suggest the correct approach or calculation step

Be clear, concise, and actionable in your feedback."""

# System prompt for refinement
SYSTEM_PROMPT_REFINE = """You are an expert math problem solver tasked with improving a solution based on feedback.

When refining:
1. Carefully read the original problem
2. Review the previous solution and the specific feedback provided
3. Address each point of feedback directly
4. Show corrected reasoning process clearly
5. Provide the corrected final numerical answer

Always end your response with: #### [answer]
where [answer] is the numerical result."""


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


def build_generate_messages(question: str) -> List[Dict[str, str]]:
    """Build message list for generating initial solution via CoT"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT_GENERATE},
        {"role": "user", "content": f"Question: {question}\n\nPlease solve this step by step."}
    ]


def build_verify_messages(question: str, solution: str) -> List[Dict[str, str]]:
    """Build message list for verifying a solution"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT_VERIFIER},
        {
            "role": "user",
            "content": f"""Question: {question}

Proposed Solution:
{solution}

Please verify if this solution is correct by checking if the answer satisfies the problem conditions."""
        }
    ]


def build_feedback_messages(question: str, solution: str, verification_result: str) -> List[Dict[str, str]]:
    """Build message list for generating feedback on why verification failed"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT_FEEDBACK},
        {
            "role": "user",
            "content": f"""Question: {question}

Solution Provided:
{solution}

Verification Result:
{verification_result}

Based on the verification failure, please identify the specific errors in the solution and explain what went wrong."""
        }
    ]


def build_refinement_messages(question: str, solution: str, feedback: str) -> List[Dict[str, str]]:
    """Build message list for refining solution based on feedback"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT_REFINE},
        {
            "role": "user",
            "content": f"""Question: {question}

Previous Solution:
{solution}

Feedback on the previous solution:
{feedback}

Please provide an improved solution that addresses this feedback. Show your corrected reasoning process clearly and end with: #### [answer]"""
        }
    ]


def generate_initial_solution(client: PoeAPIClient, question: str) -> Tuple[str, int]:
    """
    Generate initial solution for a question using CoT
    
    Args:
        client: PoeAPIClient instance
        question: The math problem question
        
    Returns:
        Tuple of (solution_text, tokens_used)
    """
    messages = build_generate_messages(question)
    
    response_data = client.query_claude_sonnet(
        messages=messages,
        max_tokens=DEFAULT_CONFIG['max_tokens'],
        temperature=DEFAULT_CONFIG['temperature']
    )
    
    solution_text = response_data['content']
    tokens_used = response_data['usage']['completion_tokens']
    
    return solution_text, tokens_used


def verify_solution(client: PoeAPIClient, question: str, solution: str) -> Tuple[bool, str, int]:
    """
    Verify if a solution is correct
    
    Args:
        client: PoeAPIClient instance
        question: The math problem question
        solution: The proposed solution
        
    Returns:
        Tuple of (is_correct, verification_text, tokens_used)
    """
    messages = build_verify_messages(question, solution)
    
    response_data = client.query_claude_sonnet(
        messages=messages,
        max_tokens=1000,
        temperature=0.0  # Use temperature 0 for consistent verification
    )
    
    verification_text = response_data['content'].strip()
    tokens_used = response_data['usage']['completion_tokens']
    
    # Check if verification passed
    is_correct = "VERIFICATION_PASSED" in verification_text.upper()
    
    return is_correct, verification_text, tokens_used


def generate_feedback(client: PoeAPIClient, question: str, solution: str, 
                     verification_result: str) -> Tuple[str, int]:
    """
    Generate feedback on why verification failed
    
    Args:
        client: PoeAPIClient instance
        question: The math problem question
        solution: The proposed solution
        verification_result: The verification result explanation
        
    Returns:
        Tuple of (feedback_text, tokens_used)
    """
    messages = build_feedback_messages(question, solution, verification_result)
    
    response_data = client.query_claude_sonnet(
        messages=messages,
        max_tokens=800,
        temperature=0.0
    )
    
    feedback_text = response_data['content'].strip()
    tokens_used = response_data['usage']['completion_tokens']
    
    return feedback_text, tokens_used


def refine_solution(client: PoeAPIClient, question: str, solution: str, 
                   feedback: str) -> Tuple[str, int]:
    """
    Refine solution based on feedback
    
    Args:
        client: PoeAPIClient instance
        question: The math problem question
        solution: The previous solution
        feedback: Feedback on the previous solution
        
    Returns:
        Tuple of (refined_solution_text, tokens_used)
    """
    messages = build_refinement_messages(question, solution, feedback)
    
    response_data = client.query_claude_sonnet(
        messages=messages,
        max_tokens=DEFAULT_CONFIG['max_tokens'],
        temperature=DEFAULT_CONFIG['temperature']
    )
    
    refined_solution = response_data['content']
    tokens_used = response_data['usage']['completion_tokens']
    
    return refined_solution, tokens_used


def run_combined_method(client: PoeAPIClient, question: str, ground_truth: str, 
                       max_refinement_iterations: int = 3) -> Dict:
    """
    Run combined verification + refinement method on a single question
    
    Strategy: Verify → Refine loop
    1. Generate initial solution
    2. Verify the solution
    3. If verification fails, generate feedback and refine
    4. Repeat verification until correct or max iterations reached
    
    Args:
        client: PoeAPIClient instance
        question: The math problem question
        ground_truth: Ground truth answer
        max_refinement_iterations: Maximum refinement iterations
        
    Returns:
        Dictionary with combined method history and metrics
    """
    
    process_log = []
    total_tokens = 0
    
    # Step 1: Generate initial solution
    try:
        solution, tokens = generate_initial_solution(client, question)
        total_tokens += tokens
        predicted_answer = extract_ans_from_response(solution)
        
        process_log.append({
            "stage": "generation",
            "iteration": 0,
            "action": "initial_generation",
            "solution": solution,
            "predicted_answer": str(predicted_answer),
            "verification_result": None,
            "feedback": None,
            "tokens_used": tokens
        })
    except Exception as e:
        return {
            "question": question,
            "ground_truth": ground_truth,
            "success": False,
            "error": f"Initial generation failed: {str(e)}",
            "process_log": process_log,
            "total_tokens": total_tokens,
            "final_answer": None,
            "is_correct": False,
            "num_verification_cycles": 0
        }
    
    # Step 2: Verification + Refinement loop
    current_solution = solution
    verification_passed = False
    num_refinement_iterations = 0
    
    for refinement_iteration in range(max_refinement_iterations):
        try:
            # Verify current solution
            is_correct, verification_text, tokens = verify_solution(client, question, current_solution)
            total_tokens += tokens
            
            predicted_answer = extract_ans_from_response(current_solution)
            
            process_log.append({
                "stage": "verification",
                "iteration": refinement_iteration + 1,
                "action": "verification",
                "solution": current_solution,
                "predicted_answer": str(predicted_answer),
                "verification_result": verification_text,
                "feedback": None,
                "is_correct": is_correct,
                "tokens_used": tokens
            })
            
            if is_correct:
                verification_passed = True
                break
            
            # Verification failed - generate feedback
            feedback, tokens = generate_feedback(client, question, current_solution, verification_text)
            total_tokens += tokens
            
            process_log.append({
                "stage": "feedback_generation",
                "iteration": refinement_iteration + 1,
                "action": "feedback_generation",
                "solution": current_solution,
                "predicted_answer": str(predicted_answer),
                "verification_result": verification_text[:200] + "..." if len(verification_text) > 200 else verification_text,
                "feedback": feedback,
                "tokens_used": tokens
            })
            
            # Refine solution based on feedback
            refined_solution, tokens = refine_solution(client, question, current_solution, feedback)
            total_tokens += tokens
            
            predicted_answer = extract_ans_from_response(refined_solution)
            
            process_log.append({
                "stage": "refinement",
                "iteration": refinement_iteration + 1,
                "action": "refinement",
                "solution": refined_solution,
                "predicted_answer": str(predicted_answer),
                "verification_result": None,
                "feedback": feedback[:150] + "..." if len(feedback) > 150 else feedback,
                "tokens_used": tokens
            })
            
            current_solution = refined_solution
            num_refinement_iterations += 1
            
        except Exception as e:
            print(f"Error during verification-refinement cycle {refinement_iteration}: {e}")
            break
    
    # Extract final answer
    final_answer = extract_ans_from_response(current_solution)
    
    # Check correctness against ground truth
    try:
        ground_truth_num = int(float(ground_truth))
        predicted_num = int(float(str(final_answer)))
        is_correct = ground_truth_num == predicted_num
    except (ValueError, TypeError):
        is_correct = str(final_answer).strip() == str(ground_truth).strip()
    
    return {
        "question": question,
        "ground_truth": ground_truth,
        "success": True,
        "final_solution": current_solution,
        "final_answer": str(final_answer),
        "is_correct": is_correct,
        "verification_passed_per_llm": verification_passed,
        "process_log": process_log,
        "total_tokens": total_tokens,
        "num_verification_cycles": num_refinement_iterations + 1  # +1 for initial verification
    }


def run_combined_baseline(client: PoeAPIClient, questions: List[str], answers: List[str],
                          max_refinement_iterations: int = 3, max_problems: int = None,
                          output_file: str = "combined_verify_refine.jsonl") -> Dict:
    """
    Run combined verification + refinement baseline on GSM8K
    
    Args:
        client: PoeAPIClient instance
        questions: List of questions
        answers: List of ground truth answers
        max_refinement_iterations: Maximum refinement iterations per question
        max_problems: Maximum number of problems to evaluate (None for all)
        output_file: Output JSONL file path
        
    Returns:
        Dictionary with accuracy and timing metrics
    """
    print(f"\n{'='*60}")
    print(f"TASK 3: COMBINED METHOD (Verify → Refine)")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Strategy: Sequential Verification → Refinement")
    print(f"  Max refinement iterations: {max_refinement_iterations}")
    print(f"  Verification temperature: 0.0 (deterministic)")
    print(f"  Generation/Refinement temperature: {DEFAULT_CONFIG['temperature']}")
    
    max_problems = max_problems or len(questions)
    num_correct = 0
    num_verification_passed = 0
    total_tokens_generated = 0
    total_verification_cycles = 0
    start_time = time.time()
    
    results = []
    
    with tqdm(total=max_problems, desc="Combined Verify→Refine") as pbar:
        for idx in range(min(max_problems, len(questions))):
            question = questions[idx]
            ground_truth = answers[idx]
            
            try:
                result = run_combined_method(
                    client, question, ground_truth,
                    max_refinement_iterations=max_refinement_iterations
                )
                
                total_tokens_generated += result['total_tokens']
                total_verification_cycles += result['num_verification_cycles']
                
                if result['is_correct']:
                    num_correct += 1
                
                if result['verification_passed_per_llm']:
                    num_verification_passed += 1
                
                results.append({
                    "question_id": idx,
                    "question": question,
                    "predicted_answer": result['final_answer'],
                    "ground_truth": ground_truth,
                    "is_correct": result['is_correct'],
                    "verification_passed_per_llm": result['verification_passed_per_llm'],
                    "final_solution": result['final_solution'],
                    "process_log": result['process_log'],
                    "total_tokens": result['total_tokens'],
                    "num_verification_cycles": result['num_verification_cycles']
                })
                
            except Exception as e:
                print(f"Error processing question {idx}: {e}")
                results.append({
                    "question_id": idx,
                    "question": question,
                    "predicted_answer": None,
                    "ground_truth": ground_truth,
                    "is_correct": False,
                    "error": str(e)
                })
            
            pbar.update(1)
    
    end_time = time.time()
    wall_clock_time = end_time - start_time
    
    # Calculate metrics
    accuracy = (num_correct / max_problems) * 100 if max_problems > 0 else 0
    verification_passed_rate = (num_verification_passed / max_problems) * 100 if max_problems > 0 else 0
    avg_tokens_per_problem = total_tokens_generated / max_problems if max_problems > 0 else 0
    avg_verification_cycles = total_verification_cycles / max_problems if max_problems > 0 else 0
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    metrics = {
        "method": f"combined-verify-refine-{max_refinement_iterations}",
        "accuracy": accuracy,
        "num_correct": num_correct,
        "verification_passed_rate": verification_passed_rate,
        "num_verification_passed": num_verification_passed,
        "total_problems": max_problems,
        "wall_clock_time": wall_clock_time,
        "avg_tokens_per_problem": avg_tokens_per_problem,
        "avg_verification_cycles": avg_verification_cycles,
        "total_tokens": total_tokens_generated,
        "output_file": output_file
    }
    
    print(f"\nCombined Method Results (Verify → Refine):")
    print(f"  Final Accuracy: {accuracy:.2f}%")
    print(f"  Correct: {num_correct}/{max_problems}")
    print(f"  Verification Passed Rate: {verification_passed_rate:.2f}%")
    print(f"  Verification Passed: {num_verification_passed}/{max_problems}")
    print(f"  Wall-clock time: {wall_clock_time:.2f}s")
    print(f"  Avg tokens per problem: {avg_tokens_per_problem:.1f}")
    print(f"  Avg verification cycles: {avg_verification_cycles:.2f}")
    print(f"  Total tokens: {total_tokens_generated}")
    print(f"  Results saved to: {output_file}")
    
    return metrics


def main():
    """Main function to run Task 3 combined method baseline"""
    
    print("\n" + "="*60)
    print("TASK 3: COMBINED METHOD (Verify → Refine) FOR GSM8K")
    print("="*60)
    
    # Initialize API client
    print("\nInitializing API client...")
    client = initialize_api_client()
    
    # Load dataset
    dataset_path = "data/GSM8K/test.jsonl"
    print(f"Loading dataset from {dataset_path}...")
    questions, answers = load_gsm8k_dataset(dataset_path)
    print(f"Loaded {len(questions)} questions")
    
    # Run combined method with different iteration limits
    max_test_problems = DEFAULT_CONFIG['max_problems']
    
    # Test with 2 refinement iterations
    combined_2_metrics = run_combined_baseline(
        client, questions, answers,
        max_refinement_iterations=2,
        max_problems=max_test_problems,
        output_file="combined_verify_refine_2iter.jsonl"
    )
    
    # Test with 3 refinement iterations
    combined_3_metrics = run_combined_baseline(
        client, questions, answers,
        max_refinement_iterations=3,
        max_problems=max_test_problems,
        output_file="combined_verify_refine_3iter.jsonl"
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("TASK 3 COMBINED METHOD SUMMARY")
    print(f"{'='*60}")
    print(f"\nWith 2 refinement iterations:")
    print(f"  Final Accuracy: {combined_2_metrics['accuracy']:.2f}%")
    print(f"  Verification Passed Rate: {combined_2_metrics['verification_passed_rate']:.2f}%")
    print(f"  Avg Verification Cycles: {combined_2_metrics['avg_verification_cycles']:.2f}")
    print(f"\nWith 3 refinement iterations:")
    print(f"  Final Accuracy: {combined_3_metrics['accuracy']:.2f}%")
    print(f"  Verification Passed Rate: {combined_3_metrics['verification_passed_rate']:.2f}%")
    print(f"  Avg Verification Cycles: {combined_3_metrics['avg_verification_cycles']:.2f}")
    

if __name__ == "__main__":
    main()
