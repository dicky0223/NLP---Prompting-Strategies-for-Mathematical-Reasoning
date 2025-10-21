"""
Quick Test All Methods - Random sampling of 5 problems for rapid testing
Tests all implemented methods: Zero-shot, Few-shot, Self-Refine, Self-Verification, and Combined

Reproducible random sampling with seed=42 for consistent testing

Author: Assignment 1
Date: 2025
"""

import json
import time
import random
from typing import List, Dict, Tuple
from api_client import PoeAPIClient
from config import initialize_api_client, DEFAULT_CONFIG
from data.GSM8K.evaluation import extract_ans_from_response


# Import method builders from task files
from task1_baseline import (
    build_zero_shot_messages, 
    build_few_shot_messages,
    FEW_SHOT_EXAMPLES
)
from task2_self_refine import (
    build_generate_messages,
    build_feedback_messages,
    build_refinement_messages,
    get_initial_solution,
    get_feedback,
    refine_solution,
    is_correct_solution
)
from task2_self_verification import (
    build_cot_messages,
    build_verifier_messages,
    verify_answer
)
from task3_combined_method import (
    generate_initial_solution as combined_generate_initial_solution,
    verify_solution as combined_verify_solution,
    generate_feedback as combined_generate_feedback,
    refine_solution as combined_refine_solution
)

# Set seed globally for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def load_gsm8k_dataset(dataset_path: str, num_samples: int = 5) -> Tuple[List[str], List[str], List[int]]:
    """
    Load and randomly sample problems from GSM8K dataset
    
    Args:
        dataset_path: Path to the JSONL file
        num_samples: Number of random samples to load
        
    Returns:
        Tuple of (questions, answers, sample_indices)
    """
    questions = []
    answers = []
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            questions.append(data['question'])
            answer = data['answer'].split('#### ')[-1].strip()
            answers.append(answer)
    
    # Random sampling
    sample_indices = random.sample(range(len(questions)), min(num_samples, len(questions)))
    sample_indices.sort()
    
    sampled_questions = [questions[i] for i in sample_indices]
    sampled_answers = [answers[i] for i in sample_indices]
    
    return sampled_questions, sampled_answers, sample_indices


def test_zero_shot(client: PoeAPIClient, question: str, ground_truth: str) -> Dict:
    """
    Test zero-shot method on a single question
    
    Args:
        client: PoeAPIClient instance
        question: The math problem question
        ground_truth: Ground truth answer
        
    Returns:
        Dictionary with test results
    """
    try:
        messages = build_zero_shot_messages(question)
        start_time = time.time()
        response_data = client.query_claude_sonnet(
            messages=messages,
            max_tokens=DEFAULT_CONFIG['max_tokens'],
            temperature=DEFAULT_CONFIG['temperature']
        )
        elapsed = time.time() - start_time
        
        response_text = response_data['content']
        tokens_used = response_data['usage']['completion_tokens']
        predicted_answer = extract_ans_from_response(response_text)
        
        # Check correctness
        try:
            ground_truth_num = int(float(ground_truth))
            predicted_num = int(float(str(predicted_answer)))
            is_correct = ground_truth_num == predicted_num
        except (ValueError, TypeError):
            is_correct = str(predicted_answer).strip() == str(ground_truth).strip()
        
        return {
            "method": "zero-shot",
            "success": True,
            "predicted_answer": str(predicted_answer),
            "ground_truth": ground_truth,
            "is_correct": is_correct,
            "response": response_text,
            "tokens_used": tokens_used,
            "time_elapsed": elapsed
        }
    except Exception as e:
        return {
            "method": "zero-shot",
            "success": False,
            "error": str(e),
            "predicted_answer": None,
            "is_correct": False
        }


def test_few_shot(client: PoeAPIClient, question: str, ground_truth: str, num_examples: int = 5) -> Dict:
    """
    Test few-shot method on a single question
    
    Args:
        client: PoeAPIClient instance
        question: The math problem question
        ground_truth: Ground truth answer
        num_examples: Number of demonstration examples
        
    Returns:
        Dictionary with test results
    """
    try:
        messages = build_few_shot_messages(question, num_examples=num_examples)
        start_time = time.time()
        response_data = client.query_claude_sonnet(
            messages=messages,
            max_tokens=DEFAULT_CONFIG['max_tokens'],
            temperature=DEFAULT_CONFIG['temperature']
        )
        elapsed = time.time() - start_time
        
        response_text = response_data['content']
        tokens_used = response_data['usage']['completion_tokens']
        predicted_answer = extract_ans_from_response(response_text)
        
        # Check correctness
        try:
            ground_truth_num = int(float(ground_truth))
            predicted_num = int(float(str(predicted_answer)))
            is_correct = ground_truth_num == predicted_num
        except (ValueError, TypeError):
            is_correct = str(predicted_answer).strip() == str(ground_truth).strip()
        
        return {
            "method": f"few-shot-{num_examples}",
            "success": True,
            "predicted_answer": str(predicted_answer),
            "ground_truth": ground_truth,
            "is_correct": is_correct,
            "response": response_text,
            "tokens_used": tokens_used,
            "time_elapsed": elapsed
        }
    except Exception as e:
        return {
            "method": f"few-shot-{num_examples}",
            "success": False,
            "error": str(e),
            "predicted_answer": None,
            "is_correct": False
        }


def test_self_refine(client: PoeAPIClient, question: str, ground_truth: str, max_iterations: int = 3) -> Dict:
    """
    Test self-refine method on a single question
    
    Args:
        client: PoeAPIClient instance
        question: The math problem question
        ground_truth: Ground truth answer
        max_iterations: Maximum refinement iterations
        
    Returns:
        Dictionary with test results
    """
    try:
        total_tokens = 0
        iterations_log = []
        start_time = time.time()
        
        # Step 1: Generate initial solution
        solution, tokens = get_initial_solution(client, question)
        total_tokens += tokens
        predicted_answer = extract_ans_from_response(solution)
        
        iterations_log.append({
            "iteration": 0,
            "type": "generation",
            "answer": str(predicted_answer)
        })
        
        # Step 2: Iterative refinement loop
        current_solution = solution
        is_final_correct = False
        
        for iteration in range(1, max_iterations + 1):
            # Get feedback
            feedback, tokens = get_feedback(client, question, current_solution)
            total_tokens += tokens
            
            # Check if solution is correct
            if is_correct_solution(feedback):
                is_final_correct = True
                iterations_log.append({
                    "iteration": iteration,
                    "type": "feedback",
                    "is_correct": True
                })
                break
            
            iterations_log.append({
                "iteration": iteration,
                "type": "feedback",
                "is_correct": False
            })
            
            # Refine solution
            refined_solution, tokens = refine_solution(client, question, current_solution, feedback)
            total_tokens += tokens
            
            predicted_answer = extract_ans_from_response(refined_solution)
            iterations_log.append({
                "iteration": iteration,
                "type": "refinement",
                "answer": str(predicted_answer)
            })
            
            current_solution = refined_solution
        
        elapsed = time.time() - start_time
        
        # Extract final answer
        final_answer = extract_ans_from_response(current_solution)
        
        # Check correctness
        try:
            ground_truth_num = int(float(ground_truth))
            predicted_num = int(float(str(final_answer)))
            is_correct = ground_truth_num == predicted_num
        except (ValueError, TypeError):
            is_correct = str(final_answer).strip() == str(ground_truth).strip()
        
        return {
            "method": f"self-refine-{max_iterations}",
            "success": True,
            "predicted_answer": str(final_answer),
            "ground_truth": ground_truth,
            "is_correct": is_correct,
            "is_correct_per_feedback": is_final_correct,
            "final_solution": current_solution,
            "tokens_used": total_tokens,
            "time_elapsed": elapsed,
            "num_iterations": len([log for log in iterations_log if log["type"] == "refinement"]) + 1,
            "iterations_log": iterations_log
        }
    except Exception as e:
        return {
            "method": f"self-refine-{max_iterations}",
            "success": False,
            "error": str(e),
            "predicted_answer": None,
            "is_correct": False
        }


def test_self_verification(client: PoeAPIClient, question: str, ground_truth: str, num_candidates: int = 3) -> Dict:
    """
    Test self-verification method on a single question
    
    Args:
        client: PoeAPIClient instance
        question: The math problem question
        ground_truth: Ground truth answer
        num_candidates: Number of candidate solutions to generate
        
    Returns:
        Dictionary with test results
    """
    try:
        total_tokens = 0
        start_time = time.time()
        
        candidates = []
        
        # Generate multiple candidates with different temperatures
        for i, temp in enumerate([0.3, 0.5, 0.7]):
            try:
                messages = build_cot_messages(question, num_examples=5)
                response_data = client.query_claude_sonnet(
                    messages=messages,
                    max_tokens=DEFAULT_CONFIG['max_tokens'],
                    temperature=temp
                )
                
                response_text = response_data['content']
                tokens_used = response_data['usage']['completion_tokens']
                total_tokens += tokens_used
                
                predicted_answer = extract_ans_from_response(response_text)
                
                # Verify this candidate
                verification_score, verification_text = verify_answer(question, str(predicted_answer), client)
                
                candidates.append({
                    "candidate_id": i + 1,
                    "answer": str(predicted_answer),
                    "temperature": temp,
                    "verification_score": verification_score,
                    "tokens_used": tokens_used
                })
            except Exception as e:
                pass
        
        elapsed = time.time() - start_time
        
        if not candidates:
            return {
                "method": f"self-verification-{num_candidates}",
                "success": False,
                "error": "No candidates generated",
                "predicted_answer": None,
                "is_correct": False
            }
        
        # Select best candidate based on verification score
        best_candidate = max(candidates, key=lambda x: x['verification_score'])
        final_answer = best_candidate['answer']
        
        # Check correctness
        try:
            ground_truth_num = int(float(ground_truth))
            predicted_num = int(float(str(final_answer)))
            is_correct = ground_truth_num == predicted_num
        except (ValueError, TypeError):
            is_correct = str(final_answer).strip() == str(ground_truth).strip()
        
        return {
            "method": f"self-verification-{num_candidates}",
            "success": True,
            "predicted_answer": final_answer,
            "ground_truth": ground_truth,
            "is_correct": is_correct,
            "best_verification_score": best_candidate['verification_score'],
            "num_candidates": len(candidates),
            "candidates": candidates,
            "tokens_used": total_tokens,
            "time_elapsed": elapsed
        }
    except Exception as e:
        return {
            "method": f"self-verification-{num_candidates}",
            "success": False,
            "error": str(e),
            "predicted_answer": None,
            "is_correct": False
        }


def test_combined_method(client: PoeAPIClient, question: str, ground_truth: str, max_iterations: int = 3) -> Dict:
    """
    Test combined verification + refinement method on a single question
    
    Args:
        client: PoeAPIClient instance
        question: The math problem question
        ground_truth: Ground truth answer
        max_iterations: Maximum refinement iterations
        
    Returns:
        Dictionary with test results
    """
    try:
        total_tokens = 0
        iterations_log = []
        start_time = time.time()
        
        # Step 1: Generate initial solution
        solution, tokens = combined_generate_initial_solution(client, question)
        total_tokens += tokens
        predicted_answer = extract_ans_from_response(solution)
        
        iterations_log.append({
            "iteration": 0,
            "type": "generation",
            "answer": str(predicted_answer)
        })
        
        current_solution = solution
        verification_passed = False
        
        # Step 2: Verification + Refinement loop
        for iteration in range(1, max_iterations + 1):
            # Verify current solution
            is_correct, verification_text, tokens = combined_verify_solution(client, question, current_solution)
            total_tokens += tokens
            
            predicted_answer = extract_ans_from_response(current_solution)
            
            iterations_log.append({
                "iteration": iteration,
                "type": "verification",
                "answer": str(predicted_answer),
                "is_correct": is_correct
            })
            
            if is_correct:
                verification_passed = True
                break
            
            # Generate feedback on verification failure
            feedback, tokens = combined_generate_feedback(client, question, current_solution, verification_text)
            total_tokens += tokens
            
            iterations_log.append({
                "iteration": iteration,
                "type": "feedback"
            })
            
            # Refine solution
            refined_solution, tokens = combined_refine_solution(client, question, current_solution, feedback)
            total_tokens += tokens
            
            predicted_answer = extract_ans_from_response(refined_solution)
            iterations_log.append({
                "iteration": iteration,
                "type": "refinement",
                "answer": str(predicted_answer)
            })
            
            current_solution = refined_solution
        
        elapsed = time.time() - start_time
        
        # Extract final answer
        final_answer = extract_ans_from_response(current_solution)
        
        # Check correctness
        try:
            ground_truth_num = int(float(ground_truth))
            predicted_num = int(float(str(final_answer)))
            is_correct = ground_truth_num == predicted_num
        except (ValueError, TypeError):
            is_correct = str(final_answer).strip() == str(ground_truth).strip()
        
        return {
            "method": f"combined-{max_iterations}",
            "success": True,
            "predicted_answer": final_answer,
            "ground_truth": ground_truth,
            "is_correct": is_correct,
            "verification_passed_per_llm": verification_passed,
            "final_solution": current_solution,
            "tokens_used": total_tokens,
            "time_elapsed": elapsed,
            "num_cycles": len([log for log in iterations_log if log["type"] == "verification"]),
            "iterations_log": iterations_log
        }
    except Exception as e:
        return {
            "method": f"combined-{max_iterations}",
            "success": False,
            "error": str(e),
            "predicted_answer": None,
            "is_correct": False
        }


def print_problem_header(idx: int, sample_idx: int, question: str):
    """Print a formatted problem header"""
    print(f"\n{'='*80}")
    print(f"PROBLEM {idx + 1} (Dataset Index: {sample_idx})")
    print(f"{'='*80}")
    print(f"Question:")
    print(f"{question}\n")


def print_test_result(result: Dict):
    """Print a formatted test result"""
    method = result['method']
    
    if not result['success']:
        print(f"  ✗ {method:30} ERROR: {result.get('error', 'Unknown error')[:40]}")
        return
    
    status = "✓" if result['is_correct'] else "✗"
    
    if method.startswith('self-refine'):
        print(f"  {status} {method:28} | Answer: {result['predicted_answer']:10} | "
              f"Correct: {result['is_correct']!s:5} | Iter: {result['num_iterations']:2} | "
              f"Tokens: {result['tokens_used']:4} | Time: {result['time_elapsed']:.2f}s")
    elif method.startswith('self-verification'):
        print(f"  {status} {method:28} | Answer: {result['predicted_answer']:10} | "
              f"Correct: {result['is_correct']!s:5} | Score: {result['best_verification_score']:.2f} | "
              f"Tokens: {result['tokens_used']:4} | Time: {result['time_elapsed']:.2f}s")
    elif method.startswith('combined'):
        print(f"  {status} {method:28} | Answer: {result['predicted_answer']:10} | "
              f"Correct: {result['is_correct']!s:5} | Cycles: {result['num_cycles']:2} | "
              f"Tokens: {result['tokens_used']:4} | Time: {result['time_elapsed']:.2f}s")
    else:  # zero-shot, few-shot
        print(f"  {status} {method:28} | Answer: {result['predicted_answer']:10} | "
              f"Correct: {result['is_correct']!s:5} | Tokens: {result['tokens_used']:4} | "
              f"Time: {result['time_elapsed']:.2f}s")


def print_summary(all_results: List[Dict]):
    """Print a summary of all test results"""
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    # Group results by method
    methods = {}
    for result in all_results:
        if result['success']:
            method = result['method']
            if method not in methods:
                methods[method] = {'correct': 0, 'total': 0, 'tokens': 0, 'time': 0}
            methods[method]['total'] += 1
            if result['is_correct']:
                methods[method]['correct'] += 1
            methods[method]['tokens'] += result.get('tokens_used', 0)
            methods[method]['time'] += result.get('time_elapsed', 0)
    
    print(f"\n{'Method':<30} {'Accuracy':<12} {'Tokens':<10} {'Time (s)':<10}")
    print("-" * 65)
    
    for method in sorted(methods.keys()):
        stats = methods[method]
        accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        avg_tokens = stats['tokens'] / stats['total'] if stats['total'] > 0 else 0
        avg_time = stats['time'] / stats['total'] if stats['total'] > 0 else 0
        
        print(f"{method:<30} {accuracy:>6.1f}%       {avg_tokens:>7.1f}   {avg_time:>8.2f}")


def main():
    """Main function to run quick tests"""
    
    print("\n" + "="*100)
    print("QUICK TEST - ALL METHODS")
    print("="*100)
    print("Testing: Zero-shot, Few-shot (5-shot), Self-Refine (3 iter),")
    print("         Self-Verification (3 candidates), Combined Verify→Refine (3 iter)")
    print(f"Reproducible Random Seed: {RANDOM_SEED}")
    print()
    
    # Initialize API client
    print("Initializing API client...")
    client = initialize_api_client()
    
    # Load random sample
    dataset_path = "data/GSM8K/test.jsonl"
    print(f"Loading random sample from {dataset_path}...")
    questions, answers, sample_indices = load_gsm8k_dataset(dataset_path, num_samples=5)
    print(f"Loaded {len(questions)} random problems (indices: {sample_indices})\n")
    
    # Store all results
    all_results = []
    
    # Test each problem with all methods
    for idx, (question, ground_truth, sample_idx) in enumerate(zip(questions, answers, sample_indices)):
        print_problem_header(idx, sample_idx, question)
        print(f"Ground Truth: {ground_truth}\n")
        
        # Test zero-shot
        result = test_zero_shot(client, question, ground_truth)
        all_results.append(result)
        print_test_result(result)
        
        # Test few-shot
        result = test_few_shot(client, question, ground_truth, num_examples=5)
        all_results.append(result)
        print_test_result(result)
        
        # Test self-refine
        result = test_self_refine(client, question, ground_truth, max_iterations=3)
        all_results.append(result)
        print_test_result(result)
        
        # Test self-verification
        result = test_self_verification(client, question, ground_truth, num_candidates=3)
        all_results.append(result)
        print_test_result(result)
        
        # Test combined method
        result = test_combined_method(client, question, ground_truth, max_iterations=3)
        all_results.append(result)
        print_test_result(result)
    
    # Print summary
    print_summary(all_results)
    
    # Save detailed results
    output_file = "quick_test_results.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in all_results:
            # Remove long response text for cleaner output
            result_copy = result.copy()
            if 'response' in result_copy:
                result_copy['response'] = result_copy['response'][:200] + "..." if len(result_copy['response']) > 200 else result_copy['response']
            if 'final_solution' in result_copy:
                result_copy['final_solution'] = result_copy['final_solution'][:200] + "..." if len(result_copy['final_solution']) > 200 else result_copy['final_solution']
            f.write(json.dumps(result_copy) + '\n')
    
    print(f"\n✓ Detailed results saved to: {output_file}")
    print("\n" + "="*100)


if __name__ == "__main__":
    main()
