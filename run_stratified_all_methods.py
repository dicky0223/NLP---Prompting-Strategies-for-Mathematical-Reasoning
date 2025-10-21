"""
Stratified Sampling - All Methods Evaluation
Runs all methods (Baseline Zero-shot, Few-shot, Self-Verification, Self-Refinement, Combined)
with stratified sampling of 200 problems from GSM8K

Uses the same evaluation structure as quick_test_results.jsonl for consistency

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

# Import method builders
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

# Configuration
STRATIFIED_SAMPLE_PATH = "stratified_samples/gsm8k_stratified_sample_200.json"
GSM8K_TEST_PATH = "data/GSM8K/test.jsonl"

# Global stats tracking
total_tokens_used = 0


def load_stratified_sample(sample_path: str) -> List[int]:
    """Load problem indices from stratified sample"""
    with open(sample_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract all problem IDs from stratification
    problem_ids = []
    for difficulty_level, difficulty_data in data['stratification'].items():
        problem_ids.extend(difficulty_data['problem_ids'])
    
    return sorted(problem_ids)


def load_gsm8k_dataset_subset(dataset_path: str, problem_ids: List[int]) -> Tuple[List[str], List[str]]:
    """
    Load specific problems from GSM8K dataset by their indices
    
    Args:
        dataset_path: Path to the JSONL file
        problem_ids: List of problem indices to load
        
    Returns:
        Tuple of (questions, answers) for specified problem IDs
    """
    questions = []
    answers = []
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx in problem_ids:
                data = json.loads(line)
                questions.append(data['question'])
                answer = data['answer'].split('#### ')[-1].strip()
                answers.append(answer)
    
    return questions, answers


def test_zero_shot(client: PoeAPIClient, question: str, ground_truth: str) -> Dict:
    """Test zero-shot method on a single question"""
    try:
        messages = build_zero_shot_messages(question)
        start_time = time.time()
        response_data = client.query_claude_sonnet(
            messages=messages,
            max_tokens=DEFAULT_CONFIG['max_tokens'],
        )
        time_elapsed = time.time() - start_time
        
        response = response_data['response']
        tokens_used = response_data['tokens_used']
        
        # Extract answer
        predicted_answer = extract_ans_from_response(response)
        is_correct = predicted_answer == ground_truth
        
        return {
            'method': 'zero-shot',
            'success': True,
            'predicted_answer': predicted_answer,
            'ground_truth': ground_truth,
            'is_correct': is_correct,
            'response': response[:500],  # Truncate for storage
            'tokens_used': tokens_used,
            'time_elapsed': time_elapsed
        }
    except Exception as e:
        return {
            'method': 'zero-shot',
            'success': False,
            'error': str(e),
            'ground_truth': ground_truth
        }


def test_few_shot(client: PoeAPIClient, question: str, ground_truth: str, num_shots: int = 5) -> Dict:
    """Test few-shot method on a single question"""
    try:
        messages = build_few_shot_messages(question, num_shots=num_shots)
        start_time = time.time()
        response_data = client.query_claude_sonnet(
            messages=messages,
            max_tokens=DEFAULT_CONFIG['max_tokens'],
        )
        time_elapsed = time.time() - start_time
        
        response = response_data['response']
        tokens_used = response_data['tokens_used']
        
        # Extract answer
        predicted_answer = extract_ans_from_response(response)
        is_correct = predicted_answer == ground_truth
        
        return {
            'method': f'few-shot-{num_shots}',
            'success': True,
            'predicted_answer': predicted_answer,
            'ground_truth': ground_truth,
            'is_correct': is_correct,
            'response': response[:500],
            'tokens_used': tokens_used,
            'time_elapsed': time_elapsed
        }
    except Exception as e:
        return {
            'method': f'few-shot-{num_shots}',
            'success': False,
            'error': str(e),
            'ground_truth': ground_truth
        }


def test_self_refine(client: PoeAPIClient, question: str, ground_truth: str, max_iterations: int = 3) -> Dict:
    """Test self-refinement method on a single question"""
    try:
        total_tokens = 0
        start_time = time.time()
        iterations_log = []
        
        # Generate initial solution
        messages = build_generate_messages(question)
        response_data = client.query_claude_sonnet(
            messages=messages,
            max_tokens=DEFAULT_CONFIG['max_tokens'],
        )
        response = response_data['response']
        total_tokens += response_data['tokens_used']
        
        predicted_answer = extract_ans_from_response(response)
        iterations_log.append({
            'iteration': 0,
            'type': 'generation',
            'answer': predicted_answer
        })
        
        # Refinement loop
        num_iterations = 0
        is_correct_per_feedback = False
        
        for iteration in range(max_iterations):
            # Get feedback
            feedback_messages = build_feedback_messages(question, response)
            feedback_data = client.query_claude_sonnet(
                messages=feedback_messages,
                max_tokens=DEFAULT_CONFIG['max_tokens'],
            )
            feedback = feedback_data['response']
            total_tokens += feedback_data['tokens_used']
            
            is_correct_per_feedback = is_correct_solution(feedback)
            iterations_log.append({
                'iteration': iteration + 1,
                'type': 'feedback',
                'is_correct': is_correct_per_feedback
            })
            
            num_iterations = iteration + 1
            
            if is_correct_per_feedback:
                break
            
            # Refine solution
            refinement_messages = build_refinement_messages(question, response, feedback)
            refinement_data = client.query_claude_sonnet(
                messages=refinement_messages,
                max_tokens=DEFAULT_CONFIG['max_tokens'],
            )
            response = refinement_data['response']
            total_tokens += refinement_data['tokens_used']
            
            predicted_answer = extract_ans_from_response(response)
        
        time_elapsed = time.time() - start_time
        is_correct = predicted_answer == ground_truth
        
        return {
            'method': f'self-refine-{max_iterations}',
            'success': True,
            'predicted_answer': predicted_answer,
            'ground_truth': ground_truth,
            'is_correct': is_correct,
            'is_correct_per_feedback': is_correct_per_feedback,
            'final_solution': response[:500],
            'tokens_used': total_tokens,
            'time_elapsed': time_elapsed,
            'num_iterations': num_iterations,
            'iterations_log': iterations_log
        }
    except Exception as e:
        return {
            'method': f'self-refine-{max_iterations}',
            'success': False,
            'error': str(e),
            'ground_truth': ground_truth
        }


def test_self_verification(client: PoeAPIClient, question: str, ground_truth: str, num_candidates: int = 3) -> Dict:
    """Test self-verification method on a single question"""
    try:
        total_tokens = 0
        start_time = time.time()
        candidates = []
        
        # Generate multiple candidates
        for i in range(num_candidates):
            messages = build_cot_messages(question)
            response_data = client.query_claude_sonnet(
                messages=messages,
                max_tokens=DEFAULT_CONFIG['max_tokens'],
            )
            response = response_data['response']
            total_tokens += response_data['tokens_used']
            
            predicted_answer = extract_ans_from_response(response)
            
            candidates.append({
                'candidate_id': i + 1,
                'answer': predicted_answer,
                'temperature': [0.3, 0.5, 0.7][i] if i < 3 else 0.7,
                'tokens_used': response_data['tokens_used']
            })
        
        # Verify each candidate
        verification_scores = []
        for candidate in candidates:
            messages = build_verifier_messages(question, candidate['answer'])
            response_data = client.query_claude_sonnet(
                messages=messages,
                max_tokens=DEFAULT_CONFIG['max_tokens'],
            )
            verification_response = response_data['response']
            total_tokens += response_data['tokens_used']
            
            # Check if verification passed
            verification_score = verify_answer(verification_response)
            candidate['verification_score'] = verification_score
            verification_scores.append(verification_score)
        
        # Select best candidate based on verification score
        best_idx = verification_scores.index(max(verification_scores))
        predicted_answer = candidates[best_idx]['answer']
        best_verification_score = candidates[best_idx]['verification_score']
        
        time_elapsed = time.time() - start_time
        is_correct = predicted_answer == ground_truth
        
        return {
            'method': f'self-verification-{num_candidates}',
            'success': True,
            'predicted_answer': predicted_answer,
            'ground_truth': ground_truth,
            'is_correct': is_correct,
            'best_verification_score': best_verification_score,
            'num_candidates': num_candidates,
            'candidates': candidates,
            'tokens_used': total_tokens,
            'time_elapsed': time_elapsed
        }
    except Exception as e:
        return {
            'method': f'self-verification-{num_candidates}',
            'success': False,
            'error': str(e),
            'ground_truth': ground_truth
        }


def test_combined_method(client: PoeAPIClient, question: str, ground_truth: str, max_refinement_iterations: int = 3) -> Dict:
    """Test combined method (verification + refinement) on a single question"""
    try:
        total_tokens = 0
        start_time = time.time()
        iterations_log = []
        num_cycles = 0
        
        # Step 1: Generate initial solution
        messages = combined_generate_initial_solution(question)
        response_data = client.query_claude_sonnet(
            messages=messages,
            max_tokens=DEFAULT_CONFIG['max_tokens'],
        )
        response = response_data['response']
        total_tokens += response_data['tokens_used']
        
        predicted_answer = extract_ans_from_response(response)
        iterations_log.append({
            'iteration': 0,
            'type': 'generation',
            'answer': predicted_answer
        })
        
        # Step 2: Verify solution
        messages = combined_verify_solution(question, response)
        verification_data = client.query_claude_sonnet(
            messages=messages,
            max_tokens=DEFAULT_CONFIG['max_tokens'],
        )
        verification_response = verification_data['response']
        total_tokens += verification_data['tokens_used']
        
        is_correct_verification = verify_answer(verification_response)
        
        iterations_log.append({
            'iteration': 1,
            'type': 'verification',
            'answer': predicted_answer,
            'is_correct': is_correct_verification
        })
        num_cycles = 1
        
        # Step 3: Refinement if needed
        if not is_correct_verification:
            # Get feedback for refinement
            feedback_messages = combined_generate_feedback(question, response, verification_response)
            feedback_data = client.query_claude_sonnet(
                messages=feedback_messages,
                max_tokens=DEFAULT_CONFIG['max_tokens'],
            )
            feedback = feedback_data['response']
            total_tokens += feedback_data['tokens_used']
            
            # Refinement loop
            for iteration in range(max_refinement_iterations):
                refinement_messages = combined_refine_solution(question, response, feedback)
                refinement_data = client.query_claude_sonnet(
                    messages=refinement_messages,
                    max_tokens=DEFAULT_CONFIG['max_tokens'],
                )
                response = refinement_data['response']
                total_tokens += refinement_data['tokens_used']
                
                predicted_answer = extract_ans_from_response(response)
                
                # Verify refined solution
                messages = combined_verify_solution(question, response)
                verification_data = client.query_claude_sonnet(
                    messages=messages,
                    max_tokens=DEFAULT_CONFIG['max_tokens'],
                )
                verification_response = verification_data['response']
                total_tokens += verification_data['tokens_used']
                
                is_correct_verification = verify_answer(verification_response)
                num_cycles += 1
                
                iterations_log.append({
                    'iteration': len(iterations_log),
                    'type': 'verification',
                    'answer': predicted_answer,
                    'is_correct': is_correct_verification
                })
                
                if is_correct_verification:
                    break
        
        time_elapsed = time.time() - start_time
        is_correct = predicted_answer == ground_truth
        
        return {
            'method': f'combined-{max_refinement_iterations}',
            'success': True,
            'predicted_answer': predicted_answer,
            'ground_truth': ground_truth,
            'is_correct': is_correct,
            'verification_passed_per_llm': is_correct_verification,
            'final_solution': response[:500],
            'tokens_used': total_tokens,
            'time_elapsed': time_elapsed,
            'num_cycles': num_cycles,
            'iterations_log': iterations_log
        }
    except Exception as e:
        return {
            'method': f'combined-{max_refinement_iterations}',
            'success': False,
            'error': str(e),
            'ground_truth': ground_truth
        }


def run_all_methods_stratified(client: PoeAPIClient, questions: List[str], answers: List[str], output_file: str):
    """
    Run all methods on stratified samples and save results
    
    Args:
        client: PoeAPIClient instance
        questions: List of questions
        answers: List of ground truth answers
        output_file: Output JSONL file path
    """
    results = []
    
    print(f"\n{'='*80}")
    print(f"Running all methods on {len(questions)} stratified samples")
    print(f"{'='*80}\n")
    
    # Track start time for overall stats
    overall_start_time = time.time()
    total_tokens = 0
    
    for question_idx, (question, ground_truth) in enumerate(tqdm(zip(questions, answers), total=len(questions))):
        print(f"\n{'='*60}")
        print(f"Problem {question_idx + 1}/{len(questions)}")
        print(f"{'='*60}")
        
        # 1. Zero-shot
        print("  Testing Zero-shot...")
        result = test_zero_shot(client, question, ground_truth)
        results.append(result)
        if result['success']:
            total_tokens += result['tokens_used']
            print(f"    ✓ Answer: {result['predicted_answer']}, Correct: {result['is_correct']}")
        
        # 2. Few-shot (5 shots)
        print("  Testing Few-shot (5 examples)...")
        result = test_few_shot(client, question, ground_truth, num_shots=5)
        results.append(result)
        if result['success']:
            total_tokens += result['tokens_used']
            print(f"    ✓ Answer: {result['predicted_answer']}, Correct: {result['is_correct']}")
        
        # 3. Self-Refinement (3 iterations)
        print("  Testing Self-Refinement (3 iterations)...")
        result = test_self_refine(client, question, ground_truth, max_iterations=3)
        results.append(result)
        if result['success']:
            total_tokens += result['tokens_used']
            print(f"    ✓ Answer: {result['predicted_answer']}, Correct: {result['is_correct']}")
        
        # 4. Self-Verification (3 candidates)
        print("  Testing Self-Verification (3 candidates)...")
        result = test_self_verification(client, question, ground_truth, num_candidates=3)
        results.append(result)
        if result['success']:
            total_tokens += result['tokens_used']
            print(f"    ✓ Answer: {result['predicted_answer']}, Correct: {result['is_correct']}")
        
        # 5. Combined Method (3 iterations)
        print("  Testing Combined Method (verification→refinement, 3 iterations)...")
        result = test_combined_method(client, question, ground_truth, max_refinement_iterations=3)
        results.append(result)
        if result['success']:
            total_tokens += result['tokens_used']
            print(f"    ✓ Answer: {result['predicted_answer']}, Correct: {result['is_correct']}")
        
        # Save results incrementally
        with open(output_file, 'a', encoding='utf-8') as f:
            for r in results[-5:]:  # Save last 5 results
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    overall_time = time.time() - overall_start_time
    
    return results, total_tokens, overall_time


def generate_summary_report(results: List[Dict], total_tokens: int, overall_time: float, num_problems: int):
    """Generate summary statistics"""
    
    print(f"\n{'='*80}")
    print("STRATIFIED SAMPLING EVALUATION SUMMARY")
    print(f"{'='*80}\n")
    
    # Group results by method
    methods = {}
    for result in results:
        method = result.get('method', 'unknown')
        if method not in methods:
            methods[method] = {'correct': 0, 'total': 0, 'tokens': 0, 'time': 0}
        
        methods[method]['total'] += 1
        if result.get('success', False):
            if result.get('is_correct', False):
                methods[method]['correct'] += 1
            methods[method]['tokens'] += result.get('tokens_used', 0)
            methods[method]['time'] += result.get('time_elapsed', 0)
    
    # Print method comparison
    print(f"{'Method':<30} {'Accuracy':<15} {'Avg Tokens':<20} {'Total Time (s)':<20}")
    print("-" * 85)
    
    for method in sorted(methods.keys()):
        stats = methods[method]
        accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        avg_tokens = stats['tokens'] / stats['total'] if stats['total'] > 0 else 0
        total_time = stats['time']
        
        print(f"{method:<30} {accuracy:>6.2f}%{'':<7} {avg_tokens:>10.1f}{'':<9} {total_time:>10.2f}")
    
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS")
    print(f"{'='*80}")
    print(f"Total problems evaluated: {num_problems}")
    print(f"Total tokens used (all methods): {total_tokens}")
    print(f"Overall wall-clock time: {overall_time:.2f}s")
    print(f"Average time per problem set: {overall_time/num_problems:.2f}s")


if __name__ == "__main__":
    # Load stratified sample
    print("Loading stratified sample...")
    problem_ids = load_stratified_sample(STRATIFIED_SAMPLE_PATH)
    print(f"Loaded {len(problem_ids)} problem indices from stratified sample")
    
    # Load GSM8K subset
    print("Loading GSM8K dataset subset...")
    questions, answers = load_gsm8k_dataset_subset(GSM8K_TEST_PATH, set(problem_ids))
    print(f"Loaded {len(questions)} questions and answers")
    
    # Initialize API client
    print("Initializing API client...")
    client = initialize_api_client()
    
    # Run all methods
    output_file = "stratified_sample_200_all_methods.jsonl"
    if os.path.exists(output_file):
        os.remove(output_file)
    
    results, total_tokens, overall_time = run_all_methods_stratified(
        client, questions, answers, output_file
    )
    
    # Generate summary
    generate_summary_report(results, total_tokens, overall_time, len(questions))
    
    print(f"\nResults saved to: {output_file}")
