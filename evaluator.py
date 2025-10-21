"""
Unified Evaluator for All Prompting Methods
===========================================

A comprehensive evaluation framework that can run:
- Baseline methods (Zero-shot, Few-shot)
- Self-Refinement method
- Self-Verification method
- Combined method (Verification + Refinement)

On either:
- Quick test (5 random samples)
- Full GSM8K dataset
- Stratified sample (200 problems)
- Custom subset

Usage:
    # Run all methods on quick test
    python evaluator.py --mode all --dataset quick-test
    
    # Run specific method on stratified sample
    python evaluator.py --mode combined --dataset stratified
    
    # Run few-shot only on full dataset
    python evaluator.py --mode few-shot --dataset full
    
    # Run specific method with custom output
    python evaluator.py --mode zero-shot --dataset full --output my_results.jsonl

Methods:
    all              - Run all methods
    zero-shot        - Zero-shot baseline
    few-shot         - Few-shot baseline (5 examples)
    self-refine      - Self-refinement with iterative feedback
    self-verify      - Self-verification with multiple candidates
    combined         - Combined method (verify then refine)

Datasets:
    quick-test       - 5 random problems (fast testing)
    full             - All GSM8K test set (1319 problems)
    stratified       - 200 stratified samples by difficulty

Author: Assignment 1
Date: 2025
"""

import json
import time
import argparse
import sys
import random
import os
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from pathlib import Path

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
    build_verifier_messages
)


class Evaluator:
    """Unified evaluator for all prompting methods"""
    
    def __init__(self, output_file: Optional[str] = None):
        """Initialize evaluator"""
        self.client = initialize_api_client()
        self.output_file = output_file
        self.results = []
    
    def load_dataset(self, dataset_type: str, num_samples: int = 5) -> Tuple[List[str], List[str]]:
        """
        Load dataset based on type
        
        Args:
            dataset_type: 'quick-test', 'full', or 'stratified'
            num_samples: For quick-test, number of samples to load
            
        Returns:
            Tuple of (questions, answers)
        """
        gsm8k_test_path = "data/GSM8K/test.jsonl"
        
        if dataset_type == "quick-test":
            return self._load_quick_test(gsm8k_test_path, num_samples)
        elif dataset_type == "stratified":
            return self._load_stratified_sample(gsm8k_test_path)
        elif dataset_type == "full":
            return self._load_full_dataset(gsm8k_test_path)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    def _load_quick_test(self, dataset_path: str, num_samples: int) -> Tuple[List[str], List[str]]:
        """Load random quick test samples"""
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
        
        print(f"Loaded {len(sampled_questions)} random samples from GSM8K")
        return sampled_questions, sampled_answers
    
    def _load_full_dataset(self, dataset_path: str) -> Tuple[List[str], List[str]]:
        """Load full GSM8K dataset"""
        questions = []
        answers = []
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                questions.append(data['question'])
                answer = data['answer'].split('#### ')[-1].strip()
                answers.append(answer)
        
        print(f"Loaded {len(questions)} problems from full GSM8K dataset")
        return questions, answers
    
    def _load_stratified_sample(self, dataset_path: str) -> Tuple[List[str], List[str]]:
        """Load stratified sample (200 problems)"""
        stratified_path = "stratified_samples/gsm8k_stratified_sample_200.json"
        
        # Load stratified sample metadata
        with open(stratified_path, 'r', encoding='utf-8') as f:
            sample_data = json.load(f)
        
        # Extract all problem IDs
        problem_ids = []
        for difficulty_level, difficulty_data in sample_data['stratification'].items():
            problem_ids.extend(difficulty_data['problem_ids'])
        
        problem_ids_set = set(problem_ids)
        
        # Load only the stratified problems
        questions = []
        answers = []
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx in problem_ids_set:
                    data = json.loads(line)
                    questions.append(data['question'])
                    answer = data['answer'].split('#### ')[-1].strip()
                    answers.append(answer)
        
        print(f"Loaded {len(questions)} stratified samples from GSM8K")
        return questions, answers
    
    # =========================================================================
    # UTILITY FUNCTIONS
    # =========================================================================
    
    def _verify_answer_response(self, verification_response: str) -> float:
        """
        Extract verification score from response text
        Simple heuristic: check if response contains positive verification keywords
        
        Args:
            verification_response: Response from verification LLM
            
        Returns:
            Verification score (0.0 to 1.0)
        """
        lower_response = verification_response.lower()
        
        # Check for positive verification indicators
        positive_keywords = ['correct', 'valid', 'yes', 'verified', 'accurate', 'true']
        negative_keywords = ['incorrect', 'invalid', 'no', 'wrong', 'false', 'error']
        
        positive_count = sum(1 for kw in positive_keywords if kw in lower_response)
        negative_count = sum(1 for kw in negative_keywords if kw in lower_response)
        
        if positive_count > negative_count:
            return 1.0
        elif negative_count > positive_count:
            return 0.0
        else:
            return 0.5
    
    def _compare_answers(self, predicted: str, ground_truth: str) -> bool:
        """
        Compare two answers (handles both numeric and string answers)
        
        Args:
            predicted: Predicted answer from model
            ground_truth: Ground truth answer
            
        Returns:
            True if answers match, False otherwise
        """
        try:
            # Try numeric comparison
            ground_truth_num = int(float(ground_truth))
            predicted_num = int(float(str(predicted)))
            return ground_truth_num == predicted_num
        except (ValueError, TypeError):
            # Fall back to string comparison
            return str(predicted).strip() == str(ground_truth).strip()
    
    # =========================================================================
    # METHOD IMPLEMENTATIONS
    # =========================================================================
    
    def evaluate_zero_shot(self, question: str, ground_truth: str) -> Dict:
        """Evaluate zero-shot method"""
        try:
            messages = build_zero_shot_messages(question)
            start_time = time.time()
            response_data = self.client.query_claude_sonnet(
                messages=messages,
                max_tokens=DEFAULT_CONFIG['max_tokens'],
            )
            elapsed = time.time() - start_time
            
            response = response_data['content']
            predicted_answer = extract_ans_from_response(response)
            
            tokens_used = response_data['usage'].get('completion_tokens', 0)
            
            return {
                'method': 'zero-shot',
                'success': True,
                'predicted_answer': predicted_answer,
                'ground_truth': ground_truth,
                'is_correct': self._compare_answers(predicted_answer, ground_truth),
                'response': response[:500],
                'tokens_used': tokens_used,
                'time_elapsed': elapsed
            }
        except Exception as e:
            return {
                'method': 'zero-shot',
                'success': False,
                'error': str(e),
                'ground_truth': ground_truth
            }
    
    def evaluate_few_shot(self, question: str, ground_truth: str, num_shots: int = 5) -> Dict:
        """Evaluate few-shot method"""
        try:
            messages = build_few_shot_messages(question, num_examples=num_shots)
            start_time = time.time()
            response_data = self.client.query_claude_sonnet(
                messages=messages,
                max_tokens=DEFAULT_CONFIG['max_tokens'],
            )
            elapsed = time.time() - start_time
            
            response = response_data['content']
            predicted_answer = extract_ans_from_response(response)
            tokens_used = response_data['usage'].get('completion_tokens', 0)
            
            return {
                'method': f'few-shot-{num_shots}',
                'success': True,
                'predicted_answer': predicted_answer,
                'ground_truth': ground_truth,
                'is_correct': self._compare_answers(predicted_answer, ground_truth),
                'response': response[:500],
                'tokens_used': tokens_used,
                'time_elapsed': elapsed
            }
        except Exception as e:
            return {
                'method': f'few-shot-{num_shots}',
                'success': False,
                'error': str(e),
                'ground_truth': ground_truth
            }
    
    def evaluate_self_refine(self, question: str, ground_truth: str, max_iterations: int = 3) -> Dict:
        """Evaluate self-refinement method"""
        try:
            total_tokens = 0
            start_time = time.time()
            iterations_log = []
            
            # Generate initial solution
            messages = build_generate_messages(question)
            response_data = self.client.query_claude_sonnet(
                messages=messages,
                max_tokens=DEFAULT_CONFIG['max_tokens'],
            )
            response = response_data['content']
            total_tokens += response_data['usage'].get('completion_tokens', 0)
            
            predicted_answer = extract_ans_from_response(response)
            iterations_log.append({'iteration': 0, 'type': 'generation', 'answer': predicted_answer})
            
            # Refinement loop
            num_iterations = 0
            is_correct_per_feedback = False
            
            for iteration in range(max_iterations):
                # Get feedback
                feedback_messages = build_feedback_messages(question, response)
                feedback_data = self.client.query_claude_sonnet(
                    messages=feedback_messages,
                    max_tokens=DEFAULT_CONFIG['max_tokens'],
                )
                feedback = feedback_data['content']
                total_tokens += feedback_data['usage'].get('completion_tokens', 0)
                
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
                refinement_data = self.client.query_claude_sonnet(
                    messages=refinement_messages,
                    max_tokens=DEFAULT_CONFIG['max_tokens'],
                )
                response = refinement_data['content']
                total_tokens += refinement_data['usage'].get('completion_tokens', 0)
                
                predicted_answer = extract_ans_from_response(response)
            
            elapsed = time.time() - start_time
            
            return {
                'method': f'self-refine-{max_iterations}',
                'success': True,
                'predicted_answer': predicted_answer,
                'ground_truth': ground_truth,
                'is_correct': self._compare_answers(predicted_answer, ground_truth),
                'is_correct_per_feedback': is_correct_per_feedback,
                'final_solution': response[:500],
                'tokens_used': total_tokens,
                'time_elapsed': elapsed,
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
    
    def evaluate_self_verify(self, question: str, ground_truth: str, num_candidates: int = 3) -> Dict:
        """Evaluate self-verification method"""
        try:
            total_tokens = 0
            start_time = time.time()
            candidates = []
            
            # Generate multiple candidates
            for i in range(num_candidates):
                messages = build_cot_messages(question)
                response_data = self.client.query_claude_sonnet(
                    messages=messages,
                    max_tokens=DEFAULT_CONFIG['max_tokens'],
                )
                response = response_data['content']
                tokens_used = response_data['usage'].get('completion_tokens', 0)
                total_tokens += tokens_used
                
                predicted_answer = extract_ans_from_response(response)
                
                candidates.append({
                    'candidate_id': i + 1,
                    'answer': predicted_answer,
                    'temperature': [0.3, 0.5, 0.7][i] if i < 3 else 0.7,
                    'tokens_used': tokens_used
                })
            
            # Verify each candidate
            verification_scores = []
            for candidate in candidates:
                messages = build_verifier_messages(question, candidate['answer'])
                response_data = self.client.query_claude_sonnet(
                    messages=messages,
                    max_tokens=DEFAULT_CONFIG['max_tokens'],
                )
                verification_response = response_data['content']
                total_tokens += response_data['usage'].get('completion_tokens', 0)
                
                verification_score = self._verify_answer_response(verification_response)
                candidate['verification_score'] = verification_score
                verification_scores.append(verification_score)
            
            # Select best candidate
            best_idx = verification_scores.index(max(verification_scores))
            predicted_answer = candidates[best_idx]['answer']
            best_verification_score = candidates[best_idx]['verification_score']
            
            elapsed = time.time() - start_time
            
            return {
                'method': f'self-verification-{num_candidates}',
                'success': True,
                'predicted_answer': predicted_answer,
                'ground_truth': ground_truth,
                'is_correct': self._compare_answers(predicted_answer, ground_truth),
                'best_verification_score': best_verification_score,
                'num_candidates': num_candidates,
                'candidates': candidates,
                'tokens_used': total_tokens,
                'time_elapsed': elapsed
            }
        except Exception as e:
            return {
                'method': f'self-verification-{num_candidates}',
                'success': False,
                'error': str(e),
                'ground_truth': ground_truth
            }
    
    def evaluate_combined(self, question: str, ground_truth: str, max_refinement_iterations: int = 3) -> Dict:
        """Evaluate combined method"""
        try:
            total_tokens = 0
            start_time = time.time()
            iterations_log = []
            num_cycles = 0
            
            # Generate initial solution
            messages = build_generate_messages(question)
            response_data = self.client.query_claude_sonnet(
                messages=messages,
                max_tokens=DEFAULT_CONFIG['max_tokens'],
            )
            response = response_data['content']
            total_tokens += response_data['usage'].get('completion_tokens', 0)
            
            predicted_answer = extract_ans_from_response(response)
            iterations_log.append({'iteration': 0, 'type': 'generation', 'answer': predicted_answer})
            
            # Verify solution
            messages = build_verifier_messages(question, response)
            verification_data = self.client.query_claude_sonnet(
                messages=messages,
                max_tokens=DEFAULT_CONFIG['max_tokens'],
            )
            verification_response = verification_data['content']
            total_tokens += verification_data['usage'].get('completion_tokens', 0)
            
            is_correct_verification = self._verify_answer_response(verification_response) > 0.5
            
            iterations_log.append({
                'iteration': 1,
                'type': 'verification',
                'answer': predicted_answer,
                'is_correct': is_correct_verification
            })
            num_cycles = 1
            
            # Refinement if needed
            if not is_correct_verification:
                # Get feedback using build_feedback_messages
                feedback_messages = build_feedback_messages(question, response)
                feedback_data = self.client.query_claude_sonnet(
                    messages=feedback_messages,
                    max_tokens=DEFAULT_CONFIG['max_tokens'],
                )
                feedback = feedback_data['content']
                total_tokens += feedback_data['usage'].get('completion_tokens', 0)
                
                # Refinement loop
                for iteration in range(max_refinement_iterations):
                    refinement_messages = build_refinement_messages(question, response, feedback)
                    refinement_data = self.client.query_claude_sonnet(
                        messages=refinement_messages,
                        max_tokens=DEFAULT_CONFIG['max_tokens'],
                    )
                    response = refinement_data['content']
                    total_tokens += refinement_data['usage'].get('completion_tokens', 0)
                    
                    predicted_answer = extract_ans_from_response(response)
                    
                    # Verify refined solution
                    messages = build_verifier_messages(question, response)
                    verification_data = self.client.query_claude_sonnet(
                        messages=messages,
                        max_tokens=DEFAULT_CONFIG['max_tokens'],
                    )
                    verification_response = verification_data['content']
                    total_tokens += verification_data['usage'].get('completion_tokens', 0)
                    
                    is_correct_verification = self._verify_answer_response(verification_response) > 0.5
                    num_cycles += 1
                    
                    iterations_log.append({
                        'iteration': len(iterations_log),
                        'type': 'verification',
                        'answer': predicted_answer,
                        'is_correct': is_correct_verification
                    })
                    
                    if is_correct_verification:
                        break
            
            elapsed = time.time() - start_time
            
            return {
                'method': f'combined-{max_refinement_iterations}',
                'success': True,
                'predicted_answer': predicted_answer,
                'ground_truth': ground_truth,
                'is_correct': self._compare_answers(predicted_answer, ground_truth),
                'verification_passed_per_llm': is_correct_verification,
                'final_solution': response[:500],
                'tokens_used': total_tokens,
                'time_elapsed': elapsed,
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
    
    # =========================================================================
    # EVALUATION RUNNER
    # =========================================================================
    
    def evaluate(self, method: str, questions: List[str], answers: List[str]):
        """
        Run evaluation for specified method(s)
        
        Args:
            method: 'all', 'zero-shot', 'few-shot', 'self-refine', 'self-verify', or 'combined'
            questions: List of questions
            answers: List of answers
        """
        methods_to_run = self._get_methods_to_run(method)
        
        print(f"\n{'='*80}")
        print(f"EVALUATING METHODS: {', '.join(methods_to_run)}")
        print(f"DATASET SIZE: {len(questions)} problems")
        print(f"{'='*80}\n")
        
        for idx, (question, ground_truth) in enumerate(tqdm(zip(questions, answers), total=len(questions))):
            print(f"\n{'='*60}")
            print(f"Problem {idx + 1}/{len(questions)}")
            print(f"{'='*60}")
            
            for method_name in methods_to_run:
                print(f"  Running {method_name}...")
                
                if method_name == 'zero-shot':
                    result = self.evaluate_zero_shot(question, ground_truth)
                elif method_name == 'few-shot':
                    result = self.evaluate_few_shot(question, ground_truth, num_shots=5)
                elif method_name == 'self-refine':
                    result = self.evaluate_self_refine(question, ground_truth, max_iterations=3)
                elif method_name == 'self-verify':
                    result = self.evaluate_self_verify(question, ground_truth, num_candidates=3)
                elif method_name == 'combined':
                    result = self.evaluate_combined(question, ground_truth, max_refinement_iterations=3)
                else:
                    continue
                
                self.results.append(result)
                
                if result['success']:
                    status = "✓" if result['is_correct'] else "✗"
                    print(f"    {status} Answer: {result['predicted_answer']}, Correct: {result['is_correct']}")
                else:
                    print(f"    ✗ Error: {result.get('error', 'Unknown error')}")
                
                # Save incrementally
                if self.output_file:
                    with open(self.output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    def _get_methods_to_run(self, method: str) -> List[str]:
        """Get list of methods to run"""
        if method == 'all':
            return ['zero-shot', 'few-shot', 'self-refine', 'self-verify', 'combined']
        else:
            return [method]
    
    def print_summary(self):
        """Print evaluation summary"""
        if not self.results:
            print("No results to summarize")
            return
        
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY")
        print(f"{'='*80}\n")
        
        # Group by method
        methods = {}
        for result in self.results:
            method = result.get('method', 'unknown')
            if method not in methods:
                methods[method] = {'correct': 0, 'total': 0, 'tokens': 0, 'time': 0}
            
            methods[method]['total'] += 1
            if result.get('success', False):
                if result.get('is_correct', False):
                    methods[method]['correct'] += 1
                methods[method]['tokens'] += result.get('tokens_used', 0)
                methods[method]['time'] += result.get('time_elapsed', 0)
        
        # Print comparison table
        print(f"{'Method':<30} {'Accuracy':<15} {'Avg Tokens':<20} {'Total Time (s)':<15}")
        print("-" * 80)
        
        for method_name in sorted(methods.keys()):
            stats = methods[method_name]
            accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
            avg_tokens = stats['tokens'] / stats['total'] if stats['total'] > 0 else 0
            total_time = stats['time']
            
            print(f"{method_name:<30} {accuracy:>6.2f}%{'':<8} {avg_tokens:>10.1f}{'':<9} {total_time:>10.2f}")
        
        print(f"\n{'='*80}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Unified Evaluator for All Prompting Methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all methods on quick test
  python evaluator.py --mode all --dataset quick-test
  
  # Run specific method on stratified sample
  python evaluator.py --mode combined --dataset stratified
  
  # Run few-shot only on full dataset
  python evaluator.py --mode few-shot --dataset full --output results.jsonl
        """
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['all', 'zero-shot', 'few-shot', 'self-refine', 'self-verify', 'combined'],
        default='all',
        help='Method(s) to evaluate (default: all)'
    )
    
    parser.add_argument(
        '--dataset', '-d',
        choices=['quick-test', 'full', 'stratified'],
        default='quick-test',
        help='Dataset to use (default: quick-test)'
    )
    
    parser.add_argument(
        '--samples', '-s',
        type=int,
        default=5,
        help='Number of samples for quick-test (default: 5)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file for results (JSONL format)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    
    # Determine output file if not specified
    output_file = args.output
    if not output_file:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"evaluation_{args.mode}_{args.dataset}_{timestamp}.jsonl"
    
    # Clear output file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # Create evaluator
    evaluator = Evaluator(output_file=output_file)
    
    # Load dataset
    print(f"\nLoading {args.dataset} dataset...")
    questions, answers = evaluator.load_dataset(args.dataset, num_samples=args.samples)
    
    # Run evaluation
    evaluator.evaluate(args.mode, questions, answers)
    
    # Print summary
    evaluator.print_summary()
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
