"""
Comprehensive Test Runner for All Task 1 & Task 2 Methods
Allows easy comparison of baseline and self-refine approaches

Author: Assignment 1
Date: 2025
"""

import json
import argparse
import sys
from typing import Dict, List, Tuple
from api_client import PoeAPIClient
from config import initialize_api_client, DEFAULT_CONFIG
from task1_baseline import (
    load_gsm8k_dataset,
    run_zero_shot_baseline,
    run_few_shot_baseline
)
from task2_self_refine import run_self_refine_baseline


def load_results_from_file(filepath: str) -> List[Dict]:
    """Load results from JSONL file"""
    results = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                results.append(json.loads(line))
    except FileNotFoundError:
        print(f"Warning: File {filepath} not found")
    return results


def compare_methods(methods_results: Dict[str, Dict]) -> None:
    """
    Compare results across different methods
    
    Args:
        methods_results: Dictionary with method names as keys and metrics as values
    """
    print(f"\n{'='*80}")
    print("COMPARISON OF ALL METHODS")
    print(f"{'='*80}\n")
    
    # Print header
    print(f"{'Method':<30} {'Accuracy':<12} {'Tokens/Prob':<12} {'Time (s)':<12} {'Problems':<10}")
    print("-" * 80)
    
    for method_name, metrics in methods_results.items():
        if 'error' in metrics:
            print(f"{method_name:<30} {'ERROR':<12}")
        else:
            accuracy = metrics.get('accuracy', 0)
            tokens_per = metrics.get('avg_tokens_per_problem', 0)
            time_taken = metrics.get('wall_clock_time', 0)
            problems = metrics.get('total_problems', 0)
            
            print(f"{method_name:<30} {accuracy:>10.2f}% {tokens_per:>10.1f} {time_taken:>10.2f}s {problems:>10}")
    
    print("-" * 80)
    
    # Calculate improvements
    if len(methods_results) > 1:
        method_names = list(methods_results.keys())
        baseline_accuracy = methods_results[method_names[0]].get('accuracy', 0)
        
        print("\nImprovements over first method:")
        for method_name in method_names[1:]:
            if 'error' not in methods_results[method_name]:
                current_accuracy = methods_results[method_name].get('accuracy', 0)
                improvement = current_accuracy - baseline_accuracy
                print(f"  {method_name}: {improvement:+.2f}%")
    
    print()


def print_detailed_analysis(results: List[Dict], method_name: str) -> None:
    """
    Print detailed analysis of results
    
    Args:
        results: List of result dictionaries from JSONL
        method_name: Name of the method
    """
    print(f"\n{'='*80}")
    print(f"DETAILED ANALYSIS: {method_name}")
    print(f"{'='*80}\n")
    
    if not results:
        print("No results found")
        return
    
    # Count statistics
    total = len(results)
    correct = sum(1 for r in results if r.get('is_correct', False))
    errors = sum(1 for r in results if 'error' in r)
    
    print(f"Total problems: {total}")
    print(f"Correct: {correct}/{total} ({100*correct/total:.2f}%)")
    print(f"Errors: {errors}")
    
    # Token analysis
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        total_tokens = sum(r.get('total_tokens', 0) for r in valid_results)
        avg_tokens = total_tokens / len(valid_results)
        print(f"\nToken Statistics:")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Avg tokens per problem: {avg_tokens:.1f}")
    
    # Analysis by difficulty (based on answer magnitude)
    if valid_results:
        print(f"\nDetailed Statistics:")
        
        # Show some correct examples
        correct_results = [r for r in valid_results if r.get('is_correct', False)]
        if correct_results:
            print(f"\n  Sample correct predictions ({len(correct_results)} total):")
            for r in correct_results[:3]:
                print(f"    Q: {r['question'][:60]}...")
                print(f"    A: {r['predicted_answer']}")
        
        # Show some incorrect examples
        incorrect_results = [r for r in valid_results if not r.get('is_correct', False)]
        if incorrect_results:
            print(f"\n  Sample incorrect predictions ({len(incorrect_results)} total):")
            for r in incorrect_results[:3]:
                print(f"    Q: {r['question'][:60]}...")
                print(f"    Predicted: {r['predicted_answer']}, Expected: {r['ground_truth']}")
    
    print()


def run_benchmark(max_problems: int = None, methods: List[str] = None) -> Dict[str, Dict]:
    """
    Run benchmarks for specified methods
    
    Args:
        max_problems: Maximum problems to test (None for full dataset)
        methods: List of method names to run. Options:
                 ['zero-shot', 'few-shot', 'self-refine-2', 'self-refine-3']
        
    Returns:
        Dictionary with results for each method
    """
    
    if methods is None:
        methods = ['zero-shot', 'few-shot', 'self-refine-2', 'self-refine-3']
    
    if max_problems is None:
        max_problems = DEFAULT_CONFIG['max_problems']
    
    print(f"\n{'='*80}")
    print("STARTING BENCHMARKS")
    print(f"{'='*80}")
    print(f"Max problems: {max_problems}")
    print(f"Methods: {', '.join(methods)}")
    
    # Initialize API client
    print("\nInitializing API client...")
    client = initialize_api_client()
    
    # Load dataset
    dataset_path = "data/GSM8K/test.jsonl"
    print(f"Loading dataset from {dataset_path}...")
    questions, answers = load_gsm8k_dataset(dataset_path)
    print(f"Loaded {len(questions)} questions")
    
    all_metrics = {}
    
    # Run Zero-shot
    if 'zero-shot' in methods:
        print("\n" + "="*80)
        try:
            metrics = run_zero_shot_baseline(
                client, questions, answers,
                max_problems=max_problems,
                output_file="zeroshot.baseline.jsonl"
            )
            all_metrics['zero-shot'] = metrics
        except Exception as e:
            print(f"Error running zero-shot: {e}")
            all_metrics['zero-shot'] = {'error': str(e)}
    
    # Run Few-shot
    if 'few-shot' in methods:
        print("\n" + "="*80)
        try:
            metrics = run_few_shot_baseline(
                client, questions, answers,
                num_examples=5,
                max_problems=max_problems,
                output_file="fewshot.baseline.jsonl"
            )
            all_metrics['few-shot'] = metrics
        except Exception as e:
            print(f"Error running few-shot: {e}")
            all_metrics['few-shot'] = {'error': str(e)}
    
    # Run Self-Refine with 2 iterations
    if 'self-refine-2' in methods:
        print("\n" + "="*80)
        try:
            metrics = run_self_refine_baseline(
                client, questions, answers,
                max_iterations=2,
                max_problems=max_problems,
                output_file="self_refine_2iter.jsonl"
            )
            all_metrics['self-refine-2'] = metrics
        except Exception as e:
            print(f"Error running self-refine-2: {e}")
            all_metrics['self-refine-2'] = {'error': str(e)}
    
    # Run Self-Refine with 3 iterations
    if 'self-refine-3' in methods:
        print("\n" + "="*80)
        try:
            metrics = run_self_refine_baseline(
                client, questions, answers,
                max_iterations=3,
                max_problems=max_problems,
                output_file="self_refine_3iter.jsonl"
            )
            all_metrics['self-refine-3'] = metrics
        except Exception as e:
            print(f"Error running self-refine-3: {e}")
            all_metrics['self-refine-3'] = {'error': str(e)}
    
    return all_metrics


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run and compare Task 1 and Task 2 methods on GSM8K"
    )
    parser.add_argument(
        '--max-problems',
        type=int,
        default=None,
        help='Maximum number of problems to test (default: from config)'
    )
    parser.add_argument(
        '--methods',
        type=str,
        nargs='+',
        default=['zero-shot', 'few-shot', 'self-refine-2', 'self-refine-3'],
        help='Methods to run (default: all)'
    )
    parser.add_argument(
        '--analyze-only',
        action='store_true',
        help='Only analyze existing results, do not run new benchmarks'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Print detailed analysis of results'
    )
    
    args = parser.parse_args()
    
    # Run benchmarks or analyze existing results
    if args.analyze_only:
        print("Analyzing existing results only...")
        all_metrics = {}
        
        file_mapping = {
            'zero-shot': 'zeroshot.baseline.jsonl',
            'few-shot': 'fewshot.baseline.jsonl',
            'self-refine-2': 'self_refine_2iter.jsonl',
            'self-refine-3': 'self_refine_3iter.jsonl'
        }
        
        for method, filepath in file_mapping.items():
            results = load_results_from_file(filepath)
            if results:
                correct = sum(1 for r in results if r.get('is_correct', False))
                total = len(results)
                accuracy = 100 * correct / total if total > 0 else 0
                
                # Estimate from available data
                all_metrics[method] = {
                    'accuracy': accuracy,
                    'num_correct': correct,
                    'total_problems': total
                }
    else:
        # Run full benchmarks
        all_metrics = run_benchmark(
            max_problems=args.max_problems,
            methods=args.methods
        )
    
    # Display comparison
    compare_methods(all_metrics)
    
    # Detailed analysis if requested
    if args.detailed:
        file_mapping = {
            'zero-shot': 'zeroshot.baseline.jsonl',
            'few-shot': 'fewshot.baseline.jsonl',
            'self-refine-2': 'self_refine_2iter.jsonl',
            'self-refine-3': 'self_refine_3iter.jsonl'
        }
        
        for method, filepath in file_mapping.items():
            results = load_results_from_file(filepath)
            if results:
                print_detailed_analysis(results, method)
    
    print("\n" + "="*80)
    print("BENCHMARKS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
