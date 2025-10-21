"""
Comparison Script for All Math Reasoning Methods
Compares Baseline, Self-Verification, Self-Refinement, and Combined Method

This script runs all methods and generates a comprehensive comparison report.
"""

import json
import time
from typing import Dict, List
from api_client import PoeAPIClient
from config import initialize_api_client, DEFAULT_CONFIG
from data.GSM8K.evaluation import extract_ans_from_response

# Import the methods
from task1_baseline import run_baseline_prompting
from task2_self_verification import run_self_verification
from task2_self_refine import run_self_refine_baseline
from task3_combined_method import run_combined_baseline


def load_gsm8k_dataset(dataset_path: str = "data/GSM8K/test.jsonl"):
    """Load GSM8K dataset"""
    questions = []
    answers = []
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            questions.append(data['question'])
            answer = data['answer'].split('#### ')[-1].strip()
            answers.append(answer)
    
    return questions, answers


def compare_all_methods():
    """Compare all math reasoning methods"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON: ALL MATH REASONING METHODS")
    print("="*80)
    
    # Initialize API client
    print("\nInitializing API client...")
    client = initialize_api_client()
    
    # Load dataset
    print("Loading GSM8K test dataset...")
    questions, answers = load_gsm8k_dataset()
    max_problems = DEFAULT_CONFIG['max_problems']
    print(f"Loaded {len(questions)} questions (using {max_problems} for evaluation)")
    
    results_summary = {}
    all_methods = []
    
    # 1. Baseline Method (Task 1)
    print("\n" + "="*80)
    print("METHOD 1: BASELINE - Simple Chain-of-Thought")
    print("="*80)
    try:
        baseline_metrics = run_baseline_prompting(
            client, questions, answers,
            max_problems=max_problems,
            output_file="comparison_baseline.jsonl"
        )
        results_summary['baseline'] = baseline_metrics
        all_methods.append(('Baseline (CoT)', baseline_metrics))
    except Exception as e:
        print(f"Error running baseline: {e}")
        results_summary['baseline'] = {'error': str(e)}
    
    # 2. Self-Verification (Task 2.1)
    print("\n" + "="*80)
    print("METHOD 2: SELF-VERIFICATION - Multiple Attempts with Consensus")
    print("="*80)
    try:
        sv_metrics = run_self_verification(
            client, questions, answers,
            num_candidates=3,
            max_problems=max_problems,
            output_file="comparison_self_verification.jsonl"
        )
        results_summary['self_verification'] = sv_metrics
        all_methods.append(('Self-Verification', sv_metrics))
    except Exception as e:
        print(f"Error running self-verification: {e}")
        results_summary['self_verification'] = {'error': str(e)}
    
    # 3. Self-Refinement (Task 2.2)
    print("\n" + "="*80)
    print("METHOD 3: SELF-REFINEMENT - Iterative Refinement with Feedback")
    print("="*80)
    try:
        sr_metrics = run_self_refine_baseline(
            client, questions, answers,
            max_iterations=3,
            max_problems=max_problems,
            output_file="comparison_self_refinement_3iter.jsonl"
        )
        results_summary['self_refinement'] = sr_metrics
        all_methods.append(('Self-Refinement (3 iter)', sr_metrics))
    except Exception as e:
        print(f"Error running self-refinement: {e}")
        results_summary['self_refinement'] = {'error': str(e)}
    
    # 4. Combined Method (Task 3)
    print("\n" + "="*80)
    print("METHOD 4: COMBINED - Sequential Verification + Refinement")
    print("="*80)
    try:
        combined_metrics = run_combined_baseline(
            client, questions, answers,
            max_refinement_iterations=3,
            max_problems=max_problems,
            output_file="comparison_combined_3iter.jsonl"
        )
        results_summary['combined'] = combined_metrics
        all_methods.append(('Combined (Verify→Refine, 3 iter)', combined_metrics))
    except Exception as e:
        print(f"Error running combined method: {e}")
        results_summary['combined'] = {'error': str(e)}
    
    # Generate comparison report
    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON REPORT")
    print("="*80)
    
    generate_comparison_report(all_methods, max_problems)
    
    # Save detailed results
    with open('comparison_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: comparison_results.json")


def generate_comparison_report(all_methods: List, max_problems: int):
    """Generate a formatted comparison report"""
    
    if not all_methods:
        print("No methods to compare.")
        return
    
    # Filter out methods with errors
    valid_methods = [(name, metrics) for name, metrics in all_methods 
                     if 'error' not in metrics]
    
    if not valid_methods:
        print("All methods failed. Check error messages above.")
        return
    
    # Create comparison table
    print("\n" + "="*80)
    print("ACCURACY COMPARISON")
    print("="*80)
    
    print(f"{'Method':<40} {'Accuracy':<15} {'Correct':<15}")
    print("-" * 70)
    
    for method_name, metrics in valid_methods:
        accuracy = metrics.get('accuracy', 0)
        correct = metrics.get('num_correct', 0)
        print(f"{method_name:<40} {accuracy:>6.2f}%{'':<8} {correct}/{max_problems}")
    
    print("\n" + "="*80)
    print("TOKEN EFFICIENCY COMPARISON")
    print("="*80)
    
    print(f"{'Method':<40} {'Avg Tokens':<20} {'Total Tokens':<15}")
    print("-" * 75)
    
    for method_name, metrics in valid_methods:
        avg_tokens = metrics.get('avg_tokens_per_problem', 0)
        total_tokens = metrics.get('total_tokens', 0)
        print(f"{method_name:<40} {avg_tokens:>8.1f}{'':<11} {total_tokens}")
    
    print("\n" + "="*80)
    print("EXECUTION TIME COMPARISON")
    print("="*80)
    
    print(f"{'Method':<40} {'Wall Clock Time (s)':<25} {'Time per Problem (ms)':<25}")
    print("-" * 90)
    
    for method_name, metrics in valid_methods:
        wall_clock = metrics.get('wall_clock_time', 0)
        time_per_problem = (wall_clock / max_problems * 1000) if max_problems > 0 else 0
        print(f"{method_name:<40} {wall_clock:>10.2f}{'':<14} {time_per_problem:>8.1f}")
    
    print("\n" + "="*80)
    print("DETAILED METRICS COMPARISON")
    print("="*80)
    
    for method_name, metrics in valid_methods:
        print(f"\n{method_name}:")
        print(f"  Accuracy: {metrics.get('accuracy', 0):.2f}%")
        print(f"  Correct: {metrics.get('num_correct', 0)}/{max_problems}")
        print(f"  Wall-clock time: {metrics.get('wall_clock_time', 0):.2f}s")
        print(f"  Avg tokens per problem: {metrics.get('avg_tokens_per_problem', 0):.1f}")
        print(f"  Total tokens: {metrics.get('total_tokens', 0)}")
        
        # Method-specific metrics
        if 'avg_verification_cycles' in metrics:
            print(f"  Avg verification cycles: {metrics['avg_verification_cycles']:.2f}")
        if 'verification_passed_rate' in metrics:
            print(f"  Verification passed rate: {metrics['verification_passed_rate']:.2f}%")
        if 'num_candidates' in metrics:
            print(f"  Number of candidates: {metrics['num_candidates']}")
    
    # Ranking
    print("\n" + "="*80)
    print("RANKINGS")
    print("="*80)
    
    # Accuracy ranking
    print("\nAccuracy Ranking (Best → Worst):")
    sorted_by_accuracy = sorted(valid_methods, 
                                key=lambda x: x[1].get('accuracy', 0), 
                                reverse=True)
    for i, (name, metrics) in enumerate(sorted_by_accuracy, 1):
        print(f"  {i}. {name:<35} {metrics.get('accuracy', 0):.2f}%")
    
    # Token efficiency ranking
    print("\nToken Efficiency Ranking (Lowest → Highest tokens):")
    sorted_by_tokens = sorted(valid_methods, 
                              key=lambda x: x[1].get('avg_tokens_per_problem', float('inf')))
    for i, (name, metrics) in enumerate(sorted_by_tokens, 1):
        print(f"  {i}. {name:<35} {metrics.get('avg_tokens_per_problem', 0):.1f} avg tokens/problem")
    
    # Speed ranking
    print("\nSpeed Ranking (Fastest → Slowest):")
    sorted_by_speed = sorted(valid_methods, 
                             key=lambda x: x[1].get('wall_clock_time', float('inf')))
    for i, (name, metrics) in enumerate(sorted_by_speed, 1):
        time_per_problem = (metrics.get('wall_clock_time', 0) / max_problems * 1000) if max_problems > 0 else 0
        print(f"  {i}. {name:<35} {time_per_problem:.1f} ms/problem")
    
    # Summary insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    best_accuracy = sorted_by_accuracy[0] if sorted_by_accuracy else None
    most_efficient = sorted_by_tokens[0] if sorted_by_tokens else None
    fastest = sorted_by_speed[0] if sorted_by_speed else None
    
    if best_accuracy:
        print(f"\n✓ Best Accuracy: {best_accuracy[0]} ({best_accuracy[1].get('accuracy', 0):.2f}%)")
    
    if most_efficient:
        print(f"✓ Most Token Efficient: {most_efficient[0]} ({most_efficient[1].get('avg_tokens_per_problem', 0):.1f} tokens/problem)")
    
    if fastest:
        time_per_problem = (fastest[1].get('wall_clock_time', 0) / max_problems * 1000) if max_problems > 0 else 0
        print(f"✓ Fastest: {fastest[0]} ({time_per_problem:.1f} ms/problem)")
    
    # Accuracy improvement
    if len(valid_methods) > 1:
        baseline_acc = next((m[1].get('accuracy', 0) for m in valid_methods if 'Baseline' in m[0]), 0)
        if baseline_acc > 0:
            print(f"\n✓ Improvement over Baseline:")
            for name, metrics in valid_methods:
                if 'Baseline' not in name:
                    improvement = metrics.get('accuracy', 0) - baseline_acc
                    pct_improvement = (improvement / baseline_acc * 100) if baseline_acc > 0 else 0
                    symbol = "↑" if improvement > 0 else "↓" if improvement < 0 else "="
                    print(f"  {symbol} {name}: {improvement:+.2f}% ({pct_improvement:+.1f}%)")


if __name__ == "__main__":
    compare_all_methods()
