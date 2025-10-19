"""
Main Runner Script for COMP7506 NLP Assignment 1
Executes all tasks and generates comprehensive reports

Usage:
    python main_runner.py --task all
    python main_runner.py --task 1
    python main_runner.py --task 2
    python main_runner.py --task 3

Author: Assignment 1
Date: 2025
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Import task modules
from task1_baseline import (
    load_gsm8k_dataset,
    run_zero_shot_baseline,
    run_few_shot_baseline
)
from task2_advanced_methods import (
    run_cot_method,
    run_self_verification_method
)
from task3_combined_method import run_combined_method

from config import initialize_api_client, DEFAULT_CONFIG


def print_header(title: str):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def print_section(title: str):
    """Print formatted section"""
    print(f"\n{title}")
    print("-" * len(title))


def load_dataset():
    """Load GSM8K dataset"""
    print_section("Loading Dataset")
    dataset_path = "data/GSM8K/test.jsonl"
    print(f"Loading from: {dataset_path}")
    
    with open(dataset_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    questions = [item['question'] for item in data]
    answers = [item['answer'].split('#### ')[-1].strip() for item in data]
    
    print(f"✓ Loaded {len(questions)} questions")
    return questions, answers


def run_task1(client, questions: List[str], answers: List[str], max_problems: int):
    """Run Task 1: Baseline methods"""
    print_header("TASK 1: BASELINE METHODS")
    
    metrics_zero = run_zero_shot_baseline(
        client, questions, answers,
        max_problems=max_problems,
        output_file="zeroshot.baseline.jsonl"
    )
    
    metrics_few = run_few_shot_baseline(
        client, questions, answers,
        num_examples=5,
        max_problems=max_problems,
        output_file="fewshot.baseline.jsonl"
    )
    
    return {
        "zero_shot": metrics_zero,
        "few_shot": metrics_few
    }


def run_task2(client, questions: List[str], answers: List[str], max_problems: int):
    """Run Task 2: Advanced methods (CoT and Self-Verification)"""
    print_header("TASK 2: ADVANCED PROMPTING METHODS")
    
    metrics_cot = run_cot_method(
        client, questions, answers,
        max_problems=max_problems,
        output_file="cot.jsonl"
    )
    
    metrics_sv = run_self_verification_method(
        client, questions, answers,
        num_attempts=3,
        max_problems=max_problems,
        output_file="self_verification.jsonl"
    )
    
    return {
        "chain_of_thought": metrics_cot,
        "self_verification": metrics_sv
    }


def run_task3(client, questions: List[str], answers: List[str], max_problems: int):
    """Run Task 3: Combined method"""
    print_header("TASK 3: COMBINED METHOD")
    
    metrics_combined = run_combined_method(
        client, questions, answers,
        num_attempts=3,
        max_problems=max_problems,
        output_file="combined_cot_verification.jsonl"
    )
    
    return {
        "combined": metrics_combined
    }


def generate_comparison_report(all_metrics: Dict) -> str:
    """Generate comprehensive comparison report"""
    report = []
    report.append("\n" + "="*70)
    report.append("COMPREHENSIVE RESULTS COMPARISON")
    report.append("="*70)
    
    # Collect all methods
    all_methods = {}
    
    if "task1" in all_metrics:
        all_methods.update(all_metrics["task1"])
    if "task2" in all_metrics:
        all_methods.update(all_metrics["task2"])
    if "task3" in all_metrics:
        all_methods.update(all_metrics["task3"])
    
    # Sort by accuracy
    sorted_methods = sorted(
        [(name, m) for name, m in all_methods.items()],
        key=lambda x: x[1]['accuracy'],
        reverse=True
    )
    
    # Print comparison table
    report.append("\nRanking by Accuracy:\n")
    report.append(f"{'Rank':<6} {'Method':<40} {'Accuracy':<12} {'Tokens/Q':<12} {'Time (s)':<12}")
    report.append("-" * 82)
    
    for rank, (name, metrics) in enumerate(sorted_methods, 1):
        report.append(
            f"{rank:<6} {name:<40} {metrics['accuracy']:>10.2f}% "
            f"{metrics['avg_tokens_per_problem']:>11.1f} {metrics['wall_clock_time']:>11.2f}"
        )
    
    # Detailed breakdown
    report.append("\n" + "="*70)
    report.append("DETAILED METRICS")
    report.append("="*70)
    
    for method_name, method_metrics in sorted_methods:
        report.append(f"\n{method_name}:")
        report.append(f"  Accuracy:              {method_metrics['accuracy']:.2f}%")
        report.append(f"  Correct:               {method_metrics['num_correct']}/{method_metrics['total_problems']}")
        report.append(f"  Wall-clock Time:       {method_metrics['wall_clock_time']:.2f} seconds")
        report.append(f"  Avg Tokens per Problem: {method_metrics['avg_tokens_per_problem']:.1f}")
        report.append(f"  Total Tokens:          {method_metrics['total_tokens']}")
        report.append(f"  Output File:           {method_metrics['output_file']}")
    
    # Analysis
    report.append("\n" + "="*70)
    report.append("ANALYSIS & INSIGHTS")
    report.append("="*70)
    
    best_method = sorted_methods[0]
    zero_shot = all_methods.get('zero_shot', {})
    
    if zero_shot:
        improvement = best_method[1]['accuracy'] - zero_shot.get('accuracy', 0)
        report.append(f"\nBest Performer: {best_method[0]} with {best_method[1]['accuracy']:.2f}% accuracy")
        report.append(f"Improvement over Zero-shot: {improvement:+.2f}%")
    
    # Token efficiency
    report.append("\nToken Efficiency (tokens per problem):")
    sorted_by_tokens = sorted(
        [(name, m) for name, m in all_methods.items()],
        key=lambda x: x[1]['avg_tokens_per_problem']
    )
    for name, metrics in sorted_by_tokens[:3]:
        report.append(f"  {name}: {metrics['avg_tokens_per_problem']:.1f} tokens/problem")
    
    # Speed analysis
    report.append("\nSpeed (wall-clock time):")
    sorted_by_time = sorted(
        [(name, m) for name, m in all_methods.items()],
        key=lambda x: x[1]['wall_clock_time']
    )
    for name, metrics in sorted_by_time[:3]:
        report.append(f"  {name}: {metrics['wall_clock_time']:.2f}s total")
    
    report.append("\n" + "="*70)
    
    return "\n".join(report)


def save_results(all_metrics: Dict, report_text: str):
    """Save all results to files"""
    
    # Save metrics JSON
    metrics_file = "experiment_results.json"
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n✓ Metrics saved to: {metrics_file}")
    
    # Save report
    report_file = "experiment_report.txt"
    with open(report_file, 'w') as f:
        f.write(report_text)
    print(f"✓ Report saved to: {report_file}")
    
    # Print report to console
    print(report_text)


def main():
    parser = argparse.ArgumentParser(
        description="Run COMP7506 NLP Assignment 1 - Math Reasoning Tasks"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["all", "1", "2", "3"],
        help="Which task to run"
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Maximum number of problems to evaluate (default: use config)"
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip combining and verifying results"
    )
    
    args = parser.parse_args()
    
    print_header("COMP7506 NLP ASSIGNMENT 1 - MATH REASONING")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize API client
    print_section("Initializing")
    client = initialize_api_client()
    print("✓ API client initialized")
    
    # Load dataset
    questions, answers = load_dataset()
    
    max_problems = args.max_problems or DEFAULT_CONFIG['max_problems']
    print(f"✓ Evaluating on {max_problems} problems")
    
    all_metrics = {}
    
    try:
        # Run requested tasks
        if args.task in ["all", "1"]:
            all_metrics["task1"] = run_task1(client, questions, answers, max_problems)
        
        if args.task in ["all", "2"]:
            all_metrics["task2"] = run_task2(client, questions, answers, max_problems)
        
        if args.task in ["all", "3"]:
            all_metrics["task3"] = run_task3(client, questions, answers, max_problems)
        
        # Generate and save results
        report = generate_comparison_report(all_metrics)
        save_results(all_metrics, report)
        
        print_header("EXECUTION COMPLETED SUCCESSFULLY")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except KeyboardInterrupt:
        print("\n\n❌ Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
