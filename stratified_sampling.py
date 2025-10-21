"""
Stratified Sampling and Evaluation Framework for GSM8K Dataset

This module implements:
1. Stratified sampling of GSM8K problems by difficulty
2. Evaluation framework for multiple prompting methods
3. Detailed error analysis and comparison

Author: Assignment 1
Date: 2025
"""

import json
import random
import os
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import statistics


@dataclass
class DifficultyStats:
    """Statistics for a difficulty level"""
    level: str
    count: int
    problem_ids: List[int]
    avg_length: float
    min_length: int
    max_length: int


@dataclass
class StratifiedSample:
    """Container for stratified sample data"""
    sample_size: int
    seed: int
    stratification: Dict[str, Dict]
    problems: List[Dict]
    difficulty_stats: Dict[str, DifficultyStats]


class StratifiedSampler:
    """
    Handles stratified sampling of GSM8K dataset by difficulty level using balanced allocation.
    
    Allocation Strategy:
    1. Sample ALL available easy problems (40 out of 40)
    2. Proportionally allocate remaining samples between medium and hard
    
    Difficulty is determined by solution length:
    - Easy: < 100 characters
    - Medium: 100-200 characters
    - Hard: > 200 characters
    """
    
    def __init__(self, test_jsonl_path: str, random_seed: int = 42):
        """
        Initialize the sampler.
        
        Args:
            test_jsonl_path: Path to the GSM8K test.jsonl file
            random_seed: Random seed for reproducibility (default: 42)
        """
        self.test_jsonl_path = test_jsonl_path
        self.random_seed = random_seed
        random.seed(random_seed)
        
        self.problems = []
        self.problems_by_difficulty = {
            'easy': [],
            'medium': [],
            'hard': []
        }
        
        self._load_problems()
        self._categorize_by_difficulty()
    
    def _load_problems(self):
        """Load all problems from the JSONL file."""
        with open(self.test_jsonl_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                problem = json.loads(line.strip())
                problem['problem_id'] = idx
                self.problems.append(problem)
        
        print(f"[OK] Loaded {len(self.problems)} problems from GSM8K test set")
    
    def _categorize_by_difficulty(self):
        """
        Categorize problems by solution length (difficulty).
        
        Difficulty levels:
        - Easy: solution length < 100 characters
        - Medium: solution length 100-200 characters
        - Hard: solution length > 200 characters
        """
        for problem in self.problems:
            answer = problem.get('answer', '')
            solution_length = len(answer)
            
            if solution_length < 100:
                self.problems_by_difficulty['easy'].append({
                    'problem_id': problem['problem_id'],
                    'length': solution_length
                })
            elif solution_length <= 200:
                self.problems_by_difficulty['medium'].append({
                    'problem_id': problem['problem_id'],
                    'length': solution_length
                })
            else:
                self.problems_by_difficulty['hard'].append({
                    'problem_id': problem['problem_id'],
                    'length': solution_length
                })
        
        # Print statistics
        print("\n[STATS] Difficulty Distribution:")
        for level, items in self.problems_by_difficulty.items():
            count = len(items)
            lengths = [item['length'] for item in items]
            print(f"  {level.upper():6s}: {count:4d} problems | "
                  f"length: {min(lengths):3d}-{max(lengths):3d} chars "
                  f"(avg: {statistics.mean(lengths):6.1f})")
    
    def _calculate_proportional_allocation(self, total_sample: int) -> Dict[str, int]:
        """
        Calculate balanced allocation across difficulty levels.
        
        Strategy:
        1. Sample ALL available easy problems (constraint: only 40 exist)
        2. Distribute remaining samples proportionally between medium and hard
        
        For 200 samples and 1,319 total problems:
        - Easy: ALL 40 available
        - Medium & Hard: Remaining 160 samples split proportionally
        
        This ensures:
        ✓ No waste of rare easy problems
        ✓ Balanced representation across available difficulties
        ✓ Fair sampling for medium and hard problems
        
        Args:
            total_sample: Total number of samples to draw
            
        Returns:
            Dictionary with counts for each difficulty level
        """
        allocations = {}
        
        # Step 1: Allocate all available easy problems
        easy_available = len(self.problems_by_difficulty['easy'])
        allocations['easy'] = min(easy_available, total_sample)
        
        # Step 2: Distribute remaining samples proportionally among medium and hard
        remaining_samples = total_sample - allocations['easy']
        medium_available = len(self.problems_by_difficulty['medium'])
        hard_available = len(self.problems_by_difficulty['hard'])
        total_medium_hard = medium_available + hard_available
        
        if total_medium_hard > 0:
            # Calculate proportions for medium and hard
            medium_proportion = medium_available / total_medium_hard
            hard_proportion = hard_available / total_medium_hard
            
            allocations['medium'] = round(medium_proportion * remaining_samples)
            allocations['hard'] = remaining_samples - allocations['medium']
        else:
            allocations['medium'] = 0
            allocations['hard'] = remaining_samples
        
        return allocations
    
    def sample(self, total_samples: int = 200) -> StratifiedSample:
        """
        Draw a stratified sample from the dataset.
        
        Args:
            total_samples: Total number of samples to draw (default: 200)
            
        Returns:
            StratifiedSample object containing sample metadata and problems
        """
        print(f"\n[SAMPLING] Drawing stratified sample of {total_samples} problems...")
        
        # Calculate proportional allocation
        allocations = self._calculate_proportional_allocation(total_samples)
        
        # Sample from each difficulty level
        sampled_problem_ids = {
            'easy': [],
            'medium': [],
            'hard': []
        }
        
        print("\n[ALLOCATION] Sampling allocation:")
        for level, count in allocations.items():
            available_ids = [item['problem_id'] 
                           for item in self.problems_by_difficulty[level]]
            sampled_ids = random.sample(available_ids, min(count, len(available_ids)))
            sampled_problem_ids[level] = sorted(sampled_ids)
            print(f"  {level.upper():6s}: {len(sampled_ids):3d} sampled from {len(available_ids):4d} available")
        
        # Gather full problem data for sampled IDs
        all_sampled_ids = set()
        for ids in sampled_problem_ids.values():
            all_sampled_ids.update(ids)
        
        sampled_problems = [self.problems[pid] for pid in sorted(all_sampled_ids)]
        
        # Calculate statistics
        difficulty_stats = {}
        for level, ids in sampled_problem_ids.items():
            lengths = [len(self.problems[pid]['answer']) for pid in ids]
            difficulty_stats[level] = DifficultyStats(
                level=level,
                count=len(ids),
                problem_ids=ids,
                avg_length=statistics.mean(lengths) if lengths else 0,
                min_length=min(lengths) if lengths else 0,
                max_length=max(lengths) if lengths else 0
            )
        
        # Create stratification metadata
        stratification = {
            level: {
                'count': len(sampled_problem_ids[level]),
                'problem_ids': sampled_problem_ids[level]
            }
            for level in ['easy', 'medium', 'hard']
        }
        
        sample = StratifiedSample(
            sample_size=total_samples,
            seed=self.random_seed,
            stratification=stratification,
            problems=sampled_problems,
            difficulty_stats={
                level: asdict(stats) 
                for level, stats in difficulty_stats.items()
            }
        )
        
        print(f"\n[OK] Total sampled: {len(sampled_problems)} problems")
        return sample
    
    def save_sample(self, sample: StratifiedSample, output_path: str):
        """
        Save stratified sample to JSON file.
        
        Args:
            sample: StratifiedSample object to save
            output_path: Path to save the JSON file
        """
        # Convert to serializable format
        output_data = {
            'sample_size': sample.sample_size,
            'seed': sample.seed,
            'stratification': sample.stratification,
            'difficulty_stats': sample.difficulty_stats,
            'problems': sample.problems
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SAVED] Sample saved to: {output_path}")


class EvaluationFramework:
    """
    Evaluation framework for testing and comparing multiple prompting methods.
    """
    
    def __init__(self, sample_file_path: str):
        """
        Initialize evaluation framework with sampled problems.
        
        Args:
            sample_file_path: Path to the stratified sample JSON file
        """
        self.sample_file_path = sample_file_path
        self.sample_data = self._load_sample()
        self.problems = self.sample_data['problems']
        self.results = {}
    
    def _load_sample(self) -> Dict:
        """Load the stratified sample from file."""
        with open(self.sample_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_problems_by_difficulty(self, difficulty: str) -> List[Dict]:
        """
        Get problems for a specific difficulty level.
        
        Args:
            difficulty: 'easy', 'medium', or 'hard'
            
        Returns:
            List of problem dictionaries for that difficulty level
        """
        problem_ids = self.sample_data['stratification'][difficulty]['problem_ids']
        return [p for p in self.problems if p['problem_id'] in problem_ids]
    
    def get_all_problems_by_difficulty(self) -> Dict[str, List[Dict]]:
        """
        Get all problems organized by difficulty level.
        
        Returns:
            Dictionary with 'easy', 'medium', 'hard' keys containing problem lists
        """
        return {
            'easy': self.get_problems_by_difficulty('easy'),
            'medium': self.get_problems_by_difficulty('medium'),
            'hard': self.get_problems_by_difficulty('hard')
        }
    
    def extract_answer(self, response: str) -> Optional[str]:
        """
        Extract numerical answer from model response.
        
        Expected format: "#### <answer>"
        
        Args:
            response: Model response text
            
        Returns:
            Extracted answer or None if not found
        """
        if not response:
            return None
        
        # Look for "#### <answer>" pattern
        lines = response.split('\n')
        for line in reversed(lines):  # Search from end
            line = line.strip()
            if line.startswith('####'):
                try:
                    answer = line.replace('####', '').strip()
                    return answer
                except:
                    pass
        
        return None
    
    def extract_ground_truth(self, problem: Dict) -> Optional[str]:
        """
        Extract ground truth answer from problem.
        
        Args:
            problem: Problem dictionary
            
        Returns:
            Numerical answer from the problem
        """
        answer_text = problem.get('answer', '')
        lines = answer_text.split('\n')
        
        for line in reversed(lines):
            line = line.strip()
            if line.startswith('####'):
                try:
                    answer = line.replace('####', '').strip()
                    return answer
                except:
                    pass
        
        return None
    
    def compare_answers(self, predicted: str, ground_truth: str) -> bool:
        """
        Compare predicted and ground truth answers.
        Handles numerical equivalence and common variations.
        
        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            True if answers match, False otherwise
        """
        if not predicted or not ground_truth:
            return False
        
        # Normalize: remove commas, whitespace
        pred_norm = str(predicted).strip().replace(',', '')
        truth_norm = str(ground_truth).strip().replace(',', '')
        
        # Direct comparison
        if pred_norm == truth_norm:
            return True
        
        # Try numeric comparison
        try:
            pred_num = float(pred_norm)
            truth_num = float(truth_norm)
            return pred_num == truth_num
        except:
            return False
    
    def evaluate_method_result(self, method_name: str, results: List[Dict]) -> Dict:
        """
        Evaluate results from a prompting method.
        
        Args:
            method_name: Name of the prompting method
            results: List of result dictionaries, each containing:
                - 'problem_id': int
                - 'response': str (model response)
                - 'correct': bool (manual evaluation if needed)
                
        Returns:
            Evaluation dictionary with metrics and analysis
        """
        print(f"\n[EVAL] Evaluating method: {method_name}")
        
        correct_count = 0
        total_count = 0
        errors_by_difficulty = {'easy': [], 'medium': [], 'hard': []}
        correct_by_difficulty = {'easy': 0, 'medium': 0, 'hard': 0}
        total_by_difficulty = {'easy': 0, 'medium': 0, 'hard': 0}
        
        # Get problems by difficulty for reference
        problems_by_id = {p['problem_id']: p for p in self.problems}
        problems_by_difficulty = self.get_all_problems_by_difficulty()
        
        # Create difficulty lookup
        difficulty_lookup = {}
        for difficulty, problems in problems_by_difficulty.items():
            for problem in problems:
                difficulty_lookup[problem['problem_id']] = difficulty
        
        for result in results:
            problem_id = result['problem_id']
            problem = problems_by_id.get(problem_id)
            
            if not problem:
                continue
            
            difficulty = difficulty_lookup.get(problem_id, 'unknown')
            total_by_difficulty[difficulty] += 1
            total_count += 1
            
            # Extract ground truth
            ground_truth = self.extract_ground_truth(problem)
            
            # Extract predicted answer
            predicted = self.extract_answer(result['response'])
            
            # Check if correct
            is_correct = self.compare_answers(predicted, ground_truth)
            
            if is_correct:
                correct_count += 1
                correct_by_difficulty[difficulty] += 1
            else:
                errors_by_difficulty[difficulty].append({
                    'problem_id': problem_id,
                    'question': problem['question'],
                    'predicted': predicted,
                    'ground_truth': ground_truth,
                    'response': result['response'][:200] + '...' if len(result['response']) > 200 else result['response']
                })
        
        # Calculate metrics
        accuracy = correct_count / total_count if total_count > 0 else 0
        accuracy_by_difficulty = {
            difficulty: correct_by_difficulty[difficulty] / total_by_difficulty[difficulty]
            if total_by_difficulty[difficulty] > 0 else 0
            for difficulty in ['easy', 'medium', 'hard']
        }
        
        evaluation = {
            'method_name': method_name,
            'total_problems': total_count,
            'correct_count': correct_count,
            'accuracy': accuracy,
            'accuracy_by_difficulty': accuracy_by_difficulty,
            'problems_by_difficulty': total_by_difficulty,
            'errors_by_difficulty': {
                difficulty: len(errors_by_difficulty[difficulty])
                for difficulty in ['easy', 'medium', 'hard']
            },
            'error_details': errors_by_difficulty
        }
        
        self._print_evaluation_summary(evaluation)
        
        return evaluation
    
    def _print_evaluation_summary(self, evaluation: Dict):
        """Print formatted evaluation summary."""
        print(f"\n  Overall Accuracy: {evaluation['accuracy']:.2%} "
              f"({evaluation['correct_count']}/{evaluation['total_problems']})")
        
        print("\n  Accuracy by Difficulty:")
        for difficulty, acc in evaluation['accuracy_by_difficulty'].items():
            count = evaluation['problems_by_difficulty'][difficulty]
            errors = evaluation['errors_by_difficulty'][difficulty]
            correct = count - errors
            print(f"    {difficulty.upper():6s}: {acc:.2%} ({correct}/{count})")
    
    def compare_methods(self, method_evaluations: List[Dict]) -> Dict:
        """
        Compare multiple method evaluations.
        
        Args:
            method_evaluations: List of evaluation dictionaries from evaluate_method_result
            
        Returns:
            Comparison dictionary with rankings and insights
        """
        print("\n" + "="*60)
        print("[COMPARISON] METHOD COMPARISON ANALYSIS")
        print("="*60)
        
        # Sort by accuracy
        sorted_methods = sorted(
            method_evaluations,
            key=lambda x: x['accuracy'],
            reverse=True
        )
        
        # Print rankings
        print("\n[RANKINGS] Rankings by Overall Accuracy:")
        for rank, eval_data in enumerate(sorted_methods, 1):
            print(f"  {rank}. {eval_data['method_name']:25s} | "
                  f"Accuracy: {eval_data['accuracy']:.2%} | "
                  f"Correct: {eval_data['correct_count']}/{eval_data['total_problems']}")
        
        # Difficulty-specific analysis
        print("\n[BY_DIFFICULTY] Performance by Difficulty Level:")
        for difficulty in ['easy', 'medium', 'hard']:
            print(f"\n  {difficulty.upper()}:")
            ranked = sorted(
                method_evaluations,
                key=lambda x: x['accuracy_by_difficulty'][difficulty],
                reverse=True
            )
            for eval_data in ranked:
                acc = eval_data['accuracy_by_difficulty'][difficulty]
                problems = eval_data['problems_by_difficulty'][difficulty]
                correct = problems - eval_data['errors_by_difficulty'][difficulty]
                print(f"    {eval_data['method_name']:25s}: {acc:.2%} ({correct}/{problems})")
        
        # Calculate average accuracy across methods
        avg_accuracy = statistics.mean(e['accuracy'] for e in method_evaluations)
        print(f"\n  Average Accuracy across methods: {avg_accuracy:.2%}")
        
        # Find method with best performance on each difficulty
        print("\n[BEST] Best performing method per difficulty:")
        for difficulty in ['easy', 'medium', 'hard']:
            best = max(
                method_evaluations,
                key=lambda x: x['accuracy_by_difficulty'][difficulty]
            )
            print(f"  {difficulty.upper():6s}: {best['method_name']}")
        
        return {
            'sorted_methods': sorted_methods,
            'average_accuracy': avg_accuracy,
            'comparison_complete': True
        }
    
    def analyze_error_patterns(self, evaluation: Dict, max_errors_shown: int = 5) -> Dict:
        """
        Analyze common error patterns in evaluation results.
        
        Args:
            evaluation: Evaluation dictionary from evaluate_method_result
            max_errors_shown: Maximum errors to show per difficulty
            
        Returns:
            Analysis dictionary with insights
        """
        print(f"\n[ERROR_ANALYSIS] Error Analysis for {evaluation['method_name']}:")
        
        analysis = {
            'method_name': evaluation['method_name'],
            'total_errors': sum(evaluation['errors_by_difficulty'].values()),
            'error_breakdown': evaluation['errors_by_difficulty']
        }
        
        for difficulty, errors in evaluation['error_details'].items():
            error_count = len(errors)
            if error_count == 0:
                print(f"\n  {difficulty.upper()}: No errors! [OK]")
            else:
                print(f"\n  {difficulty.upper()}: {error_count} errors")
                # Show first few errors
                for i, error in enumerate(errors[:max_errors_shown]):
                    print(f"\n    Error {i+1}:")
                    print(f"      Problem ID: {error['problem_id']}")
                    print(f"      Q: {error['question'][:80]}...")
                    print(f"      Predicted: {error['predicted']}")
                    print(f"      Expected: {error['ground_truth']}")
                
                if error_count > max_errors_shown:
                    print(f"\n    ... and {error_count - max_errors_shown} more errors")
        
        return analysis


def main():
    """Main execution for stratified sampling and framework setup."""
    # Define paths
    workspace_root = Path(__file__).parent
    test_jsonl_path = workspace_root / "data" / "GSM8K" / "test.jsonl"
    sample_output_path = workspace_root / "stratified_samples" / "gsm8k_stratified_sample_200.json"
    
    print("="*60)
    print("GSM8K STRATIFIED SAMPLING FRAMEWORK")
    print("="*60)
    
    # Step 1: Create stratified sample
    sampler = StratifiedSampler(str(test_jsonl_path), random_seed=42)
    sample = sampler.sample(total_samples=200)
    sampler.save_sample(sample, str(sample_output_path))
    
    # Step 2: Initialize evaluation framework
    print("\n" + "="*60)
    print("EVALUATION FRAMEWORK INITIALIZED")
    print("="*60)
    
    evaluator = EvaluationFramework(str(sample_output_path))
    
    # Display sample statistics
    print("\n[STATS] Sample Statistics:")
    print(f"  Total samples: {evaluator.sample_data['sample_size']}")
    print(f"  Random seed: {evaluator.sample_data['seed']}")
    
    for difficulty, stats in evaluator.sample_data['difficulty_stats'].items():
        print(f"\n  {difficulty.upper()}:")
        print(f"    Count: {stats['count']}")
        print(f"    Avg solution length: {stats['avg_length']:.1f} chars")
        print(f"    Range: {stats['min_length']}-{stats['max_length']} chars")
    
    print("\n[OK] Framework ready for evaluation!")
    print(f"[OK] Sample file: {sample_output_path}")
    
    return {
        'sampler': sampler,
        'sample': sample,
        'evaluator': evaluator,
        'sample_path': str(sample_output_path)
    }


if __name__ == "__main__":
    result = main()
