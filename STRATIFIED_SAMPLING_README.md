# Stratified Sampling & Evaluation Framework - README

## Summary

A **production-ready stratified sampling and evaluation framework** for testing multiple prompting methods on the GSM8K mathematical reasoning dataset.

### Key Achievement: Balanced Allocation Strategy

Instead of proportional sampling that would waste rare easy problems:
- **Before**: Only 6 easy problems in a 200-sample dataset
- **After**: ALL 40 available easy problems in a 200-sample dataset

This maximizes coverage while maintaining statistical soundness.

## Quick Start (5 minutes)

### 1. Generate the Stratified Sample (One-time setup)

```python
from stratified_sampling import StratifiedSampler

sampler = StratifiedSampler("data/GSM8K/test.jsonl", random_seed=42)
sample = sampler.sample(total_samples=200)
sampler.save_sample(sample, "stratified_samples/gsm8k_sample_200.json")
```

**Result**: 200-problem sample with:
- Easy: 40 problems (ALL available)
- Medium: 43 problems
- Hard: 117 problems

### 2. Evaluate Your Method

```python
from stratified_sampling import EvaluationFramework

evaluator = EvaluationFramework("stratified_samples/gsm8k_sample_200.json")

# Run your method and collect results
results = [
    {'problem_id': 0, 'response': "... solution ...\n#### 18"},
    # ... 200 results
]

# Evaluate
evaluation = evaluator.evaluate_method_result("My Method", results)
print(f"Accuracy: {evaluation['accuracy']:.2%}")
```

### 3. Compare Methods

```python
evaluations = []
for method_name in ["Zero-shot", "Few-shot", "CoT"]:
    results = evaluate_method(method_name)
    eval_dict = evaluator.evaluate_method_result(method_name, results)
    evaluations.append(eval_dict)

comparison = evaluator.compare_methods(evaluations)
```

## What's Included

### Core Files
- **`stratified_sampling.py`** (654 lines)
  - `StratifiedSampler`: Samples problems by difficulty
  - `EvaluationFramework`: Evaluates and compares methods
  - Full docstrings and type hints

- **`stratified_samples/gsm8k_stratified_sample_200.json`**
  - Pre-generated stratified sample with seed=42
  - Contains 200 problems organized by difficulty
  - Includes full problem data and statistics

### Documentation
- **`STRATIFIED_SAMPLING_GUIDE.md`** - Comprehensive technical guide
- **`BALANCED_ALLOCATION_GUIDE.md`** - Quick start guide
- **`README.md`** - This file

## Sample Composition

| Component | Count | Notes |
|-----------|-------|-------|
| **Easy** | 40 | All available (< 100 chars) |
| **Medium** | 43 | Proportional (100-200 chars) |
| **Hard** | 117 | Proportional (> 200 chars) |
| **TOTAL** | 200 | Fixed, reproducible sample |

### Why This Allocation?

1. **All Easy problems sampled** → No waste of rare samples
2. **Medium/Hard proportional** → Maintains realistic distribution
3. **Statistically sound** → Fair comparison across methods
4. **Reproducible** → Same seed=42 always gives same sample

## API Overview

### StratifiedSampler

```python
sampler = StratifiedSampler(test_jsonl_path, random_seed=42)

# Sample
sample = sampler.sample(total_samples=200)

# Save
sampler.save_sample(sample, output_path)
```

**Methods**:
- `sample(total_samples)`: Draw sample using balanced allocation
- `save_sample(sample, path)`: Persist sample to JSON

### EvaluationFramework

```python
evaluator = EvaluationFramework(sample_file_path)

# Get problems by difficulty
easy_problems = evaluator.get_problems_by_difficulty('easy')
all_by_difficulty = evaluator.get_all_problems_by_difficulty()

# Evaluate a method
evaluation = evaluator.evaluate_method_result(method_name, results)

# Compare multiple methods
comparison = evaluator.compare_methods([eval1, eval2, eval3])

# Analyze errors
analysis = evaluator.analyze_error_patterns(evaluation)
```

**Methods**:
- `get_problems_by_difficulty(difficulty)`: Filter problems
- `get_all_problems_by_difficulty()`: All organized by difficulty
- `extract_answer(response)`: Parse `#### <answer>` format
- `evaluate_method_result(name, results)`: Evaluate one method
- `compare_methods(evaluations)`: Rank multiple methods
- `analyze_error_patterns(evaluation)`: Detailed error analysis

## Usage Examples

### Example 1: Single Method Evaluation

```python
from stratified_sampling import EvaluationFramework

evaluator = EvaluationFramework("stratified_samples/gsm8k_sample_200.json")

# Simulate running zero-shot on the sample
results = []
for problem in evaluator.problems:
    response = api_client.solve(problem['question'])
    results.append({
        'problem_id': problem['problem_id'],
        'response': response
    })

# Evaluate
evaluation = evaluator.evaluate_method_result("Zero-shot", results)

# Print results
print(f"Overall: {evaluation['accuracy']:.2%}")
print(f"Easy: {evaluation['accuracy_by_difficulty']['easy']:.2%}")
print(f"Medium: {evaluation['accuracy_by_difficulty']['medium']:.2%}")
print(f"Hard: {evaluation['accuracy_by_difficulty']['hard']:.2%}")
```

### Example 2: Error Analysis

```python
evaluation = evaluator.evaluate_method_result("Method", results)
analysis = evaluator.analyze_error_patterns(evaluation, max_errors_shown=5)

# Get errors by difficulty
for difficulty, error_list in evaluation['error_details'].items():
    print(f"{difficulty}: {len(error_list)} errors")
    for error in error_list[:3]:
        print(f"  Q: {error['question'][:60]}...")
        print(f"  Predicted: {error['predicted']}")
        print(f"  Expected: {error['ground_truth']}")
```

### Example 3: Comparing All Methods

```python
methods_to_evaluate = {
    "Zero-shot": zero_shot_solver,
    "Few-shot": few_shot_solver,
    "CoT": cot_solver,
    "Self-Verification": self_verify_solver,
    "CoT + Self-Verify": combined_solver
}

evaluations = []
for method_name, solver_func in methods_to_evaluate.items():
    print(f"Evaluating {method_name}...")
    
    results = []
    for problem in evaluator.problems:
        response = solver_func(problem)
        results.append({
            'problem_id': problem['problem_id'],
            'response': response
        })
    
    evaluation = evaluator.evaluate_method_result(method_name, results)
    evaluations.append(evaluation)

# Compare all
comparison = evaluator.compare_methods(evaluations)
```

## Data Formats

### Input: Problem Format

```python
{
    "problem_id": 0,
    "question": "Janet's ducks lay 16 eggs per day. ...",
    "answer": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\n#### 18"
}
```

### Input: Method Results

```python
{
    "problem_id": 0,
    "response": "Let me think step by step...\n9 * 2 = 18\n#### 18"
}
```

### Output: Evaluation Dictionary

```python
{
    'method_name': 'Zero-shot',
    'total_problems': 200,
    'correct_count': 137,
    'accuracy': 0.685,
    'accuracy_by_difficulty': {
        'easy': 0.75,
        'medium': 0.698,
        'hard': 0.658
    },
    'problems_by_difficulty': {'easy': 40, 'medium': 43, 'hard': 117},
    'errors_by_difficulty': {'easy': 10, 'medium': 13, 'hard': 40},
    'error_details': {...}
}
```

## Response Format Requirements

Framework expects model responses with format:
```
<reasoning and calculations>
#### <numerical_answer>
```

### Valid Examples
```
"Let me solve this...\n#### 18"                    ✓
"First step: 3 + 2 = 5\n#### 5"                    ✓
"The answer is:\n#### 42.5"                        ✓
"#### 1000"                                        ✓
"Let me work through this step by step:
- First: 10 + 5 = 15
- Second: 15 * 2 = 30
#### 30"                                           ✓
```

### Invalid Examples
```
"The answer is 18"                                  ✗ (no #### format)
"#### 18 or 19"                                     ✗ (ambiguous)
"18"                                                ✗ (wrong format)
```

## Performance

- **Sampling**: < 1 second (one-time, cached)
- **Evaluation per method**: 1-5 minutes (depends on API calls)
- **Comparison**: < 5 seconds
- **Full pipeline (5 methods)**: 10-30 minutes

## Features

✓ **Reproducible Sampling**
  - Fixed random seed (42)
  - Same results every time
  - Published experiment results

✓ **Balanced Allocation**
  - All 40 easy problems included
  - Proportional medium/hard distribution
  - No waste of rare samples

✓ **Comprehensive Evaluation**
  - Overall accuracy
  - Difficulty-specific accuracy
  - Detailed error analysis
  - Method comparison and ranking

✓ **Flexible Integration**
  - Works with any prompting method
  - Supports custom answer formats (with modification)
  - Batch processing ready

## Troubleshooting

### Problem: "FileNotFoundError: test.jsonl"
**Solution**: Ensure `data/GSM8K/test.jsonl` exists in your workspace

### Problem: "Accuracy showing 0% or 100%"
**Solution**: Check that responses include `#### <answer>` format

### Problem: "Memory issues with large samples"
**Solution**: Process methods in batches or reduce sample size

## Future Enhancements

Potential improvements:
- [ ] Alternative stratification criteria (question length, answer type)
- [ ] Statistical significance testing
- [ ] Visualization tools (charts, heatmaps)
- [ ] Support for other datasets (MATH, AQuA, etc.)
- [ ] Batch API processing
- [ ] Confidence intervals

## References

**Papers**:
- Wei et al. (2023) - "Chain-of-Thought Prompting Elicits Reasoning"
- OpenAI (2021) - "Grade School Math (GSM8K)"

**Statistics**:
- Stratified Sampling: https://en.wikipedia.org/wiki/Stratified_sampling

## Status

- ✓ Framework implemented
- ✓ Balanced allocation strategy implemented
- ✓ Stratified sample generated (seed=42, 200 problems)
- ✓ Evaluation system tested
- ✓ Documentation complete

## Files

- `stratified_sampling.py` - Main framework (654 lines)
- `stratified_samples/gsm8k_stratified_sample_200.json` - Generated sample
- `STRATIFIED_SAMPLING_GUIDE.md` - Technical documentation
- `BALANCED_ALLOCATION_GUIDE.md` - Quick start guide
- `README.md` - This file

---

**Created**: October 21, 2025  
**Version**: 1.0  
**Status**: Production Ready ✓
