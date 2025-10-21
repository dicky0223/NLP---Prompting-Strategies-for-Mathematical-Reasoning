# Unified Evaluator - Usage Guide

## Overview

The `evaluator.py` is a comprehensive, unified framework for evaluating all prompting methods implemented in this project:

- **Zero-shot**: Simple chain-of-thought prompting
- **Few-shot**: Few-shot prompting with 5 examples
- **Self-Refinement**: Iterative refinement with feedback (3 iterations)
- **Self-Verification**: Multiple candidate generation and verification (3 candidates)
- **Combined**: Sequential verification + refinement pipeline

## Quick Start

### Basic Usage

```bash
# Run all methods on quick test (5 problems)
python evaluator.py

# Run all methods on full GSM8K dataset
python evaluator.py --dataset full

# Run all methods on stratified sample (200 problems)
python evaluator.py --dataset stratified
```

### Run Specific Method

```bash
# Run only zero-shot on quick test
python evaluator.py --mode zero-shot

# Run only self-refinement on stratified data
python evaluator.py --mode self-refine --dataset stratified

# Run only combined method
python evaluator.py --mode combined --dataset full
```

### Custom Output

```bash
# Save results to custom file
python evaluator.py --output my_results.jsonl

# Run specific method with custom output
python evaluator.py --mode few-shot --dataset full --output few_shot_results.jsonl

# Quick test with 10 problems
python evaluator.py --samples 10
```

## Command-Line Arguments

```
usage: evaluator.py [-h] [-m {all,zero-shot,few-shot,self-refine,self-verify,combined}]
                    [-d {quick-test,full,stratified}]
                    [-s SAMPLES]
                    [-o OUTPUT]
                    [--seed SEED]

optional arguments:
  -h, --help                          Show help message
  
  -m, --mode {all,zero-shot,few-shot,self-refine,self-verify,combined}
                                      Method(s) to evaluate (default: all)
  
  -d, --dataset {quick-test,full,stratified}
                                      Dataset to use (default: quick-test)
  
  -s, --samples SAMPLES              Number of samples for quick-test 
                                      (default: 5)
  
  -o, --output OUTPUT                Output file for results in JSONL format
  
  --seed SEED                        Random seed for reproducibility 
                                      (default: 42)
```

## Dataset Options

| Dataset | Size | Description |
|---------|------|-------------|
| `quick-test` | 5-N problems | Random quick test for development |
| `full` | 1,319 problems | Complete GSM8K test set |
| `stratified` | 200 problems | Stratified sample: 40 easy, 80 medium, 80 hard |

## Output Format

Results are saved in **JSONL format** (one JSON object per line):

### Common Fields (All Methods)
```json
{
  "method": "zero-shot",
  "success": true,
  "predicted_answer": "5",
  "ground_truth": "5",
  "is_correct": true,
  "response": "I need to find...",
  "tokens_used": 183,
  "time_elapsed": 4.85
}
```

### Self-Refinement Method
```json
{
  "method": "self-refine-3",
  "success": true,
  "predicted_answer": "5",
  "ground_truth": "5",
  "is_correct": true,
  "is_correct_per_feedback": true,
  "final_solution": "I need to find...",
  "tokens_used": 410,
  "time_elapsed": 9.05,
  "num_iterations": 1,
  "iterations_log": [
    {"iteration": 0, "type": "generation", "answer": "5"},
    {"iteration": 1, "type": "feedback", "is_correct": true}
  ]
}
```

### Self-Verification Method
```json
{
  "method": "self-verification-3",
  "success": true,
  "predicted_answer": "5",
  "ground_truth": "5",
  "is_correct": true,
  "best_verification_score": 0.8,
  "num_candidates": 3,
  "candidates": [
    {
      "candidate_id": 1,
      "answer": "5",
      "temperature": 0.3,
      "verification_score": 0.8,
      "tokens_used": 160
    }
  ],
  "tokens_used": 484,
  "time_elapsed": 43.47
}
```

### Combined Method
```json
{
  "method": "combined-3",
  "success": true,
  "predicted_answer": "5",
  "ground_truth": "5",
  "is_correct": true,
  "verification_passed_per_llm": true,
  "final_solution": "I need to find...",
  "tokens_used": 554,
  "time_elapsed": 11.29,
  "num_cycles": 1,
  "iterations_log": [
    {"iteration": 0, "type": "generation", "answer": "5"},
    {"iteration": 1, "type": "verification", "answer": "5", "is_correct": true}
  ]
}
```

## Usage Examples

### Example 1: Quick Test of All Methods
```bash
python evaluator.py --mode all --dataset quick-test --samples 5
```
**Output:**
- Runs 5 problems
- Tests all 5 methods (25 total evaluations)
- Saves to timestamped file like `evaluation_all_quick-test_20251022_143521.jsonl`
- Prints summary statistics for each method

### Example 2: Full Dataset Evaluation of Combined Method
```bash
python evaluator.py --mode combined --dataset full --output combined_full.jsonl
```
**Output:**
- Evaluates combined method on all 1,319 GSM8K problems
- Saves results to `combined_full.jsonl`
- Shows progress with progress bar
- Final summary with accuracy and token usage

### Example 3: Stratified Sample with Specific Method
```bash
python evaluator.py --mode self-refine --dataset stratified --output stratified_refine.jsonl
```
**Output:**
- Evaluates self-refinement on 200 stratified problems
- Balanced difficulty distribution
- Results saved to `stratified_refine.jsonl`

### Example 4: Multiple Runs with Different Seeds
```bash
python evaluator.py --dataset quick-test --seed 42 --output run1.jsonl
python evaluator.py --dataset quick-test --seed 123 --output run2.jsonl
python evaluator.py --dataset quick-test --seed 456 --output run3.jsonl
```
**Purpose:** Check stability across different random samples

## Output Summary

After evaluation completes, the console shows:

```
================================================================================
EVALUATION SUMMARY
================================================================================

Method                         Accuracy        Avg Tokens           Total Time (s)
--------------------------------------------------------------------------------
zero-shot                       80.00%          185.3                      24.32
few-shot-5                      85.00%          195.7                      28.45
self-refine-3                   88.00%          425.6                      85.23
self-verification-3             83.00%          520.4                     125.67
combined-3                      90.00%          650.2                     156.89

================================================================================

Results saved to: evaluation_all_quick-test_20251022_143521.jsonl
```

## Analyzing Results

### Load and Parse Results
```python
import json

results = []
with open('evaluation_all_quick-test_20251022_143521.jsonl', 'r') as f:
    for line in f:
        results.append(json.loads(line))

# Filter by method
zero_shot = [r for r in results if r['method'] == 'zero-shot']

# Calculate accuracy
accuracy = sum(1 for r in zero_shot if r.get('is_correct')) / len(zero_shot)
print(f"Zero-shot accuracy: {accuracy:.2%}")

# Analyze token usage
avg_tokens = sum(r.get('tokens_used', 0) for r in zero_shot) / len(zero_shot)
print(f"Average tokens: {avg_tokens:.1f}")
```

## Performance Tips

1. **Quick Testing**: Use `--dataset quick-test --samples 5` for rapid development
2. **Reproducibility**: Use `--seed 42` for consistent results
3. **Stratified Data**: Use `--dataset stratified` for balanced difficulty evaluation
4. **Memory Management**: Run large datasets in batches
5. **Progress Tracking**: Check JSONL file growth in real-time

## Troubleshooting

| Issue | Solution |
|-------|----------|
| API errors | Check your API key and internet connection |
| Out of memory | Reduce sample size or run in batches |
| Timeout errors | Increase patience with longer timeouts |
| Results not saved | Check file permissions in working directory |

## Method Descriptions

### Zero-shot
Simple chain-of-thought prompting without examples. Fastest but lowest accuracy.

### Few-shot  
Chain-of-thought with 5 example problems. Better accuracy than zero-shot.

### Self-Refinement
Iterative refinement with feedback (up to 3 iterations). Generates, gets feedback, refines.

### Self-Verification
Multiple candidates (3) with verification. Selects best candidate based on verification scores.

### Combined
Sequential verification then refinement. First verifies, then refines if needed (up to 3 refinement cycles).

## Contact & Issues

For issues or questions, refer to the main README.md or ARCHITECTURE_GUIDE.md.
