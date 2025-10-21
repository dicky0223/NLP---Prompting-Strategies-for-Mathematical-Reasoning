# Task 3: Combined Method - Verification + Refinement Strategy

## Overview

This implementation combines **Self-Verification** and **Self-Refinement** methods to create a robust error-correction loop for mathematical reasoning on the GSM8K dataset.

### Strategy: Sequential Verification → Refinement

The combined method follows this synergistic approach:

1. **Generate** - Create initial solution via Chain-of-Thought reasoning
2. **Verify** - Verify the solution through backward reasoning
3. **Feedback** - If verification fails, generate specific feedback on errors
4. **Refine** - Iteratively refine the solution based on feedback
5. **Repeat** - Continue verification-refinement cycles until correct or max iterations reached

## Why This Strategy Works

The combination is effective because:

- **Verification catches errors** that the initial generation misses by working backwards from the answer
- **Refinement fixes errors** that verification detects by addressing specific feedback
- **Together they create a robust error-correction loop** that iteratively improves the solution
- **Deterministic verification** (temperature=0.0) provides consistent, reliable feedback

## Architecture

```
INPUT: Question
   ↓
[Generation] → Initial Solution
   ↓
[Verification] → Check if correct?
   ├─ YES → Output final answer ✓
   └─ NO → Continue to feedback
       ↓
[Feedback Generation] → Identify specific errors
       ↓
[Refinement] → Improve solution based on feedback
       ↓
[Verification] → Check refined solution
       ├─ YES → Output final answer ✓
       └─ NO → Repeat (up to max_iterations)
   ↓
OUTPUT: Final Answer
```

## System Prompts Used

### 1. Generation Prompt (CoT)
Instructs the model to solve math problems step-by-step and provide a numerical answer.

### 2. Verification Prompt
Instructs the model to verify if an answer is correct by:
- Working backwards from the answer
- Checking if it satisfies all problem conditions
- Responding with "VERIFICATION_PASSED" or "VERIFICATION_FAILED"

### 3. Feedback Prompt
Instructs the model to identify specific errors in a failed solution and provide actionable feedback.

### 4. Refinement Prompt
Instructs the model to improve the solution by addressing the specific feedback provided.

## Usage

### Running the Combined Method

```bash
python task3_combined_method.py
```

This will:
1. Load the GSM8K test dataset
2. Run the combined verification+refinement method with 2 iterations
3. Run the combined verification+refinement method with 3 iterations
4. Save results to JSONL files
5. Display metrics and comparison

### Output Files

- `combined_verify_refine_2iter.jsonl` - Results with 2 refinement iterations per question
- `combined_verify_refine_3iter.jsonl` - Results with 3 refinement iterations per question

### Configuration

Edit in `config.py`:
- `max_problems` - Number of test problems to evaluate
- `max_tokens` - Maximum tokens for generation/refinement
- `temperature` - Temperature for generation/refinement (0.3-0.7 recommended)

## Results Format

Each result in the JSONL file contains:

```json
{
  "question_id": 0,
  "question": "...",
  "predicted_answer": "42",
  "ground_truth": "42",
  "is_correct": true,
  "verification_passed_per_llm": true,
  "final_solution": "...",
  "process_log": [
    {
      "stage": "generation/verification/feedback_generation/refinement",
      "iteration": 0,
      "action": "initial_generation/verification/feedback_generation/refinement",
      "solution": "...",
      "predicted_answer": "...",
      "verification_result": "...",
      "feedback": "...",
      "tokens_used": 123
    }
  ],
  "total_tokens": 1234,
  "num_verification_cycles": 2
}
```

## Evaluation Metrics

The script reports:

- **Final Accuracy** - Percentage of correct answers against ground truth
- **Correct** - Number of correctly solved problems
- **Verification Passed Rate** - Percentage of answers that passed LLM verification
- **Verification Passed** - Count of answers that passed verification
- **Wall-clock time** - Total execution time in seconds
- **Avg tokens per problem** - Average token usage per problem
- **Avg verification cycles** - Average number of verification-refinement cycles
- **Total tokens** - Total tokens used across all problems

## Comparison with Other Methods

### vs. Baseline (Task 1 - Simple CoT)
- Baseline: Single attempt at solution
- Combined: Multiple verification-refinement cycles
- **Advantage**: Better accuracy through error detection and correction

### vs. Self-Verification (Task 2.1)
- Self-Verification: Multiple parallel attempts with consensus voting
- Combined: Sequential verification with targeted refinement
- **Advantage**: More efficient token usage, more targeted error correction

### vs. Self-Refinement (Task 2.2)
- Self-Refinement: Iterative refinement based on generic feedback
- Combined: Refinement based on specific verification-detected errors
- **Advantage**: Feedback is more targeted and effective

## Hyperparameters

The implementation uses these key hyperparameters:

- **max_refinement_iterations** - Maximum cycles of verify-refine (2-3 recommended)
- **verification_temperature** - Always 0.0 for deterministic verification
- **generation_temperature** - DEFAULT_CONFIG['temperature'] (typically 0.3-0.7)
- **feedback_temperature** - 0.0 for consistent, specific feedback

## Extension Possibilities

1. **Adaptive Iteration Count** - Reduce iterations if verification passes early
2. **Multi-path Exploration** - Generate multiple solutions and verify each
3. **Verification Confidence Scoring** - Use verification confidence to decide refinement
4. **Feedback-guided Generation** - Inject verification feedback into initial generation
5. **Hybrid Strategies** - Combine with self-verification voting on final answer

## Implementation Details

The method is implemented in `task3_combined_method.py` with these key functions:

- `generate_initial_solution()` - Generate solution via CoT
- `verify_solution()` - Verify if solution is correct
- `generate_feedback()` - Generate feedback on failed verification
- `refine_solution()` - Refine solution based on feedback
- `run_combined_method()` - Run full verify-refine cycle on a single question
- `run_combined_baseline()` - Run on entire dataset and report metrics

## Testing

To test with a small dataset:
1. Edit `config.py` and set `max_problems = 5`
2. Run `python task3_combined_method.py`
3. Check output files for quality

## Common Issues and Solutions

### Issue: Verification always fails
**Solution**: Check that verification prompt ends with "VERIFICATION_PASSED" text. Update prompt if needed.

### Issue: Refinement makes things worse
**Solution**: Increase feedback_temperature slightly to allow more diversity in feedback generation, or reduce max_refinement_iterations.

### Issue: Token usage is very high
**Solution**: Reduce max_refinement_iterations or reduce max_tokens in config.py.

## Future Improvements

1. Implement confidence-based verification scoring
2. Add support for multiple verification strategies
3. Implement adaptive iteration limits based on verification confidence
4. Add support for different LLM models
5. Implement result caching to reduce API calls

## References

- Self-Verification: "Large Language Models are Better Reasoners with Self-Verification" (https://arxiv.org/abs/2212.09561)
- Self-Refinement: "Self-Refine: Iterative Refinement with Self-Feedback" (similar principle)
- GSM8K: "Training Verifiers to Solve Math Word Problems" (https://arxiv.org/abs/2110.14168)

## Author Notes

This implementation demonstrates how combining error detection (verification) with error correction (refinement) creates a more robust reasoning system. The sequential verify-then-refine strategy is more efficient than parallel approaches while maintaining competitive accuracy.
