# COMP7506 NLP Assignment 1 - Math Reasoning Implementation

## Overview

This project implements prompt engineering techniques for improving mathematical reasoning in Large Language Models (LLMs) using the GSM8K dataset.

### Tasks Implemented

- **Task 1: Baseline Methods**
  - Zero-shot prompting
  - Few-shot prompting (5-shot)

- **Task 2: Advanced Methods**
  - Chain-of-Thought (CoT): Structured step-by-step reasoning
  - Self-Verification: Multiple solution generation with consensus voting

- **Task 3: Combined Method**
  - CoT + Self-Verification: Leverages both methods for optimal performance

## Project Structure

```
.
├── task1_baseline.py              # Task 1: Baseline methods
├── task2_advanced_methods.py       # Task 2: CoT and Self-Verification
├── task3_combined_method.py        # Task 3: Combined approach
├── main_runner.py                 # Main execution script
├── api_client.py                  # API client for Poe's Claude
├── config.py                      # Configuration and utilities
├── data/
│   └── GSM8K/
│       ├── test.jsonl            # Test dataset (1,319 problems)
│       ├── train.jsonl           # Training dataset
│       ├── baseline.py           # Baseline prompts
│       └── evaluation.py         # Answer extraction functions
└── self-verification-ref/         # Reference implementation
    ├── main_verifier.py
    └── utils.py
```

## Installation

### Prerequisites
- Python 3.8+
- Required packages:
  ```bash
  pip install openai tqdm
  ```

### API Setup
Set your Poe API key in `config.py`:
```python
API_KEY = "your-api-key-here"
```

## Quick Start

### Run All Tasks
```bash
python main_runner.py --task all --max-problems 30
```

### Run Individual Tasks
```bash
# Task 1: Baselines
python main_runner.py --task 1 --max-problems 30

# Task 2: Advanced Methods
python main_runner.py --task 2 --max-problems 30

# Task 3: Combined Method
python main_runner.py --task 3 --max-problems 30
```

### Full Evaluation (1,319 problems)
```bash
python main_runner.py --task all
```

## Methods Explanation

### Task 1: Baseline Methods

#### Zero-Shot Prompting
- Direct problem solving without examples
- Simple system prompt with clear instructions
- Baseline for comparison

#### Few-Shot Prompting
- 5 high-quality demonstration examples
- Shows the model the expected reasoning format
- Typically achieves better accuracy than zero-shot

### Task 2: Advanced Methods

#### Chain-of-Thought (CoT)
**Key Idea**: Decompose problems into structured steps

**Process**:
1. Given information extraction
2. Problem identification
3. Step-by-step reasoning
4. Verification
5. Final answer

**Benefits**:
- More transparent reasoning
- Better handling of multi-step problems
- Improved accuracy on complex mathematics

**Example**:
```
Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?

Given: Leah=32, Sister=42, Ate=35
Find: Remaining chocolates

Step 1: Calculate total chocolates initially
  Total = 32 + 42 = 74 chocolates

Step 2: Subtract chocolates eaten
  Remaining = 74 - 35 = 39 chocolates

Verification: 39 + 35 = 74 ✓
#### 39
```

#### Self-Verification
**Key Idea**: Generate multiple solutions and use consensus

**Process**:
1. Generate N solutions (typically 3-5) with different temperatures
2. Extract answers from each solution
3. Use majority voting to select final answer
4. More robust to occasional errors

**Benefits**:
- Reduces variance in model outputs
- Better robustness
- Self-correcting mechanism

**Temperatures Used**:
- 0.0: Most conservative/deterministic
- 0.3: Moderately constrained
- 0.5: More exploratory

### Task 3: Combined Method - CoT + Self-Verification

**Strategy**: Combine structured reasoning with consensus voting

**Process**:
1. Generate multiple CoT solutions with varying temperatures
2. Extract answers from each CoT response
3. Use majority voting for final answer selection
4. Higher confidence when all attempts agree

**Advantages**:
- Structured reasoning from CoT
- Robustness from self-verification
- Better handling of edge cases

## Output Files

After running experiments, the following files are generated:

### Result Files (JSONL format)
- `zeroshot.baseline.jsonl`: Zero-shot baseline results
- `fewshot.baseline.jsonl`: Few-shot baseline results
- `cot.jsonl`: Chain-of-Thought results
- `self_verification.jsonl`: Self-Verification results
- `combined_cot_verification.jsonl`: Combined method results

### Report Files
- `experiment_results.json`: Detailed metrics in JSON format
- `experiment_report.txt`: Human-readable comparison report

### JSONL Format
Each line is a JSON object with:
```json
{
  "question_id": 0,
  "question": "Problem statement...",
  "predicted_answer": "42",
  "ground_truth": "42",
  "is_correct": true,
  "response": "Full model response...",
  "tokens_used": 150
}
```

## Metrics

### Accuracy
- Percentage of correctly answered questions
- Extracted using the evaluation function from GSM8K

### Inference Cost
1. **Wall-clock Time**: Total execution time in seconds
2. **Average Tokens per Question**: Avg completion tokens used
3. **Total Tokens**: Sum of all tokens used

## Results Analysis

The comparison report includes:

1. **Accuracy Rankings**: Methods sorted by accuracy
2. **Detailed Metrics**: Full statistics for each method
3. **Efficiency Analysis**: 
   - Token efficiency (tokens per problem)
   - Speed comparison (wall-clock time)
4. **Improvement Analysis**: Percentage improvement over baseline

### Expected Performance Hierarchy
```
Combined (CoT + Self-Verification)
        ↓ (~5-15% better)
Self-Verification (Multiple attempts)
        ↓ (~3-8% better)
Chain-of-Thought
        ↓ (~8-15% better)
Few-Shot
        ↓ (~5-10% better)
Zero-Shot (Baseline)
```

## Configuration

Edit `config.py` to modify:
```python
DEFAULT_CONFIG = {
    "max_problems": 30,              # Number of problems to evaluate
    "temperature": 0.0,              # Temperature for base calls
    "max_tokens": 2048,              # Max tokens per response
    "model": "Claude-Sonnet-4.5",   # Model to use
    "api_retry_attempts": 3,         # Retry attempts
    "rate_limit_delay": 0.5,         # Delay between API calls (seconds)
}
```

## Key Implementation Details

### Answer Extraction
- Looks for `#### [answer]` format
- Falls back to numerical extraction if format not found
- Handles both integer and float answers
- Uses evaluation.py's `extract_ans_from_response()` function

### Consensus Mechanism (Self-Verification)
- Uses Python's `Counter` for vote counting
- Selects most common answer
- Tracks consensus strength (how many attempts agreed)

### Temperature Strategy
- **0.0**: Deterministic (same output every time)
- **0.3**: Slightly varied but consistent
- **0.5**: More exploratory, different reasoning paths
- **0.7+**: Very creative but may drift

## Troubleshooting

### API Rate Limiting
- Increase `rate_limit_delay` in config.py
- Run tasks sequentially rather than in parallel

### Token Limit Issues
- Reduce `max_tokens` in config.py
- Use fewer attempts for self-verification methods

### Memory Issues
- Process fewer problems at once
- Use `--max-problems` parameter

## Submission Checklist

- [ ] `zeroshot.baseline.jsonl` - Zero-shot results
- [ ] `fewshot.baseline.jsonl` - Few-shot results
- [ ] `cot.jsonl` - Chain-of-Thought results
- [ ] `self_verification.jsonl` - Self-Verification results
- [ ] `combined_cot_verification.jsonl` - Combined method results
- [ ] All `.py` files (source code)
- [ ] `experiment_results.json` - Detailed metrics
- [ ] `experiment_report.txt` - Comparison report
- [ ] PDF report with detailed analysis

## References

- **Chain-of-Thought**: Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
- **Self-Verification**: Weng et al., "Large Language Models are Better Reasoners with Self-Verification"
- **GSM8K Dataset**: Cobbe et al., "Training Verifiers to Solve Math Word Problems"

## Additional Resources

- GSM8K Paper: https://arxiv.org/abs/2110.14168
- CoT Paper: https://arxiv.org/abs/2201.11903
- Self-Verification Reference: `/self-verification-ref/`

## Notes

### Design Decisions

1. **Why 5 Examples for Few-Shot?**
   - Balances performance with API costs
   - Provides sufficient diversity in examples
   - Manageable context length

2. **Why 3 Attempts for Self-Verification?**
   - Good balance between accuracy and cost
   - Typically achieves consensus by attempt 2-3
   - Further attempts show diminishing returns

3. **Why Variable Temperatures?**
   - Ensures diverse reasoning paths
   - Reduces chance of same systematic error
   - Leverages model's stochasticity effectively

4. **Claude-Sonnet vs Other Models?**
   - Good balance of capability and cost
   - Reliable answer extraction
   - Consistent performance across runs

### Potential Improvements

1. **Adaptive Verification**: Increase attempts if consensus is weak
2. **Dynamic Example Selection**: Choose most relevant demonstrations
3. **Ensemble Methods**: Combine different prompting strategies
4. **Fine-tuning**: Adapt model on similar problems from training set
5. **Error Analysis**: Learn from incorrect predictions to refine prompts

## Author

Assignment 1 Implementation
COMP7506 - Natural Language Processing
University of Hong Kong

## License

This implementation is for educational purposes.

---

For questions or issues, please refer to the assignment requirements document.
