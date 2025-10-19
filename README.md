# NLP---Prompting-Strategies-for-Mathematical-Reasoning

## Overview

This project implements advanced prompt engineering techniques for improving mathematical reasoning in Large Language Models (LLMs) using the GSM8K dataset.

### Implemented Methods

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
├── verify_implementation.py        # Verification script
├── requirements.txt               # Python dependencies
├── data/
│   └── GSM8K/
│       ├── test.jsonl            # Test dataset (1,319 problems)
│       ├── train.jsonl           # Training dataset
│       ├── baseline.py           # Baseline prompts
│       └── evaluation.py         # Answer extraction functions
└── self-verification-ref/         # Reference implementation
    └── dataset/                   # Reference datasets
```

## Installation

### Prerequisites
- Python 3.8+
- Git

### Setup

1. Clone the repository:
```bash
git clone https://github.com/dicky0223/NLP---Prompting-Strategies-for-Mathematical-Reasoning.git
cd NLP---Prompting-Strategies-for-Mathematical-Reasoning
```

2. Create and activate virtual environment:
```bash
# On Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# On macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Run All Tasks
```bash
python main_runner.py --task all --max-problems 30
```

### Run Individual Tasks
```bash
# Task 1: Baseline methods
python main_runner.py --task 1 --max-problems 30

# Task 2: Advanced methods
python main_runner.py --task 2 --max-problems 30

# Task 3: Combined method
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
- Extract given information
- Identify what to find
- Set up the calculation
- Solve step by step
- Verify the answer

#### Self-Verification
**Key Idea**: Generate multiple solutions and use consensus voting
- Generate 3 different solution attempts
- Extract answers from each attempt
- Use majority voting to determine final answer
- Increases confidence through consensus

### Task 3: Combined Method

Combines CoT and Self-Verification:
- Uses Chain-of-Thought reasoning structure
- Applies Self-Verification with multiple attempts
- Leverages benefits of both methods for optimal performance

## Results & Performance

### Tested Configuration
- **Model**: Claude-Sonnet-4.5
- **Temperature**: 0.0 (deterministic)
- **Max Tokens**: 2048
- **Dataset**: GSM8K (Grade School Math 8K)

### Output Files
- `zeroshot.baseline.jsonl` - Zero-shot results
- `fewshot.baseline.jsonl` - Few-shot results
- `cot.jsonl` - Chain-of-Thought results
- `self_verification.jsonl` - Self-Verification results
- `combined.jsonl` - Combined method results

Each file contains:
- Question ID and text
- Predicted answer
- Ground truth answer
- Full reasoning chain
- Token usage
- Correctness flag

## Dependencies

- **openai>=1.3.0** - API client library
- **tqdm>=4.65.0** - Progress bar utility
- **requests>=2.31.0** - HTTP requests library

## API Setup

This project uses the Poe API (Claude-Sonnet-4.5). 

To use your own API key, update `config.py`:
```python
API_KEY = "your-api-key-here"
```

## Configuration

Edit `config.py` to customize:
- API key
- Maximum problems to evaluate
- Temperature setting
- Token limits
- API retry attempts
- Rate limiting

## References

- GSM8K Dataset: https://github.com/openai/grade-school-math
- Chain-of-Thought Prompting: Wei et al., 2022
- Self-Verification Methods: Weng et al., 2023

## Author

COMP7506 NLP Assignment 1

## License

MIT License
