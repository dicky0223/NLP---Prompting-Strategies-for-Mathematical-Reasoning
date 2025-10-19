"""
SUBMISSION GUIDE - Step-by-step instructions for completing the assignment

Follow these steps to successfully complete your assignment submission
"""

# =============================================================================
# COMP7506 NLP ASSIGNMENT 1 - COMPLETE SUBMISSION GUIDE
# =============================================================================

## STEP 1: VERIFY SETUP
# ===========================

# 1.1 Install required packages:
#     pip install openai tqdm

# 1.2 Configure API key in config.py:
#     Edit the API_KEY variable with your Poe API key

# 1.3 Test the setup:
#     python quick_start.py
#
#     This will:
#     - Test API connectivity
#     - Run sample problems through each method
#     - Verify dataset loading
#     - Show expected output format


## STEP 2: UNDERSTAND THE METHODS
# ===================================

# Task 1: BASELINES
# ----------------
# 1. Zero-shot: Ask LLM directly without examples
#    - Simple, fast
#    - Lower accuracy baseline
#
# 2. Few-shot: Provide 5 example problems with solutions
#    - Shows the model the expected format
#    - Better accuracy than zero-shot

# Task 2: ADVANCED METHODS
# -------------------------
# 1. Chain-of-Thought (CoT):
#    - Structure reasoning into explicit steps
#    - Format: Given → Find → Steps → Verification → Answer
#    - Better for complex multi-step problems
#
# 2. Self-Verification:
#    - Generate 3 independent solutions
#    - Use majority voting to pick final answer
#    - More robust, reduces errors

# Task 3: COMBINED METHOD
# ------------------------
# - Use CoT structure (Given/Find/Steps/Verify)
# - Generate multiple attempts with different temperatures
# - Combine consensus voting with structured reasoning
# - Best overall accuracy


## STEP 3: RUN EXPERIMENTS
# ==========================

# Quick test (30 problems):
#   python main_runner.py --task all --max-problems 30

# Full evaluation (1,319 problems):
#   python main_runner.py --task all

# Individual tasks:
#   python main_runner.py --task 1  # Baselines
#   python main_runner.py --task 2  # Advanced methods
#   python main_runner.py --task 3  # Combined

# For debugging with fewer problems:
#   python main_runner.py --task all --max-problems 10


## STEP 4: COLLECT REQUIRED OUTPUT FILES
# ========================================

# The scripts automatically generate these files:
#
# TASK 1 Baseline Files:
# - zeroshot.baseline.jsonl          (Zero-shot results)
# - fewshot.baseline.jsonl           (Few-shot results)
#
# TASK 2 Advanced Methods:
# - cot.jsonl                        (Chain-of-Thought results)
# - self_verification.jsonl          (Self-Verification results)
#
# TASK 3 Combined Method:
# - combined_cot_verification.jsonl  (Combined method results)
#
# Analysis Files:
# - experiment_results.json          (Detailed metrics)
# - experiment_report.txt            (Comparison report)

# Each JSONL file contains results for all evaluated problems
# Format of each line:
# {
#   "question_id": 0,
#   "question": "problem text",
#   "predicted_answer": "42",
#   "ground_truth": "42",
#   "is_correct": true,
#   "response": "full model response",
#   "tokens_used": 150
# }


## STEP 5: VERIFY OUTPUT FILES
# ==============================

# Check that files were created:
#   ls -la *.jsonl
#   ls -la experiment_results.json

# Verify file format:
#   python verify_output.py

# View sample results:
#   head -1 zeroshot.baseline.jsonl | python -m json.tool

# Count number of results:
#   wc -l *.jsonl


## STEP 6: ANALYZE RESULTS
# ==========================

# Results are automatically compiled in:
# - experiment_report.txt    (Human-readable comparison)
# - experiment_results.json  (Machine-readable metrics)

# Key metrics reported:
# 1. Accuracy (%) - Main metric
# 2. Correct count - Number of correct predictions
# 3. Wall-clock time (s) - Total execution time
# 4. Avg tokens/problem - Average tokens per question
# 5. Total tokens - Sum of all tokens used

# Expected results (approximate):
# - Zero-shot:              ~60%
# - Few-shot:               ~65-70%
# - Chain-of-Thought:       ~70-78%
# - Self-Verification:      ~75-82%
# - Combined (CoT + SV):    ~78-85%

# Analyze results:
#   python -m json.tool experiment_results.json


## STEP 7: PREPARE REPORT
# ========================

# Your PDF report should include:
#
# 1. INTRODUCTION
#    - Problem statement
#    - Dataset description (GSM8K: 1,319 math problems)
#    - Motivation for each method
#
# 2. METHODOLOGY
#    - Detailed description of each method:
#      * Zero-shot prompting
#      * Few-shot prompting
#      * Chain-of-Thought reasoning
#      * Self-Verification approach
#      * Combined method
#    - Include prompt templates
#    - Explain key parameters (temperatures, num attempts)
#
# 3. RESULTS
#    - Accuracy table with all methods
#    - Comparison chart (accuracy vs inference cost)
#    - Individual method performance
#    - Sample outputs showing reasoning
#
# 4. ANALYSIS
#    - Why CoT works better than baselines
#    - Why Self-Verification improves robustness
#    - Complementary effects of combining methods
#    - Inference cost vs accuracy tradeoffs
#    - Failure case analysis (if applicable)
#
# 5. CONCLUSION
#    - Summary of findings
#    - Best performing method
#    - Insights about prompt engineering
#    - Future improvements
#
# 6. APPENDIX
#    - Full prompt templates
#    - Detailed metrics table
#    - Sample problem analyses
#    - References


## STEP 8: PREPARE SUBMISSION ZIP
# ================================

# Create a zip file with:
#
# ├── [UniversityNumber].py          # Your Python implementation files
# │   ├── task1_baseline.py
# │   ├── task2_advanced_methods.py
# │   ├── task3_combined_method.py
# │   ├── main_runner.py
# │   ├── config.py
# │   └── api_client.py
# │
# ├── zeroshot.baseline.jsonl        # Task 1 results
# ├── fewshot.baseline.jsonl
# │
# ├── cot.jsonl                      # Task 2 results
# ├── self_verification.jsonl
# │
# ├── combined_cot_verification.jsonl # Task 3 results
# │
# ├── experiment_results.json        # Analysis
# └── experiment_report.txt


# Commands to create submission:
#
# mkdir submission
# cp task1_baseline.py submission/
# cp task2_advanced_methods.py submission/
# cp task3_combined_method.py submission/
# cp main_runner.py submission/
# cp config.py submission/
# cp api_client.py submission/
# cp *.jsonl submission/
# cp experiment_results.json submission/
# cp experiment_report.txt submission/
# cd submission
# zip -r [UniversityNumber].zip .


## STEP 9: TROUBLESHOOTING
# =========================

# Problem: API rate limiting
# Solution: Increase rate_limit_delay in config.py to 1.0 or 2.0

# Problem: Token limit errors
# Solution: Reduce max_tokens in config.py or use fewer attempts

# Problem: Empty results files
# Solution: Check API key and internet connection, run quick_start.py

# Problem: Incorrect answer extraction
# Solution: Verify the model is using #### format, check evaluation.py

# Problem: Out of memory
# Solution: Reduce max_problems or batch size


## STEP 10: FINAL CHECKLIST
# ==========================

# Before submitting, verify:
# ☐ All 5 JSONL files are generated and contain results
# ☐ experiment_results.json and experiment_report.txt exist
# ☐ All Python files are in submission folder
# ☐ PDF report is complete with analysis
# ☐ Results show improvement from baselines
# ☐ All metrics are correct and consistent
# ☐ Zip file is properly formatted
# ☐ File names match requirements
# ☐ No sensitive information (API keys) in files
# ☐ All code is properly commented and documented


# =============================================================================
# EXPECTED FILE SIZES (approximate, for reference)
# =============================================================================

# For 30 problems evaluated:
# - zeroshot.baseline.jsonl:        30-40 KB
# - fewshot.baseline.jsonl:         40-50 KB
# - cot.jsonl:                      100-150 KB
# - self_verification.jsonl:        150-200 KB
# - combined_cot_verification.jsonl: 150-200 KB

# For full 1,319 problems:
# - Each JSONL: 1-5 MB
# - experiment_results.json: 50-100 KB
# - experiment_report.txt: 30-50 KB


# =============================================================================
# SAMPLE OUTPUT - What to expect from experiment_report.txt
# =============================================================================

"""
======================================================================
COMPREHENSIVE RESULTS COMPARISON
======================================================================

Ranking by Accuracy:

Rank  Method                                   Accuracy    Tokens/Q    Time (s)
----  ------                                   --------    --------    --------
1     combined-cot-verification-3              82.50%      250.0       120.34
2     self-verification-3                      80.25%      240.0       110.21
3     chain-of-thought                         78.15%      200.0       105.43
4     few-shot-5                               68.50%      180.0       85.21
5     zero-shot                                62.75%      150.0       78.54

======================================================================
DETAILED METRICS
======================================================================

combined-cot-verification-3:
  Accuracy:               82.50%
  Correct:                1085/1319
  Wall-clock Time:        120.34 seconds
  Avg Tokens per Problem: 250.0
  Total Tokens:           329750

...

======================================================================
ANALYSIS & INSIGHTS
======================================================================

Best Performer: combined-cot-verification-3 with 82.50% accuracy
Improvement over Zero-shot: +19.75%

Token Efficiency:
  zero-shot: 150.0 tokens/problem
  few-shot-5: 180.0 tokens/problem
  chain-of-thought: 200.0 tokens/problem

Speed:
  zero-shot: 78.54s total
  few-shot-5: 85.21s total
  chain-of-thought: 105.43s total
"""


# =============================================================================
# QUICK REFERENCE - Key Files and Their Purpose
# =============================================================================

# Source Code Files:
# - task1_baseline.py: Implements zero-shot and few-shot baselines
# - task2_advanced_methods.py: Implements CoT and Self-Verification
# - task3_combined_method.py: Combines CoT with Self-Verification
# - main_runner.py: Main orchestrator that runs all tasks
# - config.py: Configuration, API initialization
# - api_client.py: API communication with Poe/Claude

# Data Files:
# - data/GSM8K/test.jsonl: Test dataset (provided)
# - data/GSM8K/evaluation.py: Answer extraction (provided)

# Output Files:
# - *.jsonl: Detailed results for each method
# - experiment_results.json: Aggregated metrics
# - experiment_report.txt: Human-readable comparison

# Utility Scripts:
# - quick_start.py: Quick verification of setup
# - This file: Submission guide


# =============================================================================
# GETTING HELP
# =============================================================================

# If something doesn't work:
#
# 1. Run quick_start.py to test basic setup
# 2. Check the README_IMPLEMENTATION.md for detailed documentation
# 3. Review method descriptions in each task file
# 4. Check error messages - they usually indicate the problem
# 5. Verify API key is correctly set
# 6. Ensure dataset is accessible at data/GSM8K/test.jsonl

print(__doc__)
