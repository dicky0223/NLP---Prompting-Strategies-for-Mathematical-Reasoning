"""
Verification script - Checks implementation completeness and correctness
"""

import os
import json
import sys
from pathlib import Path


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def check_file_exists(filepath: str, description: str = None) -> bool:
    """Check if a file exists"""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    desc = description or filepath
    print(f"{status} {desc}")
    return exists


def check_python_syntax(filepath: str) -> bool:
    """Check if Python file has valid syntax"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            compile(f.read(), filepath, 'exec')
        return True
    except SyntaxError as e:
        print(f"    Syntax error: {e}")
        return False


def verify_jsonl_format(filepath: str) -> bool:
    """Verify JSONL file format"""
    try:
        if not os.path.exists(filepath):
            return False
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            print(f"    Warning: File is empty")
            return False
        
        # Check first few lines
        for i, line in enumerate(lines[:3]):
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                print(f"    Invalid JSON at line {i+1}: {e}")
                return False
        
        print(f"    ✓ Valid JSONL format ({len(lines)} lines)")
        return True
    except Exception as e:
        print(f"    Error: {e}")
        return False


def verify_implementation():
    """Verify complete implementation"""
    
    print_section("IMPLEMENTATION VERIFICATION")
    
    all_ok = True
    
    # Check source code files
    print_section("1. SOURCE CODE FILES")
    
    source_files = {
        "task1_baseline.py": "Task 1: Baseline methods (Zero-shot & Few-shot)",
        "task2_advanced_methods.py": "Task 2: Advanced methods (CoT & Self-Verification)",
        "task3_combined_method.py": "Task 3: Combined method (CoT + Self-Verification)",
        "main_runner.py": "Main runner orchestrator",
        "config.py": "Configuration and API setup",
        "api_client.py": "API client for Poe/Claude",
        "quick_start.py": "Quick start verification script",
    }
    
    for filename, description in source_files.items():
        if check_file_exists(filename, description):
            if filename.endswith('.py'):
                if not check_python_syntax(filename):
                    all_ok = False
                    print(f"    ✗ Python syntax error in {filename}")
        else:
            all_ok = False
    
    # Check data files
    print_section("2. DATA FILES")
    
    data_files = {
        "data/GSM8K/test.jsonl": "Test dataset",
        "data/GSM8K/train.jsonl": "Training dataset",
        "data/GSM8K/evaluation.py": "Evaluation functions",
        "data/GSM8K/baseline.py": "Baseline templates",
    }
    
    for filepath, description in data_files.items():
        check_file_exists(filepath, description)
    
    # Check documentation
    print_section("3. DOCUMENTATION")
    
    doc_files = {
        "README_IMPLEMENTATION.md": "Implementation guide",
        "SUBMISSION_GUIDE.md": "Submission guide",
        "Asm1 Requirement.txt": "Assignment requirements",
    }
    
    for filepath, description in doc_files.items():
        check_file_exists(filepath, description)
    
    # Check key functions and classes
    print_section("4. IMPLEMENTATION COMPONENTS")
    
    components = {
        "task1_baseline.py": [
            "load_gsm8k_dataset",
            "build_zero_shot_messages",
            "build_few_shot_messages",
            "run_zero_shot_baseline",
            "run_few_shot_baseline",
        ],
        "task2_advanced_methods.py": [
            "build_cot_messages",
            "build_self_verification_messages",
            "run_cot_method",
            "run_self_verification_method",
        ],
        "task3_combined_method.py": [
            "build_combined_messages",
            "run_combined_method",
        ],
        "main_runner.py": [
            "run_task1",
            "run_task2",
            "run_task3",
            "generate_comparison_report",
        ],
        "config.py": [
            "initialize_api_client",
        ],
    }
    
    for filename, functions in components.items():
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            for func_name in functions:
                found = f"def {func_name}" in content or f"class {func_name}" in content
                status = "✓" if found else "✗"
                print(f"  {status} {filename}: {func_name}")
                if not found:
                    all_ok = False
    
    # Check expected outputs
    print_section("5. EXPECTED OUTPUT FILES")
    
    output_files = {
        "zeroshot.baseline.jsonl": "Zero-shot baseline results",
        "fewshot.baseline.jsonl": "Few-shot baseline results",
        "cot.jsonl": "Chain-of-Thought results",
        "self_verification.jsonl": "Self-Verification results",
        "combined_cot_verification.jsonl": "Combined method results",
        "experiment_results.json": "Aggregated metrics",
        "experiment_report.txt": "Comparison report",
    }
    
    print("Expected output files (will be created after running):\n")
    for filename, description in output_files.items():
        exists = os.path.exists(filename)
        status = "✓" if exists else "◇"  # ◇ = will be created
        print(f"  {status} {filename:<40} - {description}")
    
    # Check API configuration
    print_section("6. API CONFIGURATION")
    
    try:
        from config import API_KEY, DEFAULT_CONFIG, initialize_api_client
        
        if API_KEY and API_KEY != "your-api-key-here":
            print("✓ API key is configured")
        else:
            print("✗ API key needs to be configured in config.py")
            all_ok = False
        
        print(f"✓ Default config loaded:")
        print(f"  - Max problems: {DEFAULT_CONFIG.get('max_problems', 'N/A')}")
        print(f"  - Temperature: {DEFAULT_CONFIG.get('temperature', 'N/A')}")
        print(f"  - Max tokens: {DEFAULT_CONFIG.get('max_tokens', 'N/A')}")
        print(f"  - Model: {DEFAULT_CONFIG.get('model', 'N/A')}")
        
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        all_ok = False
    
    # Check method implementations
    print_section("7. METHOD IMPLEMENTATIONS")
    
    methods = [
        ("Zero-shot", "task1_baseline.py", "run_zero_shot_baseline"),
        ("Few-shot", "task1_baseline.py", "run_few_shot_baseline"),
        ("Chain-of-Thought", "task2_advanced_methods.py", "run_cot_method"),
        ("Self-Verification", "task2_advanced_methods.py", "run_self_verification_method"),
        ("Combined Method", "task3_combined_method.py", "run_combined_method"),
    ]
    
    for method_name, filename, func_name in methods:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            found = f"def {func_name}" in content
            status = "✓" if found else "✗"
            print(f"{status} {method_name:<25} - {func_name}")
            if not found:
                all_ok = False
    
    # Final summary
    print_section("VERIFICATION SUMMARY")
    
    if all_ok:
        print("✓ All checks passed!")
        print("\nNext steps:")
        print("1. Configure your API key in config.py")
        print("2. Run quick_start.py to test the setup")
        print("3. Run main_runner.py --task all --max-problems 30")
        print("4. Check experiment_results.json and experiment_report.txt")
        print("\nFor full evaluation on all 1,319 problems:")
        print("  python main_runner.py --task all")
    else:
        print("✗ Some checks failed!")
        print("\nPlease fix the issues listed above and try again.")
        return False
    
    return True


def show_quick_reference():
    """Show quick reference guide"""
    print_section("QUICK REFERENCE")
    
    print("""
RUNNING EXPERIMENTS:

Quick test (30 problems):
  python main_runner.py --task all --max-problems 30

Full evaluation:
  python main_runner.py --task all

Individual tasks:
  python main_runner.py --task 1  # Baselines
  python main_runner.py --task 2  # Advanced methods
  python main_runner.py --task 3  # Combined

Test setup:
  python quick_start.py


FILES GENERATED AFTER RUNNING:

Results:
  - zeroshot.baseline.jsonl
  - fewshot.baseline.jsonl
  - cot.jsonl
  - self_verification.jsonl
  - combined_cot_verification.jsonl

Analysis:
  - experiment_results.json
  - experiment_report.txt


SUBMISSION CHECKLIST:

Required files to submit:
  [ ] All .py source files (task1, task2, task3, main_runner, config, api_client)
  [ ] zeroshot.baseline.jsonl
  [ ] fewshot.baseline.jsonl
  [ ] cot.jsonl
  [ ] self_verification.jsonl
  [ ] combined_cot_verification.jsonl
  [ ] experiment_results.json
  [ ] experiment_report.txt
  [ ] PDF report with analysis
    """)


if __name__ == "__main__":
    try:
        if verify_implementation():
            show_quick_reference()
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"\nError during verification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
