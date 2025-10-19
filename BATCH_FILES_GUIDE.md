# Batch Files Guide - COMP7506 Assignment 1

Quick reference for using the batch files to run tests and manage experiments.

## Available Batch Files

### 1. **run_tests.bat** - Main Test Runner (⭐ START HERE)
**Default: 5 problems for quick testing**

Runs the complete testing pipeline:
1. Verifies implementation
2. Quick start test
3. Full experiment (5 problems)
4. Displays results

**Usage:**
```bash
run_tests.bat
```

**What it does:**
- ✓ Checks Python installation
- ✓ Verifies all implementation files
- ✓ Tests API connectivity
- ✓ Runs all tasks (Task 1, 2, 3)
- ✓ Generates results files
- ✓ Displays comparison report

**Output:**
- `zeroshot.baseline.jsonl`
- `fewshot.baseline.jsonl`
- `cot.jsonl`
- `self_verification.jsonl`
- `combined_cot_verification.jsonl`
- `experiment_results.json`
- `experiment_report.txt`

**Time:** ~5-10 minutes

---

### 2. **run_task.bat** - Task Runner with Menu
**Interactive selection of tasks and problem counts**

Allows you to select:
- All tasks or individual task (1, 2, or 3)
- Problem count (5, 30, or 1,319)

**Usage:**
```bash
run_task.bat
```

**Menu Options:**
```
1. Run all tasks (5 problems) ⭐ RECOMMENDED
2. Run all tasks (30 problems)
3. Run all tasks (full 1319 problems)
4. Run Task 1 only (baselines) - 5 problems
5. Run Task 2 only (advanced methods) - 5 problems
6. Run Task 3 only (combined) - 5 problems
7. Run Task 1 only (baselines) - 30 problems
8. Run Task 2 only (advanced methods) - 30 problems
9. Run Task 3 only (combined) - 30 problems
0. Exit
```

**Example:**
- Press `1` to run all 5 tasks with 5 problems
- Press `4` to run only Task 1 with 5 problems
- Press `3` to run all tasks on full dataset

---

### 3. **quick_test.bat** - Quick Connectivity Test
**Tests only API and basic functionality**

Runs quick start verification:
- API connectivity
- Zero-shot prompting
- Chain-of-Thought
- Self-Verification
- Dataset loading

**Usage:**
```bash
quick_test.bat
```

**Time:** ~2-3 minutes

**When to use:**
- Quick check if everything works
- Verify API key is configured
- Before running full experiments

---

### 4. **view_results.bat** - Display Results
**Shows results from previous runs**

Displays:
- Experiment report with all metrics
- List of result files
- File sizes
- Quick commands to explore results

**Usage:**
```bash
view_results.bat
```

**Displays:**
- Ranking of methods by accuracy
- Detailed metrics for each method
- Efficiency analysis
- Available result files

---

### 5. **clean.bat** - Clean Up Results
**Removes previous experiment results**

Deletes:
- All JSONL result files
- Metrics and report files
- Python cache

**Usage:**
```bash
clean.bat
```

**Prompts:**
```
Are you sure you want to delete all experiment results? (y/n):
```

**When to use:**
- Before running new experiments
- To free up space
- To avoid confusion with old results

---

## Quick Start (3 Steps)

### Step 1: Setup
```bash
REM Install dependencies
pip install openai tqdm
```

### Step 2: Configure API
Edit `config.py` and set your API key:
```python
API_KEY = "your-api-key-here"
```

### Step 3: Run Tests
```bash
run_tests.bat
```

That's it! Results will appear in about 5-10 minutes.

---

## Workflow Examples

### Example 1: Quick Test (Recommended First)
```bash
quick_test.bat          # Test setup (2-3 min)
run_tests.bat           # Run experiment with 5 problems (5-10 min)
view_results.bat        # Check results
```

### Example 2: Progressive Testing
```bash
run_task.bat            # Choose option 1 (5 problems)
[after checking results...]
run_task.bat            # Choose option 2 (30 problems)
[after checking results...]
run_task.bat            # Choose option 3 (full dataset)
```

### Example 3: Individual Task Testing
```bash
run_task.bat            # Choose option 4 (Task 1 only)
run_task.bat            # Choose option 5 (Task 2 only)
run_task.bat            # Choose option 6 (Task 3 only)
```

### Example 4: Clean and Restart
```bash
clean.bat               # Remove previous results
run_tests.bat           # Run fresh experiment
```

---

## Problem Counts & Timing

| Count | Time | Recommended For |
|-------|------|-----------------|
| 5 | 5-10 min | Quick testing, debugging |
| 30 | 20-30 min | Development testing |
| 1,319 | 2-4 hours | Full evaluation |

**Note:** Times depend on API speed and network

---

## Troubleshooting

### Error: "Python not found"
**Solution:** Install Python or add to PATH

### Error: "openai package not installed"
**Solution:** The batch file will automatically try to install it

### API connection errors
**Solution:**
1. Check API key in `config.py`
2. Check internet connection
3. Run `quick_test.bat` to diagnose

### No results generated
**Solution:**
1. Check if Python ran without errors
2. Look for error messages in console
3. Try `quick_test.bat` first
4. Check config.py settings

---

## File Structure

```
project/
├── run_tests.bat              # Main test runner (START HERE)
├── run_task.bat               # Task selection menu
├── quick_test.bat             # Quick connectivity test
├── view_results.bat           # Display results
├── clean.bat                  # Remove old results
├── config.py                  # API configuration (EDIT THIS)
├── main_runner.py             # Python orchestrator
├── task1_baseline.py
├── task2_advanced_methods.py
├── task3_combined_method.py
└── [results generated here]
```

---

## Result Files Explained

After running experiments, you'll get:

### JSONL Result Files (one per method)
- `zeroshot.baseline.jsonl` - Zero-shot baseline
- `fewshot.baseline.jsonl` - Few-shot baseline
- `cot.jsonl` - Chain-of-Thought
- `self_verification.jsonl` - Self-Verification
- `combined_cot_verification.jsonl` - Combined method

Each line is one problem result in JSON format.

### Analysis Files
- `experiment_results.json` - Machine-readable metrics
- `experiment_report.txt` - Human-readable comparison

---

## Advanced Usage

### Run from Command Line (Alternative to Batch)
```bash
REM 5 problems
python main_runner.py --task all --max-problems 5

REM 30 problems
python main_runner.py --task all --max-problems 30

REM Full dataset
python main_runner.py --task all

REM Individual tasks
python main_runner.py --task 1 --max-problems 5
python main_runner.py --task 2 --max-problems 5
python main_runner.py --task 3 --max-problems 5
```

### View Detailed Metrics
```bash
python -m json.tool experiment_results.json
```

### Compare Results
```bash
type experiment_report.txt
```

---

## Batch File Cheat Sheet

| File | Purpose | Time | Use When |
|------|---------|------|----------|
| `run_tests.bat` | Full pipeline test | 5-10 min | Getting started |
| `run_task.bat` | Choose tasks & count | Variable | Want flexibility |
| `quick_test.bat` | API verification | 2-3 min | Need quick check |
| `view_results.bat` | Show results | Instant | Already ran experiment |
| `clean.bat` | Remove old results | Instant | Starting fresh |

---

## Tips & Best Practices

✅ **Do:**
- Start with `quick_test.bat` to verify setup
- Run `run_tests.bat` for initial testing
- Check results with `view_results.bat`
- Use `run_task.bat` for specific tasks
- Use `clean.bat` between experiment runs

❌ **Don't:**
- Skip API key configuration
- Run multiple batch files simultaneously
- Delete config.py or Python files
- Use old results without regenerating
- Close console before tests finish

---

## Support

If something goes wrong:

1. **Check error message** in console
2. **Run `quick_test.bat`** to diagnose
3. **Verify API key** in config.py
4. **Check Python version**: `python --version`
5. **Review logs** in output

---

## Next Steps After Testing

Once you have results:

1. **Review experiment_report.txt** for accuracy metrics
2. **Analyze results** in experiment_results.json
3. **Write PDF report** with findings
4. **Create submission zip** with all files
5. **Submit** to Moodle

---

**Happy Testing! 🚀**
