"""
Quick Start Guide - Run this first to test your setup

This script:
1. Verifies API connectivity
2. Tests with a small sample (1 problem)
3. Demonstrates each method
4. Checks output format
"""

import json
import sys
from typing import List, Dict

from config import initialize_api_client, DEFAULT_CONFIG
from data.GSM8K.evaluation import extract_ans_from_response
from api_client import PoeAPIClient


def print_header(text: str):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def test_api_connection(client: PoeAPIClient) -> bool:
    """Test basic API connectivity"""
    print_header("TEST 1: API Connection")
    
    try:
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'API working' if you received this."}
        ]
        
        response = client.query_claude_sonnet(
            messages=test_messages,
            max_tokens=100,
            temperature=0.0
        )
        
        if response['content']:
            print(f"✓ API connection successful")
            print(f"  Response: {response['content'][:50]}...")
            return True
        else:
            print("✗ API returned empty response")
            return False
            
    except Exception as e:
        print(f"✗ API connection failed: {e}")
        return False


def test_zero_shot(client: PoeAPIClient) -> Dict:
    """Test zero-shot method"""
    print_header("TEST 2: Zero-Shot Prompting")
    
    question = "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?"
    
    messages = [
        {"role": "system", "content": "You are an expert math problem solver. Solve this math problem step by step. Use #### [answer] format for the final answer."},
        {"role": "user", "content": f"Question: {question}"}
    ]
    
    try:
        print(f"\nQuestion: {question[:60]}...")
        
        response = client.query_claude_sonnet(
            messages=messages,
            max_tokens=500,
            temperature=0.0
        )
        
        answer = extract_ans_from_response(response['content'])
        
        print(f"Model response: {response['content'][:100]}...")
        print(f"Extracted answer: {answer}")
        print(f"Tokens used: {response['usage']['completion_tokens']}")
        
        return {
            "answer": str(answer),
            "expected": "6",
            "correct": str(answer).strip() == "6",
            "tokens": response['usage']['completion_tokens']
        }
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return {"error": str(e)}


def test_cot(client: PoeAPIClient) -> Dict:
    """Test Chain-of-Thought method"""
    print_header("TEST 3: Chain-of-Thought")
    
    question = "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?"
    
    cot_system = """You are an expert math problem solver. Solve step by step:
1. Identify what we know
2. Identify what we need to find
3. Break into steps
4. Show calculations
5. Verify answer

Format:
Given: ...
Find: ...
Step 1: ...
Step 2: ...
Verification: ...
#### [answer]"""
    
    messages = [
        {"role": "system", "content": cot_system},
        {"role": "user", "content": f"Question: {question}"}
    ]
    
    try:
        print(f"\nQuestion: {question[:60]}...")
        
        response = client.query_claude_sonnet(
            messages=messages,
            max_tokens=500,
            temperature=0.0
        )
        
        answer = extract_ans_from_response(response['content'])
        
        print(f"Model response: {response['content'][:150]}...")
        print(f"Extracted answer: {answer}")
        print(f"Tokens used: {response['usage']['completion_tokens']}")
        
        return {
            "answer": str(answer),
            "expected": "39",
            "correct": str(answer).strip() == "39",
            "tokens": response['usage']['completion_tokens']
        }
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return {"error": str(e)}


def test_self_verification(client: PoeAPIClient) -> Dict:
    """Test Self-Verification method"""
    print_header("TEST 4: Self-Verification (Multiple Attempts)")
    
    question = "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?"
    
    messages = [
        {"role": "system", "content": "Solve this math problem step by step, then verify your answer. Format: #### [answer]"},
        {"role": "user", "content": f"Question: {question}"}
    ]
    
    try:
        print(f"\nQuestion: {question[:60]}...")
        print("Generating multiple solutions with different temperatures...")
        
        answers = []
        total_tokens = 0
        
        temperatures = [0.0, 0.3, 0.5]
        
        for i, temp in enumerate(temperatures, 1):
            response = client.query_claude_sonnet(
                messages=messages,
                max_tokens=500,
                temperature=temp
            )
            
            answer = extract_ans_from_response(response['content'])
            answers.append(str(answer))
            total_tokens += response['usage']['completion_tokens']
            
            print(f"  Attempt {i} (temp={temp}): answer={answer}, tokens={response['usage']['completion_tokens']}")
        
        # Use majority vote
        from collections import Counter
        most_common = Counter(answers).most_common(1)[0][0]
        consensus = Counter(answers).most_common(1)[0][1]
        
        print(f"\nFinal answer (by consensus): {most_common} (voted by {consensus}/{len(temperatures)} attempts)")
        print(f"Total tokens: {total_tokens}")
        
        return {
            "answers": answers,
            "final_answer": most_common,
            "expected": "9",
            "correct": most_common.strip() == "9",
            "consensus": consensus,
            "tokens": total_tokens
        }
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return {"error": str(e)}


def test_data_loading() -> bool:
    """Test dataset loading"""
    print_header("TEST 5: Dataset Loading")
    
    try:
        dataset_path = "data/GSM8K/test.jsonl"
        
        with open(dataset_path, 'r') as f:
            data = [json.loads(line) for line in f]
        
        print(f"✓ Successfully loaded {len(data)} problems")
        
        # Show sample
        sample = data[0]
        print(f"\nSample problem:")
        print(f"  Question: {sample['question'][:60]}...")
        print(f"  Answer: {sample['answer'][:40]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return False


def run_tests():
    """Run all tests"""
    print_header("QUICK START TESTS")
    print("Testing API and implementation setup...")
    
    results = {
        "setup": True,
        "tests": {}
    }
    
    # Initialize API
    print("\nInitializing API client...")
    try:
        client = initialize_api_client()
        print("✓ API client initialized")
    except Exception as e:
        print(f"✗ Failed to initialize API client: {e}")
        print("\nPlease check:")
        print("1. Your API key in config.py")
        print("2. Internet connection")
        print("3. API service status")
        return False
    
    # Test 1: API Connection
    if not test_api_connection(client):
        results["setup"] = False
    
    # Test 2: Zero-Shot
    results["tests"]["zero_shot"] = test_zero_shot(client)
    
    # Test 3: CoT
    results["tests"]["cot"] = test_cot(client)
    
    # Test 4: Self-Verification
    results["tests"]["self_verification"] = test_self_verification(client)
    
    # Test 5: Data Loading
    if not test_data_loading():
        results["setup"] = False
    
    # Summary
    print_header("QUICK START SUMMARY")
    
    all_passed = all(
        test.get("correct", False)
        for test in results["tests"].values()
        if isinstance(test, dict) and "correct" in test
    )
    
    print(f"\n✓ Setup: {'PASS' if results['setup'] else 'FAIL'}")
    print(f"✓ Tests: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    
    if all_passed:
        print("\n" + "="*60)
        print("All quick start tests passed! Ready to run full tasks.")
        print("\nRun full evaluation with:")
        print("  python main_runner.py --task all --max-problems 30")
        print("="*60)
    else:
        print("\n⚠ Some tests failed. Please check error messages above.")
    
    return all_passed


if __name__ == "__main__":
    try:
        success = run_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
