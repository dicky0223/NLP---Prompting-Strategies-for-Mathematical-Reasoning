"""
Task 2: Self-Verification Implementation
Large Language Models are Better Reasoners with Self-Verification
Based on: https://arxiv.org/abs/2212.09561

Method Overview:
1. Forward Reasoning: Generate N candidate solutions via Chain-of-Thought with varying temperatures
2. Backward Verification: Use the LLM to verify each candidate by converting to declarative statements
3. Ranking: Score candidates based on verification consistency
4. Selection: Choose the candidate with highest verification score

Author: Assignment 2
Date: 2025
"""

import json
import time
import os
import re
import random
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from api_client import PoeAPIClient
from config import initialize_api_client, DEFAULT_CONFIG
from data.GSM8K.evaluation import extract_ans_from_response


# Few-shot examples for forward reasoning (CoT)
FEW_SHOT_COT_EXAMPLES = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "reasoning": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.",
        "answer": "6"
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "reasoning": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.",
        "answer": "5"
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "reasoning": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.",
        "answer": "39"
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "reasoning": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.",
        "answer": "8"
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "reasoning": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.",
        "answer": "9"
    },
]

# Few-shot examples for backward verification (verifier)
FEW_SHOT_VERIFIER_EXAMPLES = [
    {
        "statement": "\"There are 'X' trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. The grove workers planted 6 trees today.\" What is the answer of 'X'?",
        "reasoning": "There are X trees originally. The grove workers planted 6 trees today. Then there were 21 trees after some more were planted. X + 6 = 21, X = 15.",
        "answer": "15"
    },
    {
        "statement": "\"If there are 'X' cars in the parking lot and 2 more cars arrive, there are 5 cars in the parking lot.\" What is the answer of 'X'?",
        "reasoning": "There are originally X cars. 2 more cars arrive and there are 5 cars finally. X + 2 = 5, X = 3.",
        "answer": "3"
    },
    {
        "statement": "\"Leah had 'X' chocolates and her sister had 42. If they ate 35, they have 39 pieces left in total.\" What is the answer of 'X'?",
        "reasoning": "Originally, Leah had X chocolates. Her sister had 42. X + 42 = Y. After eating 35, Y - 35 = 39, so Y = 74. X + 42 = 74, X = 32.",
        "answer": "32"
    },
]

SYSTEM_PROMPT_COT = """You are an expert math problem solver. Your task is to solve math word problems step by step using chain-of-thought reasoning.

When solving:
1. Read the problem carefully
2. Identify what is being asked
3. Show your reasoning process clearly step by step
4. Provide the final numerical answer

Always end your response with: #### [answer]
where [answer] is the numerical result."""

SYSTEM_PROMPT_VERIFIER = """You are an expert math problem verifier. Your task is to verify if a given answer is correct by checking if it satisfies the problem conditions.

When verifying:
1. Understand the declarative statement and the proposed answer
2. Work backwards from the answer to check if it satisfies all conditions
3. Show your verification process clearly
4. Provide the final numerical answer to the question

Always end your response with: #### [answer]
where [answer] is the numerical result."""


def load_gsm8k_dataset(dataset_path: str) -> Tuple[List[str], List[str]]:
    """Load GSM8K dataset from JSONL file"""
    questions = []
    answers = []
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            questions.append(data['question'])
            answer = data['answer'].split('#### ')[-1].strip()
            answers.append(answer)
    
    return questions, answers


def build_cot_messages(question: str, num_examples: int = 5) -> List[Dict[str, str]]:
    """Build messages for Chain-of-Thought forward reasoning"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_COT}
    ]
    
    # Add demonstration examples
    examples_to_use = FEW_SHOT_COT_EXAMPLES[:num_examples]
    for example in examples_to_use:
        messages.append({
            "role": "user",
            "content": f"Question: {example['question']}"
        })
        messages.append({
            "role": "assistant",
            "content": f"{example['reasoning']}\n\n#### {example['answer']}"
        })
    
    # Add the actual question
    messages.append({
        "role": "user",
        "content": f"Question: {question}\n\nPlease solve this step by step."
    })
    
    return messages


def build_verifier_messages(declarative_statement: str, num_examples: int = 3) -> List[Dict[str, str]]:
    """Build messages for backward verification"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_VERIFIER}
    ]
    
    # Add demonstration examples
    examples_to_use = FEW_SHOT_VERIFIER_EXAMPLES[:num_examples]
    for example in examples_to_use:
        messages.append({
            "role": "user",
            "content": f"Statement: {example['statement']}"
        })
        messages.append({
            "role": "assistant",
            "content": f"{example['reasoning']}\n\n#### {example['answer']}"
        })
    
    # Add the actual statement to verify
    messages.append({
        "role": "user",
        "content": f"Statement: {declarative_statement}\n\nPlease verify this step by step."
    })
    
    return messages


def extract_numeric_answers(text: str) -> List[str]:
    """Extract all numeric answers from text"""
    # First try to extract from #### format
    if "####" in text:
        answer_part = text.split("####")[-1].strip()
        numbers = re.findall(r'-?\d+\.?\d*', answer_part)
        if numbers:
            return numbers
    
    # Fallback: extract all numbers
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return numbers if numbers else [""]


def convert_to_declarative(question: str, answer: str, client: PoeAPIClient) -> str:
    """
    Convert a question-answer pair to a declarative statement using the LLM.
    
    Args:
        question: The original question
        answer: The proposed answer
        client: API client
        
    Returns:
        Declarative statement
    """
    messages = [
        {"role": "system", "content": "Convert the following question and answer into a complete declarative sentence."},
        {"role": "user", "content": f"Question: {question}\nAnswer: {answer}\n\nPlease convert this to a declarative sentence that states the answer."}
    ]
    
    try:
        response_data = client.query_claude_sonnet(
            messages=messages,
            max_tokens=256,
            temperature=0.0
        )
        return response_data['content'].strip()
    except Exception as e:
        print(f"Error converting to declarative: {e}")
        # Fallback declarative statement
        return f"{question.rstrip('?')}. The answer is {answer}."


def extract_unknown_value(declarative_statement: str, known_answer: str) -> str:
    """
    Extract the unknown value ('X') from a declarative statement.
    This creates a verification statement like: "X = ? (where the original answer was Y)"
    """
    # Create a backward verification statement
    # Replace the known answer with 'X' to create a verification challenge
    statement_with_x = declarative_statement.replace(known_answer, "'X'")
    return f'"{statement_with_x}" What is the value of \'X\'?'


def verify_answer(question: str, candidate_answer: str, client: PoeAPIClient) -> Tuple[float, str]:
    """
    Verify a candidate answer through backward reasoning.
    
    Args:
        question: Original question
        candidate_answer: Candidate answer to verify
        client: API client
        
    Returns:
        Tuple of (verification_score, verification_reasoning)
    """
    try:
        # Convert to declarative statement
        declarative = convert_to_declarative(question, candidate_answer, client)
        
        # Create verification statement
        verification_statement = extract_unknown_value(declarative, candidate_answer)
        
        # Get verification from LLM
        messages = build_verifier_messages(verification_statement, num_examples=3)
        
        response_data = client.query_claude_sonnet(
            messages=messages,
            max_tokens=512,
            temperature=0.2
        )
        
        verification_text = response_data['content']
        
        # Extract verified answer
        verified_answers = extract_numeric_answers(verification_text)
        
        # Score: 1.0 if verified answer matches candidate, 0.0 otherwise
        if verified_answers:
            try:
                verified_num = float(verified_answers[0])
                candidate_num = float(candidate_answer)
                score = 1.0 if abs(verified_num - candidate_num) < 0.01 else 0.0
            except (ValueError, TypeError):
                score = 1.0 if verified_answers[0] == candidate_answer else 0.0
        else:
            score = 0.0
        
        return score, verification_text
        
    except Exception as e:
        print(f"Error in verification: {e}")
        return 0.0, f"Error: {str(e)}"


def run_self_verification(client: PoeAPIClient, questions: List[str], answers: List[str],
                         num_candidates: int = 5, max_problems: int = None,
                         output_file: str = "self_verification.jsonl") -> Dict:
    """
    Run self-verification method on GSM8K dataset.
    
    Args:
        client: API client
        questions: List of questions
        answers: List of ground truth answers
        num_candidates: Number of candidate solutions to generate (N)
        max_problems: Maximum problems to evaluate
        output_file: Output file path
        
    Returns:
        Dictionary with metrics
    """
    print(f"\n{'='*60}")
    print("TASK 2: SELF-VERIFICATION METHOD")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Number of candidates (N): {num_candidates}")
    print(f"  Forward temperature: 0.8 (for diversity)")
    print(f"  Verification temperature: 0.2 (for consistency)")
    
    max_problems = max_problems or len(questions)
    num_correct = 0
    total_tokens_generated = 0
    start_time = time.time()
    
    results = []
    
    with tqdm(total=max_problems, desc="Self-Verification") as pbar:
        for idx in range(min(max_problems, len(questions))):
            question = questions[idx]
            ground_truth = answers[idx]
            
            try:
                # Step 1: Forward Reasoning - Generate N candidate solutions
                candidates = []
                candidate_responses = []
                
                for _ in range(num_candidates):
                    messages = build_cot_messages(question, num_examples=5)
                    
                    response_data = client.query_claude_sonnet(
                        messages=messages,
                        max_tokens=DEFAULT_CONFIG['max_tokens'],
                        temperature=0.8  # Higher temperature for diversity
                    )
                    
                    response_text = response_data['content']
                    tokens_used = response_data['usage']['completion_tokens']
                    total_tokens_generated += tokens_used
                    
                    # Extract answer
                    predicted_answer = extract_ans_from_response(response_text)
                    
                    if predicted_answer and predicted_answer != '':
                        # Normalize to string
                        candidates.append(str(predicted_answer))
                        candidate_responses.append(response_text)
                
                # Remove duplicates while preserving order
                unique_candidates = []
                seen = set()
                for c in candidates:
                    if c not in seen:
                        unique_candidates.append(c)
                        seen.add(c)
                
                if not unique_candidates:
                    # No valid candidates
                    results.append({
                        "question_id": idx,
                        "question": question,
                        "predicted_answer": None,
                        "ground_truth": ground_truth,
                        "is_correct": False,
                        "method": "self-verification",
                        "num_candidates": num_candidates,
                        "verification_scores": {},
                        "error": "No valid candidates generated"
                    })
                    pbar.update(1)
                    continue
                
                # Step 2: Backward Verification - Verify each candidate
                verification_scores = {}
                verification_details = {}
                
                for candidate in unique_candidates:
                    score, verification_text = verify_answer(question, candidate, client)
                    verification_scores[candidate] = score
                    verification_details[candidate] = verification_text
                
                # Step 3: Selection - Choose candidate with highest verification score
                best_candidate = max(verification_scores.items(), key=lambda x: x[1])[0]
                
                # Check correctness
                try:
                    ground_truth_num = int(float(ground_truth))
                    predicted_num = int(float(best_candidate))
                    is_correct = ground_truth_num == predicted_num
                except (ValueError, TypeError):
                    is_correct = str(best_candidate).strip() == str(ground_truth).strip()
                
                if is_correct:
                    num_correct += 1
                
                results.append({
                    "question_id": idx,
                    "question": question,
                    "predicted_answer": str(best_candidate),
                    "ground_truth": ground_truth,
                    "is_correct": is_correct,
                    "method": "self-verification",
                    "num_candidates": num_candidates,
                    "candidates": unique_candidates,
                    "verification_scores": verification_scores,
                    "best_candidate": best_candidate,
                    "best_score": verification_scores[best_candidate]
                })
                
            except Exception as e:
                print(f"Error processing question {idx}: {e}")
                results.append({
                    "question_id": idx,
                    "question": question,
                    "predicted_answer": None,
                    "ground_truth": ground_truth,
                    "is_correct": False,
                    "method": "self-verification",
                    "num_candidates": num_candidates,
                    "error": str(e)
                })
            
            pbar.update(1)
    
    end_time = time.time()
    wall_clock_time = end_time - start_time
    
    # Calculate metrics
    accuracy = (num_correct / max_problems) * 100
    avg_tokens_per_problem = total_tokens_generated / max_problems if max_problems > 0 else 0
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    metrics = {
        "method": f"self-verification-{num_candidates}",
        "accuracy": accuracy,
        "num_correct": num_correct,
        "total_problems": max_problems,
        "wall_clock_time": wall_clock_time,
        "avg_tokens_per_problem": avg_tokens_per_problem,
        "total_tokens": total_tokens_generated,
        "output_file": output_file,
        "num_candidates": num_candidates
    }
    
    print(f"\nSelf-Verification Results:")
    print(f"  Method: Self-Verification with {num_candidates} candidates")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Correct: {num_correct}/{max_problems}")
    print(f"  Wall-clock time: {wall_clock_time:.2f}s")
    print(f"  Avg tokens per problem: {avg_tokens_per_problem:.1f}")
    print(f"  Total tokens: {total_tokens_generated}")
    print(f"  Results saved to: {output_file}")
    
    return metrics


def main():
    """Main function to run Task 2 self-verification"""
    
    print("\n" + "="*60)
    print("TASK 2: SELF-VERIFICATION IMPLEMENTATION FOR GSM8K")
    print("="*60)
    
    # Initialize API client
    print("\nInitializing API client...")
    client = initialize_api_client()
    
    # Load dataset
    dataset_path = "data/GSM8K/test.jsonl"
    print(f"Loading dataset from {dataset_path}...")
    questions, answers = load_gsm8k_dataset(dataset_path)
    print(f"Loaded {len(questions)} questions")
    
    # Run self-verification with different numbers of candidates
    max_test_problems = DEFAULT_CONFIG['max_problems']
    
    # Configuration: using 5 candidates (N=5)
    num_candidates = 5
    
    metrics = run_self_verification(
        client, questions, answers,
        num_candidates=num_candidates,
        max_problems=max_test_problems,
        output_file=f"self_verification_{num_candidates}shot.jsonl"
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("TASK 2 SUMMARY")
    print(f"{'='*60}")
    print(f"Self-Verification ({num_candidates} candidates) accuracy: {metrics['accuracy']:.2f}%")
    print(f"Tokens used: {metrics['total_tokens']}")
    print(f"Time taken: {metrics['wall_clock_time']:.2f}s")
    

if __name__ == "__main__":
    main()
