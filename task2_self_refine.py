"""
Task 2: Self-Refine - Iterative Refinement with Self-Feedback
Implements the Self-Refine method for GSM8K math reasoning

Author: Assignment 1
Date: 2025
"""

import json
import time
import os
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from api_client import PoeAPIClient
from config import initialize_api_client, DEFAULT_CONFIG
from data.GSM8K.evaluation import extract_ans_from_response


# System prompts for Self-Refine approach
SYSTEM_PROMPT_GENERATE = """You are an expert math problem solver. Your task is to solve math word problems step by step.

When solving:
1. Read the problem carefully
2. Identify what is being asked
3. Show your reasoning process clearly
4. Provide the final numerical answer

Always end your response with: #### [answer]
where [answer] is the numerical result."""

SYSTEM_PROMPT_FEEDBACK = """You are an expert math teacher and problem reviewer. Your task is to carefully analyze mathematical solutions and provide constructive feedback.

When reviewing a solution:
1. Check if the mathematical reasoning is correct
2. Verify that all steps logically follow
3. Identify any errors in understanding or calculation
4. Provide specific guidance on what needs to be fixed
5. Be clear and concise in your feedback

If the solution is correct, respond with: "SOLUTION_IS_CORRECT"
Otherwise, provide detailed feedback on the errors found."""


def load_gsm8k_dataset(dataset_path: str) -> Tuple[List[str], List[str]]:
    """
    Load GSM8K dataset from JSONL file
    
    Args:
        dataset_path: Path to the JSONL file
        
    Returns:
        Tuple of (questions, answers) lists
    """
    questions = []
    answers = []
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            questions.append(data['question'])
            # Extract answer after "#### "
            answer = data['answer'].split('#### ')[-1].strip()
            answers.append(answer)
    
    return questions, answers


def build_generate_messages(question: str) -> List[Dict[str, str]]:
    """
    Build message list for generating initial solution
    
    Args:
        question: The math problem question
        
    Returns:
        List of message dictionaries
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT_GENERATE},
        {"role": "user", "content": f"Question: {question}\n\nPlease solve this step by step."}
    ]


def build_feedback_messages(question: str, solution: str) -> List[Dict[str, str]]:
    """
    Build message list for generating feedback on a solution
    
    Args:
        question: The original math problem question
        solution: The proposed solution
        
    Returns:
        List of message dictionaries
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT_FEEDBACK},
        {
            "role": "user",
            "content": f"""Question: {question}

Proposed Solution:
{solution}

Please review this solution carefully and provide feedback. If the solution is correct, respond with exactly: "SOLUTION_IS_CORRECT"
Otherwise, identify the specific errors and suggest corrections."""
        }
    ]


def build_refinement_messages(question: str, solution: str, feedback: str) -> List[Dict[str, str]]:
    """
    Build message list for refining solution based on feedback
    
    Args:
        question: The original math problem question
        solution: The previous solution
        feedback: Feedback on the previous solution
        
    Returns:
        List of message dictionaries
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT_GENERATE},
        {
            "role": "user",
            "content": f"""Question: {question}

Previous Solution:
{solution}

Feedback on the previous solution:
{feedback}

Please provide an improved solution that addresses the feedback. Show your reasoning process clearly and end with: #### [answer]"""
        }
    ]


def get_initial_solution(client: PoeAPIClient, question: str) -> Tuple[str, int]:
    """
    Generate initial solution for a question
    
    Args:
        client: PoeAPIClient instance
        question: The math problem question
        
    Returns:
        Tuple of (solution_text, tokens_used)
    """
    messages = build_generate_messages(question)
    
    response_data = client.query_claude_sonnet(
        messages=messages,
        max_tokens=DEFAULT_CONFIG['max_tokens'],
        temperature=DEFAULT_CONFIG['temperature']
    )
    
    solution_text = response_data['content']
    tokens_used = response_data['usage']['completion_tokens']
    
    return solution_text, tokens_used


def get_feedback(client: PoeAPIClient, question: str, solution: str) -> Tuple[str, int]:
    """
    Get feedback on a solution
    
    Args:
        client: PoeAPIClient instance
        question: The math problem question
        solution: The proposed solution
        
    Returns:
        Tuple of (feedback_text, tokens_used)
    """
    messages = build_feedback_messages(question, solution)
    
    response_data = client.query_claude_sonnet(
        messages=messages,
        max_tokens=1000,  # Feedback doesn't need many tokens
        temperature=0.0  # Use temperature 0 for consistent feedback
    )
    
    feedback_text = response_data['content'].strip()
    tokens_used = response_data['usage']['completion_tokens']
    
    return feedback_text, tokens_used


def refine_solution(client: PoeAPIClient, question: str, solution: str, feedback: str) -> Tuple[str, int]:
    """
    Refine solution based on feedback
    
    Args:
        client: PoeAPIClient instance
        question: The math problem question
        solution: The previous solution
        feedback: Feedback on the previous solution
        
    Returns:
        Tuple of (refined_solution_text, tokens_used)
    """
    messages = build_refinement_messages(question, solution, feedback)
    
    response_data = client.query_claude_sonnet(
        messages=messages,
        max_tokens=DEFAULT_CONFIG['max_tokens'],
        temperature=DEFAULT_CONFIG['temperature']
    )
    
    refined_solution = response_data['content']
    tokens_used = response_data['usage']['completion_tokens']
    
    return refined_solution, tokens_used


def is_correct_solution(feedback_text: str) -> bool:
    """
    Check if the feedback indicates the solution is correct
    
    Args:
        feedback_text: Feedback text from the model
        
    Returns:
        True if solution is correct, False otherwise
    """
    return "SOLUTION_IS_CORRECT" in feedback_text.upper()


def run_self_refine(client: PoeAPIClient, question: str, ground_truth: str, 
                    max_iterations: int = 3) -> Dict:
    """
    Run self-refine process on a single question
    
    Args:
        client: PoeAPIClient instance
        question: The math problem question
        ground_truth: Ground truth answer
        max_iterations: Maximum number of refinement iterations
        
    Returns:
        Dictionary with refinement history and metrics
    """
    
    iterations_log = []
    total_tokens = 0
    
    # Step 1: Generate initial solution
    try:
        solution, tokens = get_initial_solution(client, question)
        total_tokens += tokens
        
        predicted_answer = extract_ans_from_response(solution)
        
        iterations_log.append({
            "iteration": 0,
            "type": "initial_generation",
            "solution": solution,
            "predicted_answer": str(predicted_answer),
            "feedback": None,
            "tokens_used": tokens
        })
    except Exception as e:
        return {
            "question": question,
            "ground_truth": ground_truth,
            "success": False,
            "error": f"Initial generation failed: {str(e)}",
            "iterations_log": iterations_log,
            "total_tokens": total_tokens,
            "final_answer": None,
            "is_correct": False,
            "num_iterations": 0
        }
    
    # Step 2: Iterative refinement loop
    current_solution = solution
    is_final_correct = False
    
    for iteration in range(1, max_iterations + 1):
        try:
            # Get feedback on current solution
            feedback, tokens = get_feedback(client, question, current_solution)
            total_tokens += tokens
            
            # Check if solution is correct
            if is_correct_solution(feedback):
                is_final_correct = True
                iterations_log.append({
                    "iteration": iteration,
                    "type": "feedback",
                    "solution": current_solution,
                    "predicted_answer": str(extract_ans_from_response(current_solution)),
                    "feedback": feedback,
                    "tokens_used": tokens,
                    "is_correct": True
                })
                break
            
            iterations_log.append({
                "iteration": iteration,
                "type": "feedback",
                "solution": current_solution,
                "predicted_answer": str(extract_ans_from_response(current_solution)),
                "feedback": feedback,
                "tokens_used": tokens,
                "is_correct": False
            })
            
            # Refine solution based on feedback
            refined_solution, tokens = refine_solution(client, question, current_solution, feedback)
            total_tokens += tokens
            
            iterations_log.append({
                "iteration": iteration,
                "type": "refinement",
                "solution": refined_solution,
                "predicted_answer": str(extract_ans_from_response(refined_solution)),
                "feedback": f"Refined based on: {feedback[:100]}...",
                "tokens_used": tokens
            })
            
            current_solution = refined_solution
            
        except Exception as e:
            print(f"Error during refinement iteration {iteration}: {e}")
            break
    
    # Extract final answer
    final_answer = extract_ans_from_response(current_solution)
    
    # Check correctness
    try:
        ground_truth_num = int(float(ground_truth))
        predicted_num = int(float(str(final_answer)))
        is_correct = ground_truth_num == predicted_num
    except (ValueError, TypeError):
        is_correct = str(final_answer).strip() == str(ground_truth).strip()
    
    return {
        "question": question,
        "ground_truth": ground_truth,
        "success": True,
        "final_solution": current_solution,
        "final_answer": str(final_answer),
        "is_correct": is_correct,
        "is_final_correct_per_feedback": is_final_correct,
        "iterations_log": iterations_log,
        "total_tokens": total_tokens,
        "num_iterations": len([log for log in iterations_log if log["type"] == "refinement"]) + 1
    }


def run_self_refine_baseline(client: PoeAPIClient, questions: List[str], answers: List[str],
                             max_iterations: int = 3, max_problems: int = None,
                             output_file: str = "self_refine.jsonl") -> Dict:
    """
    Run self-refine baseline on GSM8K
    
    Args:
        client: PoeAPIClient instance
        questions: List of questions
        answers: List of ground truth answers
        max_iterations: Maximum refinement iterations per question
        max_problems: Maximum number of problems to evaluate (None for all)
        output_file: Output JSONL file path
        
    Returns:
        Dictionary with accuracy and timing metrics
    """
    print(f"\n{'='*60}")
    print(f"TASK 2: SELF-REFINE METHOD (max {max_iterations} iterations)")
    print(f"{'='*60}")
    
    max_problems = max_problems or len(questions)
    num_correct = 0
    num_correct_after_feedback = 0
    total_tokens_generated = 0
    total_refinements = 0
    start_time = time.time()
    
    results = []
    
    with tqdm(total=max_problems, desc="Self-Refine") as pbar:
        for idx in range(min(max_problems, len(questions))):
            question = questions[idx]
            ground_truth = answers[idx]
            
            try:
                result = run_self_refine(
                    client, question, ground_truth,
                    max_iterations=max_iterations
                )
                
                total_tokens_generated += result['total_tokens']
                total_refinements += result['num_iterations']
                
                if result['is_correct']:
                    num_correct += 1
                
                if result['is_final_correct_per_feedback']:
                    num_correct_after_feedback += 1
                
                results.append({
                    "question_id": idx,
                    "question": question,
                    "predicted_answer": result['final_answer'],
                    "ground_truth": ground_truth,
                    "is_correct": result['is_correct'],
                    "is_correct_per_feedback": result['is_final_correct_per_feedback'],
                    "final_solution": result['final_solution'],
                    "iterations_log": result['iterations_log'],
                    "total_tokens": result['total_tokens'],
                    "num_iterations": result['num_iterations']
                })
                
            except Exception as e:
                print(f"Error processing question {idx}: {e}")
                results.append({
                    "question_id": idx,
                    "question": question,
                    "predicted_answer": None,
                    "ground_truth": ground_truth,
                    "is_correct": False,
                    "error": str(e)
                })
            
            pbar.update(1)
    
    end_time = time.time()
    wall_clock_time = end_time - start_time
    
    # Calculate metrics
    accuracy = (num_correct / max_problems) * 100
    accuracy_after_feedback = (num_correct_after_feedback / max_problems) * 100
    avg_tokens_per_problem = total_tokens_generated / max_problems if max_problems > 0 else 0
    avg_iterations = total_refinements / max_problems if max_problems > 0 else 0
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    metrics = {
        "method": f"self-refine-{max_iterations}",
        "accuracy": accuracy,
        "accuracy_after_feedback": accuracy_after_feedback,
        "num_correct": num_correct,
        "num_correct_after_feedback": num_correct_after_feedback,
        "total_problems": max_problems,
        "wall_clock_time": wall_clock_time,
        "avg_tokens_per_problem": avg_tokens_per_problem,
        "avg_iterations": avg_iterations,
        "total_tokens": total_tokens_generated,
        "output_file": output_file
    }
    
    print(f"\nSelf-Refine Results (max {max_iterations} iterations):")
    print(f"  Final Accuracy: {accuracy:.2f}%")
    print(f"  Correct (final): {num_correct}/{max_problems}")
    print(f"  Correct (per feedback): {num_correct_after_feedback}/{max_problems}")
    print(f"  Accuracy after feedback: {accuracy_after_feedback:.2f}%")
    print(f"  Wall-clock time: {wall_clock_time:.2f}s")
    print(f"  Avg tokens per problem: {avg_tokens_per_problem:.1f}")
    print(f"  Avg iterations: {avg_iterations:.2f}")
    print(f"  Total tokens: {total_tokens_generated}")
    print(f"  Results saved to: {output_file}")
    
    return metrics


def main():
    """Main function to run Task 2 self-refine baseline"""
    
    print("\n" + "="*60)
    print("TASK 2: SELF-REFINE METHOD FOR GSM8K")
    print("="*60)
    
    # Initialize API client
    print("\nInitializing API client...")
    client = initialize_api_client()
    
    # Load dataset
    dataset_path = "data/GSM8K/test.jsonl"
    print(f"Loading dataset from {dataset_path}...")
    questions, answers = load_gsm8k_dataset(dataset_path)
    print(f"Loaded {len(questions)} questions")
    
    # Run self-refine with different iteration limits
    max_test_problems = DEFAULT_CONFIG['max_problems']
    
    # Test with 2 iterations
    self_refine_2_metrics = run_self_refine_baseline(
        client, questions, answers,
        max_iterations=2,
        max_problems=max_test_problems,
        output_file="self_refine_2iter.jsonl"
    )
    
    # Test with 3 iterations
    self_refine_3_metrics = run_self_refine_baseline(
        client, questions, answers,
        max_iterations=3,
        max_problems=max_test_problems,
        output_file="self_refine_3iter.jsonl"
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("TASK 2 SELF-REFINE SUMMARY")
    print(f"{'='*60}")
    print(f"\nWith 2 iterations:")
    print(f"  Final Accuracy: {self_refine_2_metrics['accuracy']:.2f}%")
    print(f"  Accuracy after feedback: {self_refine_2_metrics['accuracy_after_feedback']:.2f}%")
    print(f"\nWith 3 iterations:")
    print(f"  Final Accuracy: {self_refine_3_metrics['accuracy']:.2f}%")
    print(f"  Accuracy after feedback: {self_refine_3_metrics['accuracy_after_feedback']:.2f}%")
    

if __name__ == "__main__":
    main()
