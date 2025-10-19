"""
API client for Poe's Claude-4.5-Sonnet
"""

import openai
from typing import List, Dict


class PoeAPIClient:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.poe.com/v1"
        )

    def query_claude_sonnet(self, messages: List[Dict[str, str]], max_tokens: int = 2048, temperature: float = 0.0, model: str = "Claude-Sonnet-4.5") -> dict:
        """
        Query Claude Sonnet and return response with usage information.
        
        Returns:
            dict with keys:
                - 'content': The response text
                - 'usage': Dict with 'prompt_tokens', 'completion_tokens', 'total_tokens'
                - 'has_actual_usage': Boolean indicating if usage data came from API
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream_options={"include_usage": True}  # Enable usage tracking
            )
            
            # Extract response content
            content = response.choices[0].message.content
            
            # Try to get actual usage from response
            usage_info = {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0,
                'has_actual_usage': False
            }
            
            if hasattr(response, 'usage') and response.usage:
                usage_info['prompt_tokens'] = response.usage.prompt_tokens
                usage_info['completion_tokens'] = response.usage.completion_tokens
                usage_info['total_tokens'] = response.usage.total_tokens
                usage_info['has_actual_usage'] = True
            
            return {
                'content': content,
                'usage': usage_info
            }
        except Exception as e:
            print(f"Error in query_claude_sonnet: {e}")
            raise

    def query_with_retry(self, messages: List[Dict[str, str]], max_retries: int = 3, delay: float = 1.0, **kwargs) -> dict:
        """
        Retry wrapper for query_claude_sonnet with exponential backoff.
        
        Returns:
            dict with keys:
                - 'content': The response text (or empty string if failed)
                - 'usage': Dict with token usage information
                - 'success': Boolean indicating if query succeeded
        """
        import time
        
        for attempt in range(max_retries + 1):
            try:
                response = self.query_claude_sonnet(messages, **kwargs)
                if response['content']:
                    response['success'] = True
                    return response
                if attempt < max_retries:
                    print(f"Empty response, retrying in {delay} seconds... (attempt {attempt + 1})")
                    time.sleep(delay)
                    delay *= 2
            except Exception as e:
                if attempt < max_retries:
                    print(f"Error: {e}. Retrying in {delay} seconds... (attempt {attempt + 1})")
                    time.sleep(delay)
                    delay *= 2
                else:
                    print(f"Failed after {max_retries + 1} attempts: {e}")
        
        return {
            'content': "",
            'usage': {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0,
                'has_actual_usage': False
            },
            'success': False
        }

# Example usage and testing
if __name__ == "__main__":
    # Test the API client
    client = PoeAPIClient("mv5qhOP2NDYOKPDs2mliZkLFzISYYELqgKw9TTy1c5I")
    
    # Test with a simple math problem
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant that solves math problems step by step."},
        {"role": "user", "content": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"}
    ]
    
    print("Testing API connection with usage tracking...")
    response = client.query_with_retry(test_messages)
    print(f"\nResponse: {response['content'][:200]}...")
    print(f"\nUsage Information:")
    print(f"  Prompt Tokens: {response['usage']['prompt_tokens']}")
    print(f"  Completion Tokens: {response['usage']['completion_tokens']}")
    print(f"  Total Tokens: {response['usage']['total_tokens']}")
    print(f"  Has Actual Usage Data: {response['usage']['has_actual_usage']}")
    print(f"  Request Successful: {response.get('success', False)}")