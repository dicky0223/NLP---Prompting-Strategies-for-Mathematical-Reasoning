"""
Configuration and utility module for experiment runner
Centralizes API key, constants, and common functions
"""

import os
from api_client import PoeAPIClient

# API Configuration
API_KEY = "mv5qhOP2NDYOKPDs2mliZkLFzISYYELqgKw9TTy1c5I"

# Default experiment configuration
DEFAULT_CONFIG = {
    "max_problems": 200,  # Change to 1319 for full test set evaluation
    "temperature": 0.0,
    "max_tokens": 2048,
    "model": "Claude-Sonnet-4.5",
    "api_retry_attempts": 3,
    "rate_limit_delay": 0.5,  # Seconds between API calls
}


def initialize_api_client(api_key: str = None) -> PoeAPIClient:
    """
    Initialize and return API client
    
    Args:
        api_key: Optional API key. Uses default if not provided.
        
    Returns:
        Initialized PoeAPIClient instance
    """
    key = api_key or API_KEY
    return PoeAPIClient(key)


def test_api_connection(client: PoeAPIClient) -> bool:
    """
    Test API connection with a simple query
    
    Args:
        client: PoeAPIClient instance
        
    Returns:
        True if connection successful, False otherwise
    """
    try:
        test_messages = [
            {"role": "user", "content": "Hello! Please respond with just 'API working' if you receive this."}
        ]
        response = client.query_with_retry(test_messages)
        if response.get('success') and response.get('content'):
            print("✓ API connection successful")
            return True
        else:
            print("✗ API connection failed")
            return False
    except Exception as e:
        print(f"✗ API connection test failed: {e}")
        return False


def print_section(title: str, width: int = 70):
    """
    Print a formatted section header
    
    Args:
        title: Section title
        width: Total width of the section
    """
    print("\n" + "="*width)
    print(title.center(width))
    print("="*width)


def print_subsection(title: str, width: int = 70):
    """
    Print a formatted subsection header
    
    Args:
        title: Subsection title
        width: Total width of the subsection
    """
    print("\n" + title.center(width))
    print("-" * width)
