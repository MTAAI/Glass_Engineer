#!/usr/bin/env python3
"""
Test OpenAI API connection and estimate costs
"""

import os
import sys
from dotenv import load_dotenv

def test_api_connection():
    """Test OpenAI API connection"""
    try:
        from openai import OpenAI
        
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            print("❌ OPENAI_API_KEY not found in .env file")
            print("\nPlease:")
            print("1. Copy .env.example to .env")
            print("2. Add your OpenAI API key to .env")
            return False
        
        print("Testing OpenAI API connection...")
        print(f"API Key: {api_key[:20]}...{api_key[-4:]}")
        print()
        
        # Initialize client
        client = OpenAI(api_key=api_key)
        
        # Test with a simple completion
        print("Sending test request...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'API test successful' if you can read this."}
            ],
            max_tokens=20,
            temperature=0
        )
        
        result = response.choices[0].message.content
        print(f"✅ Response: {result}")
        print()
        
        # Show usage
        usage = response.usage
        print(f"✅ Tokens used: {usage.total_tokens}")
        print(f"   - Prompt: {usage.prompt_tokens}")
        print(f"   - Completion: {usage.completion_tokens}")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {str(e)}")
        print()
        print("Common issues:")
        print("1. Invalid API key - Check your key at platform.openai.com")
        print("2. No credits - Add payment method at platform.openai.com/billing")
        print("3. Network issue - Check your internet connection")
        return False

def estimate_costs():
    """Estimate costs for Q&A generation"""
    print("=" * 70)
    print("Cost Estimation for Q&A Generation")
    print("=" * 70)
    print()
    
    # Pricing (as of 2024)
    gpt4_mini_input = 0.150 / 1_000_000  # $0.150 per 1M input tokens
    gpt4_mini_output = 0.600 / 1_000_000  # $0.600 per 1M output tokens
    
    # Estimates
    documents = 1066
    avg_pages_per_doc = 50
    tokens_per_page = 500
    qa_pairs_per_doc = 3
    tokens_per_qa = 600  # 100 question + 500 answer
    
    # Calculate
    total_input_tokens = documents * avg_pages_per_doc * tokens_per_page
    total_qa_pairs = documents * qa_pairs_per_doc
    total_output_tokens = total_qa_pairs * tokens_per_qa
    
    input_cost = total_input_tokens * gpt4_mini_input
    output_cost = total_output_tokens * gpt4_mini_output
    total_cost = input_cost + output_cost
    
    print(f"Documents: {documents:,}")
    print(f"Estimated pages: {documents * avg_pages_per_doc:,}")
    print(f"Estimated Q&A pairs: {total_qa_pairs:,}")
    print()
    print(f"Input tokens: {total_input_tokens:,}")
    print(f"Output tokens: {total_output_tokens:,}")
    print()
    print(f"Input cost: ${input_cost:.2f}")
    print(f"Output cost: ${output_cost:.2f}")
    print(f"Total cost: ${total_cost:.2f}")
    print()
    
    # With augmentation
    augmentation_cost = total_cost * 1.5  # 1.5x for augmentation
    final_cost = total_cost + augmentation_cost
    
    print("With synthetic augmentation (2x expansion):")
    print(f"Generation: ${total_cost:.2f}")
    print(f"Augmentation: ${augmentation_cost:.2f}")
    print(f"Total: ${final_cost:.2f}")
    print()
    print(f"Final Q&A pairs: {total_qa_pairs * 2:,}")
    print(f"Cost per Q&A pair: ${final_cost / (total_qa_pairs * 2):.4f}")
    print()
    
    print("=" * 70)
    print("Note: Actual costs may vary based on:")
    print("- Actual document lengths")
    print("- Answer complexity")
    print("- API pricing changes")
    print("=" * 70)

def main():
    """Run tests"""
    print("=" * 70)
    print("OpenAI API Test")
    print("=" * 70)
    print()
    
    # Test connection
    success = test_api_connection()
    print()
    
    if success:
        # Estimate costs
        estimate_costs()
        print()
        print("✅ Ready to start Q&A generation!")
        print()
        print("Next step:")
        print("  python scripts/qa_generation/generate_qa_master.py")
    else:
        print("❌ Fix API issues before proceeding")
    
    print("=" * 70)

if __name__ == '__main__':
    main()
