#!/usr/bin/env python3
"""
Deployment Check Script for Coherent Multiplex
This script helps verify that all required environment variables are set correctly
"""

import os
import sys

def check_environment():
    """Check if all required environment variables are set"""
    print("🔍 Checking deployment environment...")
    print("=" * 50)
    
    # Check Claude API key
    claude_key = os.getenv('CLAUDE_API_KEY')
    if claude_key:
        # Don't print the actual key for security
        key_preview = f"{claude_key[:8]}...{claude_key[-4:]}" if len(claude_key) > 12 else "***"
        print(f"✅ CLAUDE_API_KEY: Found ({key_preview})")
    else:
        print("❌ CLAUDE_API_KEY: Not found!")
        return False
    
    # Check other environment variables
    debug = os.getenv('DEBUG', 'False')
    print(f"ℹ️  DEBUG: {debug}")
    
    # Check Python version
    print(f"🐍 Python version: {sys.version}")
    
    # Try importing required packages
    try:
        import anthropic
        print("✅ anthropic package: Available")
    except ImportError:
        print("❌ anthropic package: Not found!")
        return False
    
    try:
        import fastapi
        print("✅ fastapi package: Available")
    except ImportError:
        print("❌ fastapi package: Not found!")
        return False
    
    try:
        import uvicorn
        print("✅ uvicorn package: Available")
    except ImportError:
        print("❌ uvicorn package: Not found!")
        return False
    
    print("=" * 50)
    print("✅ All checks passed! Ready for deployment.")
    return True

def test_claude_connection():
    """Test Claude API connection"""
    try:
        import anthropic
        
        claude_key = os.getenv('CLAUDE_API_KEY')
        if not claude_key:
            print("❌ Cannot test Claude connection: API key not found")
            return False
        
        client = anthropic.Anthropic(api_key=claude_key)
        
        # Simple test message
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": "Say 'test'"}]
        )
        
        print("✅ Claude API connection: Working")
        return True
        
    except Exception as e:
        print(f"❌ Claude API connection failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Coherent Multiplex Deployment Check")
    print()
    
    env_ok = check_environment()
    
    if env_ok:
        print("\n🔌 Testing Claude API connection...")
        claude_ok = test_claude_connection()
        
        if claude_ok:
            print("\n🎉 All systems ready for deployment!")
            sys.exit(0)
        else:
            print("\n⚠️  Environment OK but Claude API connection failed")
            sys.exit(1)
    else:
        print("\n❌ Environment check failed")
        print("\nTo fix:")
        print("1. Set CLAUDE_API_KEY environment variable")
        print("2. Make sure all required packages are installed")
        sys.exit(1)
