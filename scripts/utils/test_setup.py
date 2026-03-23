#!/usr/bin/env python3
"""
Test setup script - Verify environment is ready for Q&A generation
"""

import sys
import os
from pathlib import Path

def test_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    return True

def test_packages():
    """Check required packages"""
    required = [
        'openai',
        'PyPDF2',
        'pdfplumber',
        'pandas',
        'numpy',
        'dotenv',
        'tqdm'
    ]
    
    missing = []
    for package in required:
        try:
            if package == 'dotenv':
                __import__('dotenv')
            else:
                __import__(package.lower())
            print(f"✅ {package} installed")
        except ImportError:
            print(f"❌ {package} NOT installed")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    return True

def test_api_key():
    """Check if OpenAI API key is configured"""
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key and api_key.startswith('sk-'):
        print("✅ OpenAI API key found")
        return True
    else:
        print("❌ OpenAI API key not configured")
        print("Edit .env file and add your API key")
        return False

def test_directories():
    """Check if data directories exist"""
    base_dir = Path('data')
    
    required_dirs = [
        'textbooks/technical',
        'textbooks/art',
        'textbooks/ancient',
        'papers/research',
        'papers/cutting_edge',
        'csv',
        'qa_pairs/raw',
        'qa_pairs/augmented',
        'qa_pairs/final'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        full_path = base_dir / dir_path
        if full_path.exists():
            print(f"✅ {dir_path}/ exists")
        else:
            print(f"⚠️  {dir_path}/ does not exist (will be created)")
            full_path.mkdir(parents=True, exist_ok=True)
            all_exist = False
    
    return True  # Always return True since we create missing dirs

def count_data_files():
    """Count data files"""
    base_dir = Path('data')
    
    counts = {
        'Technical textbooks': len(list((base_dir / 'textbooks/technical').glob('*.pdf'))) if (base_dir / 'textbooks/technical').exists() else 0,
        'Art glass books': len(list((base_dir / 'textbooks/art').glob('*.pdf'))) if (base_dir / 'textbooks/art').exists() else 0,
        'Ancient glass books': len(list((base_dir / 'textbooks/ancient').glob('*.pdf'))) if (base_dir / 'textbooks/ancient').exists() else 0,
        'Research papers': len(list((base_dir / 'papers/research').glob('*.pdf'))) if (base_dir / 'papers/research').exists() else 0,
        'Cutting-edge papers': len(list((base_dir / 'papers/cutting_edge').glob('*.pdf'))) if (base_dir / 'papers/cutting_edge').exists() else 0,
    }
    
    total = sum(counts.values())
    
    for name, count in counts.items():
        if count > 0:
            print(f"✅ Found {count} {name}")
        else:
            print(f"⚠️  No {name} found")
    
    print(f"\n✅ Total: {total} documents ready")
    
    if total == 0:
        print("\n⚠️  No data files found!")
        print("Copy your PDFs to the data/ directories")
        return False
    
    return True

def main():
    """Run all tests"""
    print("=" * 70)
    print("Glass Expert AI - Setup Test")
    print("=" * 70)
    print()
    
    tests = [
        ("Python version", test_python_version),
        ("Required packages", test_packages),
        ("OpenAI API key", test_api_key),
        ("Data directories", test_directories),
        ("Data files", count_data_files),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\nTesting {name}...")
        print("-" * 70)
        result = test_func()
        results.append(result)
        print()
    
    print("=" * 70)
    if all(results):
        print("🎉 Setup complete! Ready to generate Q&A pairs!")
        print()
        print("Next step:")
        print("  python scripts/qa_generation/generate_qa_master.py")
    else:
        print("⚠️  Setup incomplete. Fix the issues above and try again.")
    print("=" * 70)

if __name__ == '__main__':
    main()
