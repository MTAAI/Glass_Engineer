"""
Quick test to verify paths are working
"""
from pathlib import Path

print("=" * 60)
print("PATH DETECTION TEST")
print("=" * 60)

# Test 1: Current directory
print(f"\n1. Current directory: {Path.cwd()}")

# Test 2: Relative path
base_dir = Path('data')
print(f"\n2. Base dir (relative): {base_dir}")
print(f"   Absolute: {base_dir.absolute()}")
print(f"   Exists: {base_dir.exists()}")

# Test 3: Papers directory
papers_dir = base_dir / 'papers' / 'cutting_edge'
print(f"\n3. Papers dir: {papers_dir}")
print(f"   Absolute: {papers_dir.absolute()}")
print(f"   Exists: {papers_dir.exists()}")

# Test 4: Count PDFs
if papers_dir.exists():
    pdfs = list(papers_dir.glob('*.pdf'))
    print(f"   PDF count: {len(pdfs)}")
    if pdfs:
        print(f"   First 3 PDFs:")
        for pdf in pdfs[:3]:
            print(f"     - {pdf.name}")
else:
    print("   ❌ Directory doesn't exist!")

# Test 5: Check what's in data folder
print(f"\n4. Contents of data folder:")
if base_dir.exists():
    for item in base_dir.iterdir():
        if item.is_dir():
            print(f"   📁 {item.name}")
        else:
            print(f"   📄 {item.name}")
else:
    print("   ❌ data folder doesn't exist!")

print("\n" + "=" * 60)
