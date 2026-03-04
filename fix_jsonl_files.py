"""
Fix Malformed JSONL Files - Version 2
Handles files with multiple JSON objects concatenated on single lines
"""

import json
import re
from pathlib import Path
from datetime import datetime

# Configuration
QA_DIR = Path('data/qa_pairs/raw')
BACKUP_DIR = Path('data/qa_pairs/backup')


def extract_json_objects(content: str) -> list:
    """Extract all valid JSON objects from concatenated content"""
    
    objects = []
    
    # Strategy: Find all complete {...} objects using a stack-based approach
    depth = 0
    start_idx = None
    in_string = False
    escape_next = False
    
    for i, char in enumerate(content):
        # Handle string literals (to ignore { } inside strings)
        if char == '\\' and not escape_next:
            escape_next = True
            continue
        
        if char == '"' and not escape_next:
            in_string = not in_string
        
        escape_next = False
        
        # Only count braces outside of strings
        if not in_string:
            if char == '{':
                if depth == 0:
                    start_idx = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and start_idx is not None:
                    # Found a complete object
                    json_str = content[start_idx:i+1]
                    try:
                        obj = json.loads(json_str)
                        objects.append(obj)
                    except json.JSONDecodeError as e:
                        print(f"      ⚠️ Skipped malformed object at position {start_idx}: {str(e)[:50]}")
                    start_idx = None
    
    return objects


def fix_jsonl_file(input_file: Path) -> tuple:
    """Fix malformed JSONL file by extracting all JSON objects"""
    
    print(f"\n🔧 Processing: {input_file.name}")
    
    # Read entire file content
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"   ✗ Error reading file: {e}")
        return [], 0
    
    print(f"   📏 File size: {len(content):,} characters")
    
    # Extract all JSON objects
    print(f"   🔍 Extracting JSON objects...")
    objects = extract_json_objects(content)
    
    print(f"   ✓ Extracted {len(objects)} valid Q&A pairs")
    
    return objects, len(objects)


def main():
    """Fix all JSONL files"""
    
    print("=" * 70)
    print("🔧 JSONL FILE REPAIR TOOL - VERSION 2")
    print("=" * 70)
    print(f"\nInput directory: {QA_DIR}")
    print(f"Backup directory: {BACKUP_DIR}")
    print("\nThis version handles concatenated JSON objects on single lines.")
    
    # Create backup directory
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all JSONL files
    jsonl_files = list(QA_DIR.glob('*.jsonl'))
    
    if not jsonl_files:
        print("\n❌ No JSONL files found!")
        return
    
    print(f"\nFound {len(jsonl_files)} JSONL files to process")
    
    total_fixed = 0
    file_results = {}
    
    # Process each file
    for file_path in jsonl_files:
        # Backup original
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = BACKUP_DIR / f"{file_path.stem}_backup_{timestamp}.jsonl"
        
        try:
            # Read and backup original
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original_content)
            
            print(f"   📦 Backup: {backup_path.name}")
            
            # Extract all objects
            objects, count = fix_jsonl_file(file_path)
            
            if objects:
                # Write fixed file (one object per line)
                with open(file_path, 'w', encoding='utf-8') as f:
                    for obj in objects:
                        f.write(json.dumps(obj, ensure_ascii=False) + '\n')
                
                print(f"   💾 Saved {count} pairs to {file_path.name}")
                total_fixed += count
                file_results[file_path.stem] = count
            else:
                print(f"   ❌ No valid objects found")
                file_results[file_path.stem] = 0
        
        except Exception as e:
            print(f"   ✗ Error processing {file_path.name}: {e}")
            file_results[file_path.stem] = 0
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ REPAIR COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"   Files processed: {len(jsonl_files)}")
    print(f"   Total Q&A pairs recovered: {total_fixed:,}")
    print(f"\n📁 Breakdown by file:")
    for filename, count in file_results.items():
        print(f"   {filename:30s}: {count:,} pairs")
    
    print(f"\n📦 Backups saved to: {BACKUP_DIR}")
    
    if total_fixed > 0:
        print("\n✅ SUCCESS! Files have been repaired.")
        print("\n💡 Next steps:")
        print("   1. Run quality review: python review_qa_quality.py")
        print("   2. Verify the Q&A pairs look correct")
        print("   3. If satisfied, you can delete backup files")
    else:
        print("\n❌ No Q&A pairs were recovered. Please check:")
        print("   1. Are the files in the correct directory?")
        print("   2. Do the files contain valid JSON data?")
        print("   3. Check the backup files to see original content")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
