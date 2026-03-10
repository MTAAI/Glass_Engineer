"""
Q&A Quality Review Script
Analyze generated Q&A pairs for quality, coverage, and issues
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
import re

# Configuration
QA_DIR = Path('data/qa_pairs/raw')
OUTPUT_REPORT = Path('data/qa_pairs/metadata/quality_report.json')


def load_qa_pairs(file_path: Path) -> list:
    """Load Q&A pairs from JSONL file"""
    pairs = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    pairs.append(json.loads(line))
    except Exception as e:
        print(f"✗ Error loading {file_path.name}: {e}")
    return pairs


def analyze_quality(pairs: list) -> dict:
    """Analyze quality metrics"""
    
    metrics = {
        'total_pairs': len(pairs),
        'avg_question_length': 0,
        'avg_answer_length': 0,
        'questions_too_short': 0,
        'answers_too_short': 0,
        'missing_keywords': 0,
        'missing_difficulty': 0,
        'difficulty_distribution': Counter(),
        'category_distribution': Counter(),
        'keyword_frequency': Counter(),
        'issues': []
    }
    
    question_lengths = []
    answer_lengths = []
    
    for i, pair in enumerate(pairs):
        q = pair.get('question', '')
        a = pair.get('answer', '')
        
        # Length analysis
        q_len = len(q.split())
        a_len = len(a.split())
        question_lengths.append(q_len)
        answer_lengths.append(a_len)
        
        # Quality checks
        if q_len < 5:
            metrics['questions_too_short'] += 1
            metrics['issues'].append({
                'id': pair.get('id', f'pair_{i}'),
                'issue': 'Question too short',
                'question': q[:100]
            })
        
        if a_len < 50:
            metrics['answers_too_short'] += 1
            metrics['issues'].append({
                'id': pair.get('id', f'pair_{i}'),
                'issue': 'Answer too short (< 50 words)',
                'answer_length': a_len
            })
        
        # Metadata checks
        if not pair.get('keywords'):
            metrics['missing_keywords'] += 1
        else:
            for keyword in pair.get('keywords', []):
                metrics['keyword_frequency'][keyword.lower()] += 1
        
        if not pair.get('difficulty'):
            metrics['missing_difficulty'] += 1
        else:
            metrics['difficulty_distribution'][pair['difficulty']] += 1
        
        if pair.get('category'):
            metrics['category_distribution'][pair['category']] += 1
    
    # Calculate averages
    if question_lengths:
        metrics['avg_question_length'] = sum(question_lengths) / len(question_lengths)
    if answer_lengths:
        metrics['avg_answer_length'] = sum(answer_lengths) / len(answer_lengths)
    
    # Convert Counters to dicts for JSON serialization
    metrics['difficulty_distribution'] = dict(metrics['difficulty_distribution'])
    metrics['category_distribution'] = dict(metrics['category_distribution'])
    metrics['keyword_frequency'] = dict(metrics['keyword_frequency'].most_common(50))
    
    return metrics


def analyze_coverage(pairs: list) -> dict:
    """Analyze topic coverage"""
    
    # Glass science topics to check
    topics = {
        'structure': ['structure', 'network', 'bonding', 'atomic'],
        'composition': ['composition', 'formula', 'oxide', 'silica', 'soda'],
        'properties': ['property', 'properties', 'viscosity', 'density', 'hardness'],
        'thermal': ['temperature', 'thermal', 'annealing', 'tempering', 'tg'],
        'optical': ['optical', 'refractive', 'transparency', 'light'],
        'mechanical': ['strength', 'fracture', 'stress', 'toughness'],
        'chemical': ['chemical', 'durability', 'corrosion', 'weathering'],
        'manufacturing': ['manufacturing', 'melting', 'forming', 'processing'],
        'types': ['soda-lime', 'borosilicate', 'lead', 'fused silica'],
        'applications': ['application', 'window', 'container', 'fiber', 'display']
    }
    
    coverage = {}
    for topic, keywords in topics.items():
        count = 0
        for pair in pairs:
            text = (pair.get('question', '') + ' ' + pair.get('answer', '')).lower()
            if any(kw in text for kw in keywords):
                count += 1
        coverage[topic] = {
            'count': count,
            'percentage': (count / len(pairs) * 100) if pairs else 0
        }
    
    return coverage


def detect_duplicates(pairs: list) -> list:
    """Detect potential duplicate questions"""
    
    duplicates = []
    questions = {}
    
    for i, pair in enumerate(pairs):
        q = pair.get('question', '').lower().strip()
        # Normalize question
        q_normalized = re.sub(r'[^\w\s]', '', q)
        
        if q_normalized in questions:
            duplicates.append({
                'question': pair.get('question'),
                'first_id': questions[q_normalized],
                'duplicate_id': pair.get('id', f'pair_{i}')
            })
        else:
            questions[q_normalized] = pair.get('id', f'pair_{i}')
    
    return duplicates


def main():
    """Run quality review"""
    
    print("=" * 70)
    print("🔍 Q&A QUALITY REVIEW")
    print("=" * 70)
    
    # Load all Q&A files
    all_pairs = []
    file_metrics = {}
    
    for file_path in QA_DIR.glob('*.jsonl'):
        print(f"\n📄 Loading: {file_path.name}")
        pairs = load_qa_pairs(file_path)
        print(f"   Loaded: {len(pairs)} pairs")
        
        all_pairs.extend(pairs)
        file_metrics[file_path.stem] = len(pairs)
    
    print(f"\n📊 Total Q&A pairs: {len(all_pairs)}")
    
    # Analyze quality
    print("\n" + "=" * 70)
    print("📈 QUALITY ANALYSIS")
    print("=" * 70)
    
    quality = analyze_quality(all_pairs)
    
    print(f"\n✅ Overall Statistics:")
    print(f"   Total pairs: {quality['total_pairs']}")
    print(f"   Avg question length: {quality['avg_question_length']:.1f} words")
    print(f"   Avg answer length: {quality['avg_answer_length']:.1f} words")
    
    print(f"\n⚠️ Quality Issues:")
    print(f"   Questions too short: {quality['questions_too_short']}")
    print(f"   Answers too short: {quality['answers_too_short']}")
    print(f"   Missing keywords: {quality['missing_keywords']}")
    print(f"   Missing difficulty: {quality['missing_difficulty']}")
    
    print(f"\n📊 Difficulty Distribution:")
    for diff, count in quality['difficulty_distribution'].items():
        pct = (count / quality['total_pairs'] * 100)
        print(f"   {diff}: {count} ({pct:.1f}%)")
    
    if quality['category_distribution']:
        print(f"\n📂 Category Distribution:")
        for cat, count in sorted(quality['category_distribution'].items(), key=lambda x: x[1], reverse=True)[:10]:
            pct = (count / quality['total_pairs'] * 100)
            print(f"   {cat}: {count} ({pct:.1f}%)")
    
    print(f"\n🏷️ Top Keywords:")
    for keyword, count in list(quality['keyword_frequency'].items())[:15]:
        print(f"   {keyword}: {count}")
    
    # Analyze coverage
    print("\n" + "=" * 70)
    print("🎯 TOPIC COVERAGE ANALYSIS")
    print("=" * 70)
    
    coverage = analyze_coverage(all_pairs)
    
    print("\nTopic coverage:")
    for topic, data in sorted(coverage.items(), key=lambda x: x[1]['percentage'], reverse=True):
        print(f"   {topic:20s}: {data['count']:4d} pairs ({data['percentage']:5.1f}%)")
    
    # Detect duplicates
    print("\n" + "=" * 70)
    print("🔍 DUPLICATE DETECTION")
    print("=" * 70)
    
    duplicates = detect_duplicates(all_pairs)
    print(f"\nFound {len(duplicates)} potential duplicates")
    
    if duplicates:
        print("\nSample duplicates:")
        for dup in duplicates[:5]:
            print(f"   - {dup['question'][:80]}...")
            print(f"     IDs: {dup['first_id']} & {dup['duplicate_id']}")
    
    # Generate report
    report = {
        'generated_at': Path(OUTPUT_REPORT).parent.parent.name,
        'total_pairs': len(all_pairs),
        'files': file_metrics,
        'quality_metrics': quality,
        'coverage': coverage,
        'duplicates_found': len(duplicates),
        'duplicate_examples': duplicates[:20]
    }
    
    # Save report
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Quality report saved to: {OUTPUT_REPORT}")
    
    # Show critical issues
    if quality['issues']:
        print("\n" + "=" * 70)
        print("⚠️ CRITICAL ISSUES (First 10)")
        print("=" * 70)
        
        for issue in quality['issues'][:10]:
            print(f"\n❌ {issue['issue']}")
            print(f"   ID: {issue['id']}")
            if 'question' in issue:
                print(f"   Q: {issue['question']}")
            if 'answer_length' in issue:
                print(f"   Answer length: {issue['answer_length']} words")
    
    # Overall assessment
    print("\n" + "=" * 70)
    print("🎯 OVERALL ASSESSMENT")
    print("=" * 70)
    
    score = 100
    issues_found = []
    
    if quality['avg_answer_length'] < 100:
        score -= 20
        issues_found.append("⚠️ Answers are too short (avg < 100 words)")
    
    if quality['answers_too_short'] > len(all_pairs) * 0.1:
        score -= 15
        issues_found.append("⚠️ >10% of answers are too short")
    
    if quality['missing_keywords'] > len(all_pairs) * 0.2:
        score -= 10
        issues_found.append("⚠️ >20% missing keywords")
    
    if len(duplicates) > len(all_pairs) * 0.05:
        score -= 10
        issues_found.append("⚠️ >5% potential duplicates")
    
    # Check coverage gaps
    low_coverage = [topic for topic, data in coverage.items() if data['percentage'] < 10]
    if len(low_coverage) > 3:
        score -= 15
        issues_found.append(f"⚠️ Low coverage in {len(low_coverage)} topics")
    
    print(f"\n📊 Quality Score: {score}/100")
    
    if score >= 80:
        print("✅ EXCELLENT - Dataset is production-ready!")
    elif score >= 60:
        print("⚠️ GOOD - Minor improvements recommended")
    else:
        print("❌ NEEDS WORK - Significant issues found")
    
    if issues_found:
        print("\n🔧 Issues to address:")
        for issue in issues_found:
            print(f"   {issue}")
    
    print("\n" + "=" * 70)
    print("✅ Quality review complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
