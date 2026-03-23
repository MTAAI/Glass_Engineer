#!/usr/bin/env python3
"""
Synthetic Data Augmentation for Glass Expert AI
Expands existing Q&A pairs through rephrasing, variations, and multi-hop reasoning
Target: 4,400 → 8,000-10,000 Q&A pairs
"""

import json
import os
from pathlib import Path
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
import time
from tqdm import tqdm

# Load environment variables
load_dotenv()

class QAAugmenter:
    """Augments Q&A pairs with synthetic variations"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = config.get('model', 'gpt-4-mini')
        
    def rephrase_question(self, qa_pair: Dict) -> List[Dict]:
        """Generate 2-3 rephrased versions of the question"""
        
        prompt = f"""Given this Q&A pair about glass science, generate 2 alternative ways to ask the same question.
Keep the meaning identical but vary the phrasing, formality, and structure.

Original Question: {qa_pair['question']}
Original Answer: {qa_pair['answer']}

Generate 2 rephrased questions (keep answer the same):
1. [More technical/formal version]
2. [More conversational/practical version]

Return as JSON array:
[
  {{"question": "rephrased question 1", "variation_type": "formal"}},
  {{"question": "rephrased question 2", "variation_type": "conversational"}}
]
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            
            variations = json.loads(response.choices[0].message.content)
            
            # Create new Q&A pairs
            augmented = []
            for i, var in enumerate(variations):
                new_qa = qa_pair.copy()
                new_qa['id'] = f"{qa_pair['id']}_rephrase_{i+1}"
                new_qa['question'] = var['question']
                new_qa['augmentation_type'] = 'rephrase'
                new_qa['variation_type'] = var.get('variation_type', 'unknown')
                new_qa['original_id'] = qa_pair['id']
                augmented.append(new_qa)
            
            return augmented
            
        except Exception as e:
            print(f"Error rephrasing: {e}")
            return []
    
    def create_variations(self, qa_pair: Dict) -> List[Dict]:
        """Generate variations with different contexts or examples"""
        
        prompt = f"""Given this Q&A pair about glass science, create 2 variations that:
1. Add a specific example or application
2. Focus on a related but slightly different aspect

Original Question: {qa_pair['question']}
Original Answer: {qa_pair['answer']}

Generate 2 variations (modify both question AND answer):
1. [Add specific example - e.g., "in soda-lime glass" or "for automotive applications"]
2. [Related aspect - e.g., if original is about Tg, ask about measurement methods]

Return as JSON array:
[
  {{
    "question": "variation question 1",
    "answer": "variation answer 1",
    "variation_type": "example"
  }},
  {{
    "question": "variation question 2",
    "answer": "variation answer 2",
    "variation_type": "related_aspect"
  }}
]
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8
            )
            
            variations = json.loads(response.choices[0].message.content)
            
            # Create new Q&A pairs
            augmented = []
            for i, var in enumerate(variations):
                new_qa = qa_pair.copy()
                new_qa['id'] = f"{qa_pair['id']}_variation_{i+1}"
                new_qa['question'] = var['question']
                new_qa['answer'] = var['answer']
                new_qa['augmentation_type'] = 'variation'
                new_qa['variation_type'] = var.get('variation_type', 'unknown')
                new_qa['original_id'] = qa_pair['id']
                augmented.append(new_qa)
            
            return augmented
            
        except Exception as e:
            print(f"Error creating variations: {e}")
            return []
    
    def create_followup(self, qa_pair: Dict) -> List[Dict]:
        """Generate follow-up questions based on the answer"""
        
        prompt = f"""Given this Q&A pair about glass science, generate 1 natural follow-up question that:
- Digs deeper into a specific aspect mentioned in the answer
- Asks "why" or "how" about something stated
- Requests more detail or clarification

Original Question: {qa_pair['question']}
Original Answer: {qa_pair['answer']}

Generate 1 follow-up question with its answer:

Return as JSON:
{{
  "question": "follow-up question",
  "answer": "detailed answer to follow-up",
  "followup_aspect": "what aspect it follows up on"
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            
            followup = json.loads(response.choices[0].message.content)
            
            # Create new Q&A pair
            new_qa = qa_pair.copy()
            new_qa['id'] = f"{qa_pair['id']}_followup"
            new_qa['question'] = followup['question']
            new_qa['answer'] = followup['answer']
            new_qa['augmentation_type'] = 'followup'
            new_qa['followup_aspect'] = followup.get('followup_aspect', 'unknown')
            new_qa['original_id'] = qa_pair['id']
            
            return [new_qa]
            
        except Exception as e:
            print(f"Error creating follow-up: {e}")
            return []
    
    def create_multihop(self, qa_pairs: List[Dict]) -> List[Dict]:
        """Generate multi-hop reasoning questions from multiple Q&A pairs"""
        
        # Sample 2-3 related Q&A pairs
        if len(qa_pairs) < 2:
            return []
        
        # Group by category
        categories = {}
        for qa in qa_pairs:
            cat = qa.get('category', 'general')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(qa)
        
        multihop_pairs = []
        
        # Create multi-hop questions for each category with enough pairs
        for category, pairs in categories.items():
            if len(pairs) < 2:
                continue
            
            # Sample 2 pairs
            import random
            sampled = random.sample(pairs, min(2, len(pairs)))
            
            prompt = f"""Given these 2 Q&A pairs about glass science (category: {category}), create 1 multi-hop reasoning question that requires understanding BOTH concepts to answer.

Q&A Pair 1:
Q: {sampled[0]['question']}
A: {sampled[0]['answer'][:200]}...

Q&A Pair 2:
Q: {sampled[1]['question']}
A: {sampled[1]['answer'][:200]}...

Generate 1 multi-hop question that combines both concepts:

Return as JSON:
{{
  "question": "multi-hop question requiring both concepts",
  "answer": "comprehensive answer using both concepts",
  "reasoning_steps": ["step 1", "step 2"],
  "concepts_combined": ["concept 1", "concept 2"]
}}
"""
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8
                )
                
                multihop = json.loads(response.choices[0].message.content)
                
                # Create new Q&A pair
                new_qa = {
                    'id': f"multihop_{category}_{len(multihop_pairs)}",
                    'question': multihop['question'],
                    'answer': multihop['answer'],
                    'category': category,
                    'difficulty': 'advanced',
                    'augmentation_type': 'multihop',
                    'reasoning_steps': multihop.get('reasoning_steps', []),
                    'concepts_combined': multihop.get('concepts_combined', []),
                    'source_ids': [sampled[0]['id'], sampled[1]['id']],
                    'source': 'synthetic_multihop'
                }
                
                multihop_pairs.append(new_qa)
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"Error creating multi-hop for {category}: {e}")
                continue
        
        return multihop_pairs
    
    def augment_dataset(self, input_file: Path, output_dir: Path, 
                       augmentation_types: List[str] = ['rephrase', 'variation', 'followup', 'multihop'],
                       sample_size: int = None) -> Dict:
        """
        Augment entire dataset
        
        Args:
            input_file: Path to original Q&A JSONL file
            output_dir: Directory to save augmented data
            augmentation_types: Types of augmentation to apply
            sample_size: Number of original Q&A pairs to augment (None = all)
        
        Returns:
            Statistics dictionary
        """
        
        print(f"Loading Q&A pairs from {input_file}...")
        
        # Load original Q&A pairs
        original_pairs = []
        with open(input_file, 'r') as f:
            for line in f:
                original_pairs.append(json.loads(line))
        
        print(f"Loaded {len(original_pairs)} original Q&A pairs")
        
        # Sample if requested
        if sample_size and sample_size < len(original_pairs):
            import random
            original_pairs = random.sample(original_pairs, sample_size)
            print(f"Sampled {sample_size} pairs for augmentation")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Augment each pair
        all_augmented = []
        stats = {
            'original_count': len(original_pairs),
            'augmented_count': 0,
            'by_type': {}
        }
        
        for qa in tqdm(original_pairs, desc="Augmenting Q&A pairs"):
            augmented = []
            
            # Rephrase
            if 'rephrase' in augmentation_types:
                rephrased = self.rephrase_question(qa)
                augmented.extend(rephrased)
                stats['by_type']['rephrase'] = stats['by_type'].get('rephrase', 0) + len(rephrased)
            
            # Variations
            if 'variation' in augmentation_types:
                variations = self.create_variations(qa)
                augmented.extend(variations)
                stats['by_type']['variation'] = stats['by_type'].get('variation', 0) + len(variations)
            
            # Follow-ups
            if 'followup' in augmentation_types:
                followups = self.create_followup(qa)
                augmented.extend(followups)
                stats['by_type']['followup'] = stats['by_type'].get('followup', 0) + len(followups)
            
            all_augmented.extend(augmented)
            
            # Rate limiting
            time.sleep(0.5)
        
        # Multi-hop reasoning (requires multiple pairs)
        if 'multihop' in augmentation_types:
            print("\nGenerating multi-hop reasoning questions...")
            multihop = self.create_multihop(original_pairs)
            all_augmented.extend(multihop)
            stats['by_type']['multihop'] = len(multihop)
        
        stats['augmented_count'] = len(all_augmented)
        stats['total_count'] = len(original_pairs) + len(all_augmented)
        stats['expansion_ratio'] = stats['total_count'] / len(original_pairs)
        
        # Save augmented data
        augmented_file = output_dir / f"{input_file.stem}_augmented.jsonl"
        with open(augmented_file, 'w') as f:
            for qa in all_augmented:
                f.write(json.dumps(qa) + '\n')
        
        # Save combined data (original + augmented)
        combined_file = output_dir / f"{input_file.stem}_combined.jsonl"
        with open(combined_file, 'w') as f:
            for qa in original_pairs:
                f.write(json.dumps(qa) + '\n')
            for qa in all_augmented:
                f.write(json.dumps(qa) + '\n')
        
        # Save statistics
        stats_file = output_dir / f"{input_file.stem}_augmentation_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"AUGMENTATION COMPLETE")
        print(f"{'='*70}")
        print(f"Original Q&A pairs:     {stats['original_count']}")
        print(f"Augmented Q&A pairs:    {stats['augmented_count']}")
        print(f"Total Q&A pairs:        {stats['total_count']}")
        print(f"Expansion ratio:        {stats['expansion_ratio']:.2f}x")
        print(f"\nBy augmentation type:")
        for aug_type, count in stats['by_type'].items():
            print(f"  {aug_type:15s}: {count:5d}")
        print(f"\nFiles saved:")
        print(f"  Augmented only: {augmented_file}")
        print(f"  Combined:       {combined_file}")
        print(f"  Statistics:     {stats_file}")
        print(f"{'='*70}")
        
        return stats


# Configuration
CONFIG = {
    'model': 'gpt-4-mini',  # or 'gpt-4' for better quality
    'augmentation_types': ['rephrase', 'variation', 'followup', 'multihop'],
    'sample_size': None,  # None = augment all, or specify number
}


def main():
    """Main augmentation workflow"""
    
    print("="*70)
    print("GLASS EXPERT AI - SYNTHETIC DATA AUGMENTATION")
    print("="*70)
    print()
    
    # Initialize augmenter
    augmenter = QAAugmenter(CONFIG)
    
    # Define paths
    data_dir = Path('data/qa_pairs/raw')
    output_dir = Path('data/qa_pairs/augmented')
    
    # Find all Q&A files
    qa_files = list(data_dir.glob('*_qa.jsonl'))
    
    if not qa_files:
        print("ERROR: No Q&A files found in data/qa_pairs/raw/")
        print("Please run generate_qa_master.py first!")
        return
    
    print(f"Found {len(qa_files)} Q&A files to augment:")
    for f in qa_files:
        print(f"  - {f.name}")
    print()
    
    # Augment each file
    total_stats = {
        'original_total': 0,
        'augmented_total': 0,
        'combined_total': 0
    }
    
    for qa_file in qa_files:
        print(f"\nProcessing {qa_file.name}...")
        print("-" * 70)
        
        stats = augmenter.augment_dataset(
            input_file=qa_file,
            output_dir=output_dir,
            augmentation_types=CONFIG['augmentation_types'],
            sample_size=CONFIG['sample_size']
        )
        
        total_stats['original_total'] += stats['original_count']
        total_stats['augmented_total'] += stats['augmented_count']
        total_stats['combined_total'] += stats['total_count']
    
    # Print final summary
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY - ALL FILES")
    print(f"{'='*70}")
    print(f"Total original Q&A pairs:   {total_stats['original_total']}")
    print(f"Total augmented Q&A pairs:  {total_stats['augmented_total']}")
    print(f"Total combined Q&A pairs:   {total_stats['combined_total']}")
    print(f"Overall expansion ratio:    {total_stats['combined_total'] / total_stats['original_total']:.2f}x")
    print(f"{'='*70}")
    print()
    print("✅ Augmentation complete!")
    print(f"📁 Augmented files saved to: {output_dir}")
    print()
    print("Next steps:")
    print("1. Review augmented Q&A pairs for quality")
    print("2. Combine with domain-specific data")
    print("3. Create train/val/test splits")
    print("4. Export to training formats")


if __name__ == "__main__":
    main()
