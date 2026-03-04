"""
Master Q&A Generation Script - FIXED INCREMENTAL SAVING
Saves after every 50 papers processed (not after collecting X Q&A pairs)
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import time

sys.path.append(str(Path(__file__).parent.parent))

from utils.pdf_utils import PDFExtractor
from utils.text_utils import clean_text, chunk_text_semantic, validate_text_quality

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qa_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'openai_api_key': os.getenv('OPENAI_API_KEY'),
    'model': 'gpt-4.1-mini',
    'max_workers': 50,
    'chunk_size': 2000,
    'qa_per_chunk': 4,
    'output_dir': Path('data/qa_pairs'),
    'save_batch_size': 50,  # Save after processing this many papers
}


class QAGenerator:
    """Q&A Generator with REAL incremental saving"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize OpenAI
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=config['openai_api_key'])
            logger.info("✅ OpenAI client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI: {e}")
            raise
        
        # Initialize PDF extractor
        self.pdf_extractor = PDFExtractor()
        
        # Statistics
        self.stats = {
            'start_time': datetime.now().isoformat(),
            'total_qa_pairs': 0,
            'by_source': {},
            'by_category': {},
            'api_calls': 0,
            'errors': 0,
            'successful_pdfs': 0,
            'failed_pdfs': 0,
        }
    
    def generate_qa_from_text(self, text: str, source: str, category: str = 'general') -> List[Dict]:
        """Generate Q&A pairs from text"""
        
        prompt = f"""You are an expert in glass science and technology. Generate {self.config['qa_per_chunk']} high-quality question-answer pairs from the following text.

Requirements:
1. Questions should be specific and answerable from the text
2. Answers should be detailed (150-250 words)
3. Include technical details and mechanisms
4. Use proper glass science terminology
5. Each Q&A should stand alone

Text:
{text[:self.config['chunk_size']]}

Return ONLY a JSON object with this structure:
{{
  "qa_pairs": [
    {{
      "question": "...",
      "answer": "...",
      "difficulty": "beginner/intermediate/advanced",
      "keywords": ["keyword1", "keyword2"]
    }}
  ]
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.config['model'],
                messages=[
                    {"role": "system", "content": "You are an expert glass science educator. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            self.stats['api_calls'] += 1
            
            content = response.choices[0].message.content
            
            try:
                data = json.loads(content)
                qa_pairs = data.get('qa_pairs', [])
                
                if not qa_pairs:
                    logger.warning(f"⚠️ No Q&A pairs in response for {source}")
                    return []
                
                # Add metadata
                for i, qa in enumerate(qa_pairs):
                    qa['id'] = f"{source}_{int(time.time())}_{i:05d}"
                    qa['source'] = source
                    qa['category'] = category
                    qa['generated_at'] = datetime.now().isoformat()
                
                self.stats['total_qa_pairs'] += len(qa_pairs)
                self.stats['by_source'][source] = self.stats['by_source'].get(source, 0) + len(qa_pairs)
                self.stats['by_category'][category] = self.stats['by_category'].get(category, 0) + len(qa_pairs)
                
                return qa_pairs
                
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON decode error for {source}: {e}")
                logger.error(f"Response content: {content[:200]}")
                return []
            
        except Exception as e:
            logger.error(f"❌ Error generating Q&A for {source}: {e}")
            self.stats['errors'] += 1
            return []
    
    def process_pdf(self, pdf_path: Path, source_name: str = None) -> List[Dict]:
        """Process single PDF"""
        if source_name is None:
            source_name = pdf_path.stem
        
        try:
            # Extract text
            result = self.pdf_extractor.extract_text(str(pdf_path))
            text = result['text']
            
            if not text or len(text) < 100:
                logger.warning(f"⚠️ Insufficient text from {pdf_path.name} (length: {len(text)})")
                self.stats['failed_pdfs'] += 1
                return []
            
            # Validate quality
            quality = validate_text_quality(text)
            if not quality['is_valid']:
                logger.warning(f"⚠️ Low quality text from {pdf_path.name}: {quality}")
                self.stats['failed_pdfs'] += 1
                return []
            
            # Clean and chunk
            text = clean_text(text)
            chunks = chunk_text_semantic(text, max_chunk_size=self.config['chunk_size'])
            
            if not chunks:
                logger.warning(f"⚠️ No chunks created from {pdf_path.name}")
                self.stats['failed_pdfs'] += 1
                return []
            
            # Limit chunks per paper
            max_chunks = 8
            if len(chunks) > max_chunks:
                import random
                chunks = random.sample(chunks, max_chunks)
            
            # Generate Q&A from chunks
            all_qa = []
            for i, chunk in enumerate(chunks):
                if len(chunk) < 200:  # Skip very short chunks
                    continue
                    
                qa_pairs = self.generate_qa_from_text(chunk, f"{source_name}_chunk{i}", 'paper')
                all_qa.extend(qa_pairs)
            
            if all_qa:
                self.stats['successful_pdfs'] += 1
                logger.info(f"✅ {pdf_path.name}: Generated {len(all_qa)} Q&A pairs from {len(chunks)} chunks")
            else:
                self.stats['failed_pdfs'] += 1
                logger.warning(f"⚠️ {pdf_path.name}: No Q&A pairs generated")
            
            return all_qa
            
        except Exception as e:
            logger.error(f"❌ Error processing {pdf_path.name}: {e}")
            self.stats['errors'] += 1
            self.stats['failed_pdfs'] += 1
            return []
    
    def append_qa_pairs(self, qa_pairs: List[Dict], filename: str):
        """Append Q&A pairs to file"""
        
        if not qa_pairs:
            logger.warning("⚠️ No Q&A pairs to save")
            return
        
        output_path = self.output_dir / 'generated' / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Append mode
        with open(output_path, 'a', encoding='utf-8') as f:
            for qa in qa_pairs:
                f.write(json.dumps(qa, ensure_ascii=False) + '\n')
        
        file_size_mb = output_path.stat().st_size / 1024 / 1024
        logger.info(f"💾 SAVED {len(qa_pairs)} Q&A to {output_path.name} (file now: {file_size_mb:.2f} MB)")
    
    def process_pdfs_incremental(self, pdf_paths: List[Path], category: str, output_filename: str):
        """Process PDFs with REAL incremental saving - saves after every N papers processed"""
        
        batch_size = self.config['save_batch_size']
        batch_qa = []
        papers_processed_in_batch = 0
        total_processed = 0
        
        with ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
            futures = {executor.submit(self.process_pdf, pdf): pdf for pdf in pdf_paths}
            
            with tqdm(total=len(pdf_paths), desc=f"Processing {category}") as pbar:
                for future in as_completed(futures):
                    try:
                        qa_pairs = future.result()
                        
                        if qa_pairs:
                            batch_qa.extend(qa_pairs)
                        
                        papers_processed_in_batch += 1
                        total_processed += 1
                        
                        # Save after processing batch_size papers (NOT after collecting X Q&A)
                        if papers_processed_in_batch >= batch_size:
                            logger.info(f"📦 Batch complete: {papers_processed_in_batch} papers processed, {len(batch_qa)} Q&A collected")
                            self.append_qa_pairs(batch_qa, output_filename)
                            batch_qa = []
                            papers_processed_in_batch = 0
                        
                    except Exception as e:
                        logger.error(f"❌ Future failed: {e}")
                    finally:
                        pbar.update(1)
            
            # Save remaining Q&A pairs
            if batch_qa:
                logger.info(f"📦 Final batch: {papers_processed_in_batch} papers, {len(batch_qa)} Q&A")
                self.append_qa_pairs(batch_qa, output_filename)
        
        logger.info(f"✅ Completed {category}: {total_processed} PDFs processed")
        logger.info(f"   Success: {self.stats['successful_pdfs']}, Failed: {self.stats['failed_pdfs']}")
    
    def process_csv_data(self, csv_path: Path, source_name: str, max_samples: int = 300) -> List[Dict]:
        """Process CSV data"""
        
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"📊 Loaded {len(df)} records from {csv_path.name}")
            
            if len(df) > max_samples:
                df = df.sample(max_samples)
                logger.info(f"📊 Sampled {max_samples} records")
            
            all_qa = []
            
            with tqdm(total=len(df), desc=f"CSV {csv_path.stem}") as pbar:
                for idx, row in df.iterrows():
                    text = " ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                    
                    if len(text) > 100:
                        qa_pairs = self.generate_qa_from_text(text[:2000], f"{source_name}_{idx}", 'data')
                        all_qa.extend(qa_pairs)
                    
                    pbar.update(1)
            
            return all_qa
            
        except Exception as e:
            logger.error(f"❌ Error processing CSV {csv_path}: {e}")
            return []
    
    def save_qa_pairs(self, qa_pairs: List[Dict], filename: str):
        """Save Q&A pairs to file"""
        
        if not qa_pairs:
            logger.warning(f"⚠️ No Q&A pairs to save for {filename}")
            return
        
        output_path = self.output_dir / 'generated' / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for qa in qa_pairs:
                f.write(json.dumps(qa, ensure_ascii=False) + '\n')
        
        file_size_mb = output_path.stat().st_size / 1024 / 1024
        logger.info(f"💾 Saved {len(qa_pairs)} Q&A to {output_path} ({file_size_mb:.2f} MB)")
    
    def generate_statistics(self):
        """Generate statistics"""
        
        self.stats['end_time'] = datetime.now().isoformat()
        
        start = datetime.fromisoformat(self.stats['start_time'])
        end = datetime.fromisoformat(self.stats['end_time'])
        duration = (end - start).total_seconds()
        
        self.stats['duration_seconds'] = duration
        self.stats['duration_minutes'] = duration / 60
        self.stats['qa_per_minute'] = self.stats['total_qa_pairs'] / (duration / 60) if duration > 0 else 0
        
        stats_path = self.output_dir / 'metadata' / 'statistics.json'
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"📊 Statistics saved to {stats_path}")
        logger.info(f"📊 Total Q&A pairs: {self.stats['total_qa_pairs']}")
        logger.info(f"📊 Successful PDFs: {self.stats['successful_pdfs']}")
        logger.info(f"📊 Failed PDFs: {self.stats['failed_pdfs']}")
        logger.info(f"📊 API calls: {self.stats['api_calls']}")


def main():
    """Main execution"""
    
    logger.info("=" * 70)
    logger.info("🔬 Glass Expert AI - Q&A Generation (FIXED VERSION)")
    logger.info("=" * 70)
    
    generator = QAGenerator(CONFIG)
    
    base_dir = Path('data')
    
    # 1. PROCESS PAPERS
    logger.info("\n" + "=" * 70)
    logger.info("📄 PROCESSING RESEARCH PAPERS")
    logger.info("=" * 70)
    
    papers_dir = base_dir / 'papers' / 'cutting_edge'
    if papers_dir.exists():
        papers = list(papers_dir.glob('*.pdf'))
        logger.info(f"📄 Found {len(papers)} papers")
        logger.info(f"💾 Will save after every {CONFIG['save_batch_size']} papers processed")
        
        generator.process_pdfs_incremental(papers, 'paper', 'papers_qa.jsonl')
    
    # 2. PROCESS TEXTBOOKS
    logger.info("\n" + "=" * 70)
    logger.info("📚 PROCESSING TEXTBOOKS")
    logger.info("=" * 70)
    
    all_textbooks = []
    
    textbooks_dir = base_dir / 'textbooks'
    if textbooks_dir.exists():
        root_pdfs = list(textbooks_dir.glob('*.pdf'))
        logger.info(f"📚 Found {len(root_pdfs)} textbooks in root")
        all_textbooks.extend(root_pdfs)
        
        tech_dir = textbooks_dir / 'technical'
        if tech_dir.exists():
            tech_pdfs = list(tech_dir.glob('*.pdf'))
            logger.info(f"📚 Found {len(tech_pdfs)} technical textbooks")
            all_textbooks.extend(tech_pdfs)
    
    qa_textbooks_dir = base_dir / 'qa_pairs' / 'raw'
    if qa_textbooks_dir.exists():
        for subdir in ['ancient', 'art', 'technical']:
            subdir_path = qa_textbooks_dir / subdir
            if subdir_path.exists():
                pdfs = list(subdir_path.glob('*.pdf'))
                if pdfs:
                    logger.info(f"📚 Found {len(pdfs)} {subdir} textbooks")
                    all_textbooks.extend(pdfs)
    
    if all_textbooks:
        all_textbooks = list(set(all_textbooks))
        logger.info(f"📚 Total unique textbooks: {len(all_textbooks)}")
        
        generator.process_pdfs_incremental(all_textbooks, 'textbook', 'textbooks_qa.jsonl')
    
    # 3. PROCESS SCIGLASS
    sciglass_path = base_dir / 'csv' / 'glass_data.csv'
    if sciglass_path.exists():
        logger.info("\n" + "=" * 70)
        logger.info("🗃️ PROCESSING SCIGLASS")
        logger.info("=" * 70)
        
        sciglass_qa = generator.process_csv_data(sciglass_path, 'sciglass', max_samples=300)
        generator.save_qa_pairs(sciglass_qa, 'sciglass_qa.jsonl')
    
    # 4. PROCESS UCI GLASS
    uci_path = base_dir / 'databases' / 'uci_glass' / 'glass.data'
    if uci_path.exists():
        logger.info("\n" + "=" * 70)
        logger.info("🔬 PROCESSING UCI GLASS")
        logger.info("=" * 70)
        
        uci_qa = generator.process_csv_data(uci_path, 'uci_glass', max_samples=100)
        generator.save_qa_pairs(uci_qa, 'uci_qa.jsonl')
    
    # Generate statistics
    generator.generate_statistics()
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ GENERATION COMPLETE!")
    logger.info(f"📊 Total Q&A pairs: {generator.stats['total_qa_pairs']}")
    logger.info(f"📊 Successful PDFs: {generator.stats['successful_pdfs']}")
    logger.info(f"📊 Failed PDFs: {generator.stats['failed_pdfs']}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
