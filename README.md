# Glass Expert AI - Q&A Generation & Database System

**Complete system for generating high-quality Q&A datasets and clean databases from glass science data**

---

## 🎯 Overview

This project generates **5,000-7,500 expert-level Q&A pairs** from diverse glass science data sources and creates a **clean, structured database** for public access.

### **Key Features:**
- ✅ Multi-format data processing (PDF, CSV, large datasets)
- ✅ Intelligent Q&A generation using GPT-4
- ✅ Quality validation and categorization
- ✅ Parallel processing for efficiency
- ✅ Multiple export formats (JSONL, Parquet, CSV, SQLite)
- ✅ Public accessibility (GitHub, Hugging Face, hosted database)

---

## 📊 Data Sources

| Source | Format | Size | Q&A Pairs | Status |
|--------|--------|------|-----------|--------|
| **Textbooks** | PDF | 26 files | 1,350-1,750 | ✅ Ready |
| **Research Papers** | PDF | 1,029 files | 500-800 | ✅ Ready |
| **SciGlass** | CSV | 422K records | 500-1,000 | 🔄 Converting |
| **GlassBench** | Archive | 23 GB | 1,500-2,500 | 🔄 Downloading |
| **UCI Dataset** | CSV | 214 records | 50-100 | ✅ Ready |

**Total: 4,400-6,650 Q&A pairs**

---

## 🚀 Quick Start

### **1. Installation**

```bash
# Clone repository
git clone https://github.com/yourusername/glass-expert-ai.git
cd glass-expert-ai

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### **2. Prepare Your Data**

```bash
# Create data directories
mkdir -p data/{textbooks,papers,csv,large_datasets}

# Copy your data files
cp /path/to/textbooks/*.pdf data/textbooks/
cp /path/to/papers/*.pdf data/papers/
cp /path/to/sciglass.csv data/csv/
```

### **3. Generate Q&A Pairs**

```bash
# Run the master generation script
python scripts/qa_generation/generate_qa_master.py
```

### **4. View Results**

```bash
# Check generated Q&A pairs
ls -lh data/qa_pairs/raw/

# View statistics
cat data/qa_pairs/metadata/statistics.json
```

---

## 📁 Project Structure

```
glass-expert-ai/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── data/
│   ├── textbooks/          # Input: PDF textbooks
│   ├── papers/             # Input: Research papers
│   ├── csv/                # Input: CSV datasets
│   ├── large_datasets/     # Input: GlassBench, etc.
│   └── qa_pairs/           # Output: Generated Q&A
│       ├── raw/            # Raw Q&A by source
│       ├── processed/      # Train/val/test splits
│       ├── exports/        # Various formats
│       └── metadata/       # Statistics & reports
│
├── scripts/
│   ├── qa_generation/
│   │   ├── generate_qa_master.py      # Main orchestrator
│   │   ├── generate_from_pdf.py       # PDF-specific
│   │   ├── generate_from_csv.py       # CSV-specific
│   │   └── generate_from_large.py     # Large dataset handler
│   ├── database/
│   │   ├── extract_data.py            # Data extraction
│   │   ├── create_schema.py           # Database schema
│   │   └── populate_db.py             # Database population
│   └── utils/
│       ├── pdf_utils.py               # PDF extraction
│       ├── text_utils.py              # Text processing
│       └── db_utils.py                # Database utilities
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_qa_generation_demo.ipynb
│   └── 03_database_queries.ipynb
│
├── docs/
│   ├── QA_GENERATION_STRATEGY.md      # Detailed strategy
│   ├── architecture.md                # System architecture
│   ├── database_schema.md             # Database documentation
│   └── api_reference.md               # API documentation
│
└── tests/
    ├── test_pdf_extraction.py
    ├── test_qa_generation.py
    └── test_database.py
```

---

## 🔧 Configuration

### **Environment Variables (.env)**

```bash
# OpenAI API
OPENAI_API_KEY=your_api_key_here

# Model Selection
OPENAI_MODEL=gpt-4-mini  # or gpt-4 for better quality

# Processing
MAX_WORKERS=4  # Parallel processing threads
CHUNK_SIZE=1500  # Text chunk size
QA_PER_CHUNK=2  # Q&A pairs per chunk

# Database (optional)
DATABASE_URL=postgresql://user:pass@localhost:5432/glass_db
```

### **Script Configuration**

Edit `scripts/qa_generation/generate_qa_master.py`:

```python
CONFIG = {
    'model': 'gpt-4-mini',  # or 'gpt-4'
    'max_workers': 4,
    'chunk_size': 1500,
    'qa_per_chunk': 2,
    'output_dir': Path('data/qa_pairs'),
}

# Update data source paths
data_sources = {
    'textbooks': Path('data/textbooks'),
    'papers': Path('data/papers'),
    'sciglass': Path('data/csv/sciglass.csv'),
    'uci': Path('data/csv/glass.data'),
}
```

---

## 📖 Usage Examples

### **Example 1: Generate Q&A from Textbooks Only**

```python
from scripts.qa_generation.generate_qa_master import QAGenerator, CONFIG

generator = QAGenerator(CONFIG)

# Process textbooks
textbook_pdfs = list(Path('data/textbooks').glob('*.pdf'))
qa_pairs = generator.process_pdfs_parallel(textbook_pdfs, 'textbook')

# Save results
generator.save_qa_pairs(qa_pairs, 'textbooks_qa.jsonl')
```

### **Example 2: Generate Q&A from CSV Data**

```python
from scripts.qa_generation.generate_qa_master import QAGenerator, CONFIG

generator = QAGenerator(CONFIG)

# Process SciGlass
qa_pairs = generator.process_csv_data(
    Path('data/csv/sciglass.csv'),
    'sciglass'
)

# Save results
generator.save_qa_pairs(qa_pairs, 'sciglass_qa.jsonl')
```

### **Example 3: Process Large Dataset (GlassBench)**

```python
# TODO: Implement after exploring GlassBench structure
# See docs/QA_GENERATION_STRATEGY.md for details
```

---

## 📊 Output Formats

### **JSONL Format (for fine-tuning)**

```jsonl
{
  "id": "textbook_000001",
  "question": "What is the glass transition temperature and why is it important?",
  "answer": "The glass transition temperature (Tg) is the temperature range...",
  "difficulty": "intermediate",
  "keywords": ["glass transition", "Tg", "viscosity"],
  "source": "textbook_Varshneya",
  "category": "properties",
  "generated_at": "2026-02-25T12:00:00Z"
}
```

### **Parquet Format (for Hugging Face)**

```python
import pandas as pd

df = pd.read_parquet('data/qa_pairs/exports/all_qa_pairs.parquet')
print(df.head())
```

### **CSV Format (for analysis)**

```python
import pandas as pd

df = pd.read_csv('data/qa_pairs/exports/all_qa_pairs.csv')
print(df.describe())
```

---

## 🎯 Quality Metrics

### **Target Metrics:**
- ✅ Total Q&A pairs: 5,000-7,500
- ✅ Average answer length: 100-300 words
- ✅ Category coverage: All 8 categories balanced
- ✅ Difficulty levels: 30% beginner, 50% intermediate, 20% advanced
- ✅ Validation score: >0.9 for all pairs

### **Categories:**
1. **Fundamentals** - Glass structure, bonding, formation
2. **Composition** - Chemical compositions, raw materials
3. **Properties** - Physical, chemical, optical, mechanical
4. **Processing** - Melting, forming, annealing, tempering
5. **Applications** - Architectural, automotive, optical
6. **Defects** - Causes, identification, remedies
7. **Standards** - ASTM, ISO, quality control
8. **Advanced** - Coatings, glass-ceramics, specialty glasses

---

## 💰 Cost Estimation

### **OpenAI API Costs:**
- **GPT-4-mini:** $100-150 for 5,000-7,500 Q&A pairs
- **GPT-4:** $250-300 for 5,000-7,500 Q&A pairs

### **Cost Optimization Tips:**
1. Use `gpt-4-mini` for most Q&A generation
2. Use `gpt-4` only for complex technical Q&A
3. Enable caching to avoid re-generation
4. Batch multiple chunks per API call
5. Sample large datasets instead of processing all

---

## 🔍 Troubleshooting

### **Issue: OpenAI API Key Error**

```bash
# Check if API key is set
echo $OPENAI_API_KEY

# Set it in .env file
echo "OPENAI_API_KEY=your_key_here" >> .env
```

### **Issue: PDF Extraction Fails**

```bash
# Install additional dependencies
pip install PyMuPDF pdfplumber PyPDF2

# Try different extraction method
# Edit pdf_utils.py and change method preference
```

### **Issue: Out of Memory**

```bash
# Reduce parallel workers
# Edit CONFIG in generate_qa_master.py
CONFIG = {
    'max_workers': 2,  # Reduce from 4
    ...
}
```

### **Issue: Rate Limit Exceeded**

```bash
# Add rate limiting delays
# Already implemented in the script
# Adjust sleep time in generate_qa_master.py
```

---

## 📚 Documentation

- **[Q&A Generation Strategy](docs/QA_GENERATION_STRATEGY.md)** - Detailed strategy for handling diverse data
- **[System Architecture](docs/architecture.md)** - Complete system design
- **[Database Schema](docs/database_schema.md)** - Database structure and relationships
- **[API Reference](docs/api_reference.md)** - API endpoints and usage

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

### **Data Attribution:**
- Textbooks: Various authors and publishers (cited in metadata)
- Research Papers: Original authors (cited in metadata)
- SciGlass: NIST/SciGlass Database
- GlassBench: Original dataset creators
- UCI Dataset: UCI Machine Learning Repository

---

## 🎯 Roadmap

### **Phase 1: Q&A Generation** ✅ (Current)
- [x] PDF extraction utilities
- [x] Text processing utilities
- [x] Q&A generation pipeline
- [ ] Quality validation system
- [ ] Export to multiple formats

### **Phase 2: Database Extraction** ⏳ (Next)
- [ ] Data extraction from all sources
- [ ] PostgreSQL schema creation
- [ ] Database population
- [ ] API endpoints

### **Phase 3: Public Accessibility** ⏳ (Future)
- [ ] GitHub repository setup
- [ ] Hugging Face dataset upload
- [ ] Database hosting (Supabase/Neon)
- [ ] Documentation website

### **Phase 4: RAG & Fine-tuning** ⏳ (Future)
- [ ] Embedding generation
- [ ] Vector database setup
- [ ] RAG pipeline
- [ ] Fine-tune Llama-3-8B

---

## 📞 Contact

- **Project Lead:** [Your Name]
- **Email:** [your.email@example.com]
- **GitHub:** [https://github.com/yourusername/glass-expert-ai](https://github.com/yourusername/glass-expert-ai)

---

## 🙏 Acknowledgments

- OpenAI for GPT-4 API
- Glass science community for knowledge sharing
- All data providers and researchers

---

**Built with ❤️ for the glass science community**
