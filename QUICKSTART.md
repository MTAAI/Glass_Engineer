# 🚀 Quick Start Guide - Glass Expert AI

**Get started in 10 minutes!**

---

## 📋 Prerequisites

- Python 3.11+
- OpenAI API key
- 10-30 GB free disk space
- Internet connection

---

## ⚡ 5-Minute Setup

### **Step 1: Download the System**

```bash
# Download from GitHub (once published)
git clone https://github.com/yourusername/glass-expert-ai.git
cd glass-expert-ai

# OR download the archive
# Extract glass-expert-ai-complete.tar.gz
```

### **Step 2: Install Dependencies**

```bash
# Install Python packages
pip install -r requirements.txt

# Install PDF processing libraries
pip install PyMuPDF pdfplumber PyPDF2
```

### **Step 3: Configure Environment**

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OpenAI API key
nano .env
# Change: OPENAI_API_KEY=your_actual_key_here
```

### **Step 4: Prepare Your Data**

```bash
# Create data directories
mkdir -p data/{textbooks,papers,csv}

# Copy your PDF files
cp /path/to/your/textbooks/*.pdf data/textbooks/
cp /path/to/your/papers/*.pdf data/papers/
cp /path/to/sciglass.csv data/csv/
```

### **Step 5: Generate Q&A Pairs!**

```bash
# Run the master generation script
python scripts/qa_generation/generate_qa_master.py
```

**That's it!** Your Q&A pairs will be generated in `data/qa_pairs/raw/`

---

## 📊 What You'll Get

After running the script, you'll have:

```
data/qa_pairs/
├── raw/
│   ├── textbooks_qa.jsonl      # Q&A from textbooks
│   ├── papers_qa.jsonl         # Q&A from papers
│   ├── sciglass_qa.jsonl       # Q&A from SciGlass
│   └── uci_qa.jsonl            # Q&A from UCI dataset
├── metadata/
│   └── statistics.json         # Generation statistics
└── qa_generation.log           # Detailed log
```

---

## 🎯 Usage Examples

### **Example 1: Generate Q&A from Specific Textbook**

```python
from scripts.qa_generation.generate_qa_master import QAGenerator, CONFIG

# Initialize generator
generator = QAGenerator(CONFIG)

# Process single PDF
qa_pairs = generator.process_pdf(
    Path('data/textbooks/Varshneya_Fundamentals.pdf'),
    'varshneya'
)

# Save results
generator.save_qa_pairs(qa_pairs, 'varshneya_qa.jsonl')

print(f"Generated {len(qa_pairs)} Q&A pairs!")
```

### **Example 2: View Generated Q&A Pairs**

```python
import json

# Load Q&A pairs
with open('data/qa_pairs/raw/textbooks_qa.jsonl') as f:
    for line in f:
        qa = json.loads(line)
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer'][:200]}...")
        print(f"Category: {qa['category']}")
        print("-" * 50)
```

### **Example 3: Filter by Category**

```python
import json

# Load and filter
qa_pairs = []
with open('data/qa_pairs/raw/textbooks_qa.jsonl') as f:
    for line in f:
        qa = json.loads(line)
        if qa['category'] == 'properties':
            qa_pairs.append(qa)

print(f"Found {len(qa_pairs)} Q&A pairs about properties")
```

---

## 🔧 Customization

### **Change Model (Cost vs Quality)**

Edit `scripts/qa_generation/generate_qa_master.py`:

```python
CONFIG = {
    'model': 'gpt-4',  # Better quality, higher cost
    # OR
    'model': 'gpt-4-mini',  # Good quality, lower cost
    ...
}
```

### **Adjust Q&A Quantity**

```python
CONFIG = {
    'qa_per_chunk': 3,  # Generate 3 Q&A pairs per chunk (instead of 2)
    'chunk_size': 2000,  # Larger chunks = fewer chunks = less Q&A
    ...
}
```

### **Parallel Processing Speed**

```python
CONFIG = {
    'max_workers': 8,  # More workers = faster (if you have CPU cores)
    ...
}
```

---

## 💰 Cost Estimation

### **Typical Costs (using gpt-4-mini):**

| Data Source | Files | Q&A Pairs | Cost |
|-------------|-------|-----------|------|
| 26 Textbooks | 26 PDFs | 1,500 | $30-40 |
| 100 Papers | 100 PDFs | 200 | $5-10 |
| SciGlass | 1000 samples | 500 | $10-15 |
| **Total** | - | **2,200** | **$45-65** |

### **Full Dataset (using gpt-4-mini):**

| Data Source | Files | Q&A Pairs | Cost |
|-------------|-------|-----------|------|
| 26 Textbooks | 26 PDFs | 1,750 | $35-45 |
| 1,029 Papers | 1,029 PDFs | 800 | $20-30 |
| SciGlass | 10K samples | 1,000 | $20-30 |
| GlassBench | Samples | 2,000 | $40-50 |
| **Total** | - | **5,550** | **$115-155** |

**Using GPT-4:** Multiply costs by ~3x

---

## 🐛 Troubleshooting

### **Problem: "OpenAI API key not found"**

**Solution:**
```bash
# Make sure .env file exists
ls -la .env

# Check if API key is set
cat .env | grep OPENAI_API_KEY

# Set it manually
export OPENAI_API_KEY=your_key_here
```

### **Problem: "PDF extraction failed"**

**Solution:**
```bash
# Install all PDF libraries
pip install PyMuPDF pdfplumber PyPDF2

# Try different extraction method
# Edit pdf_utils.py, line 15:
self.methods = ['pdfplumber', 'pymupdf', 'pypdf2']
```

### **Problem: "Out of memory"**

**Solution:**
```python
# Reduce parallel workers
CONFIG = {
    'max_workers': 2,  # Reduce from 4
    ...
}

# Process fewer files at once
textbook_pdfs = textbook_pdfs[:10]  # Process only 10 at a time
```

### **Problem: "Rate limit exceeded"**

**Solution:**
```python
# Add delays in generate_qa_master.py
import time

for i, chunk in enumerate(chunks):
    qa_pairs = self.generate_qa_from_text(chunk, source_name)
    all_qa.extend(qa_pairs)
    
    # Add delay every 5 chunks
    if i % 5 == 0:
        time.sleep(2)  # Increase from 1 to 2 seconds
```

---

## 📚 Next Steps

### **1. Explore Generated Q&A**
```bash
# View statistics
cat data/qa_pairs/metadata/statistics.json

# Count Q&A pairs
wc -l data/qa_pairs/raw/*.jsonl
```

### **2. Create Train/Val/Test Splits**
```bash
# Run split script (create this)
python scripts/qa_generation/create_splits.py
```

### **3. Export to Different Formats**
```bash
# Export to Parquet (for Hugging Face)
python scripts/qa_generation/export_formats.py --format parquet

# Export to CSV (for analysis)
python scripts/qa_generation/export_formats.py --format csv
```

### **4. Set Up Database**
```bash
# Create PostgreSQL database
createdb glass_db

# Run schema
psql glass_db < scripts/database/schema.sql

# Populate database
python scripts/database/populate_db.py
```

### **5. Deploy Publicly**
```bash
# Follow deployment guide
cat docs/DEPLOYMENT_GUIDE.md

# Push to GitHub
git add .
git commit -m "Initial release"
git push

# Upload to Hugging Face
python scripts/export_to_huggingface.py
```

---

## 🎓 Learning Resources

### **Documentation:**
- [Complete README](README.md)
- [Q&A Generation Strategy](docs/QA_GENERATION_STRATEGY.md)
- [System Architecture](GLASS_AI_SYSTEM_ARCHITECTURE.md)
- [Database Schema](scripts/database/schema.sql)
- [Deployment Guide](docs/DEPLOYMENT_GUIDE.md)

### **Example Notebooks:**
- `notebooks/01_data_exploration.ipynb` - Explore your data
- `notebooks/02_qa_generation_demo.ipynb` - Generate Q&A interactively
- `notebooks/03_database_queries.ipynb` - Query the database

---

## 💡 Tips for Best Results

### **1. Data Quality Matters**
- Use high-quality PDFs (not scanned images)
- Ensure text is extractable
- Remove duplicate files

### **2. Start Small, Scale Up**
```bash
# Test with 5 textbooks first
textbook_pdfs = textbook_pdfs[:5]

# Then scale to all 26
textbook_pdfs = list(data_sources['textbooks'].glob('*.pdf'))
```

### **3. Monitor Progress**
```bash
# Watch the log file
tail -f qa_generation.log

# Check statistics periodically
cat data/qa_pairs/metadata/statistics.json
```

### **4. Optimize Costs**
- Use `gpt-4-mini` for most Q&A
- Sample large datasets (don't process all 422K SciGlass records)
- Cache results to avoid re-generation

### **5. Validate Quality**
```python
# Review generated Q&A manually
import json

with open('data/qa_pairs/raw/textbooks_qa.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 10: break  # Review first 10
        qa = json.loads(line)
        print(f"\nQ: {qa['question']}")
        print(f"A: {qa['answer']}")
        input("Press Enter to continue...")
```

---

## 🎉 Success!

You now have:
- ✅ Working Q&A generation system
- ✅ High-quality Q&A pairs
- ✅ Structured database schema
- ✅ Complete documentation
- ✅ Deployment-ready code

**Ready to train your Glass Expert AI!** 🚀

---

## 🤝 Need Help?

- **Documentation:** Check [README.md](README.md)
- **Issues:** Open GitHub issue
- **Questions:** Discussion forum
- **Email:** your.email@example.com

---

**Happy Q&A Generation!** 🎓
