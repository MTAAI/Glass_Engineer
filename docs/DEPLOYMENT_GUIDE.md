# Deployment & Public Accessibility Guide

**Complete guide for deploying the Glass Expert AI system and making it publicly accessible**

---

## 🎯 Overview

This guide covers:
1. **GitHub Repository Setup** - Code and documentation
2. **Hugging Face Datasets** - Q&A datasets for public download
3. **Database Hosting** - PostgreSQL database with public API
4. **Documentation Website** - Comprehensive documentation
5. **API Deployment** - REST API for developers

---

## 📦 PART 1: GitHub Repository Setup

### **Step 1: Create GitHub Repository**

```bash
# Initialize git repository
cd glass-expert-ai
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Glass Expert AI system"

# Create GitHub repository (using gh CLI)
gh repo create glass-expert-ai --public --source=. --remote=origin

# Push to GitHub
git push -u origin main
```

### **Step 2: Repository Structure**

Your repository will have:
```
glass-expert-ai/
├── README.md                    # Main documentation
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── .env.example                 # Environment template
│
├── scripts/                     # All Python scripts
├── docs/                        # Documentation
├── notebooks/                   # Jupyter notebooks
├── tests/                       # Unit tests
│
└── data/                        # Sample data only
    └── samples/
        ├── sample_qa.jsonl      # 100 sample Q&A pairs
        └── sample_compositions.csv  # 1000 sample records
```

### **Step 3: Add Sample Data**

```bash
# Create samples directory
mkdir -p data/samples

# Extract 100 sample Q&A pairs
head -100 data/qa_pairs/raw/textbooks_qa.jsonl > data/samples/sample_qa.jsonl

# Add to git
git add data/samples/
git commit -m "Add sample data"
git push
```

### **Step 4: Create GitHub Actions (CI/CD)**

Create `.github/workflows/ci.yml`:
```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest
      - name: Run tests
        run: pytest tests/
```

---

## 🤗 PART 2: Hugging Face Datasets

### **Step 1: Create Hugging Face Account**

1. Go to [huggingface.co](https://huggingface.co)
2. Create account
3. Generate access token (Settings → Access Tokens)

### **Step 2: Prepare Dataset**

```python
# scripts/export_to_huggingface.py
from datasets import Dataset, DatasetDict
import pandas as pd
import json

# Load Q&A pairs
qa_pairs = []
with open('data/qa_pairs/processed/train.jsonl') as f:
    for line in f:
        qa_pairs.append(json.loads(line))

# Create dataset
df = pd.DataFrame(qa_pairs)
dataset = Dataset.from_pandas(df)

# Create splits
dataset_dict = DatasetDict({
    'train': Dataset.from_pandas(pd.read_json('data/qa_pairs/processed/train.jsonl', lines=True)),
    'validation': Dataset.from_pandas(pd.read_json('data/qa_pairs/processed/validation.jsonl', lines=True)),
    'test': Dataset.from_pandas(pd.read_json('data/qa_pairs/processed/test.jsonl', lines=True))
})

# Push to Hugging Face
dataset_dict.push_to_hub("yourusername/glass-expert-qa")
```

### **Step 3: Create Dataset Card**

Create `dataset_card.md`:
```markdown
---
language:
- en
task_categories:
- question-answering
- text-generation
tags:
- glass-science
- materials-science
- expert-knowledge
size_categories:
- 1K<n<10K
---

# Glass Expert Q&A Dataset

## Dataset Description

This dataset contains 5,000-7,500 expert-level question-answer pairs about glass science and technology.

### Dataset Summary

- **Total Q&A pairs:** 5,500
- **Categories:** 8 (fundamentals, composition, properties, processing, applications, defects, standards, advanced)
- **Difficulty levels:** Beginner (30%), Intermediate (50%), Advanced (20%)
- **Languages:** English
- **License:** CC BY 4.0

### Data Sources

- 26 authoritative textbooks
- 1,029 research papers
- SciGlass database (422K records)
- GlassBench dataset
- UCI Glass Identification dataset

### Dataset Structure

```json
{
  "id": "qa_000001",
  "question": "What is the glass transition temperature?",
  "answer": "The glass transition temperature (Tg) is...",
  "category": "properties",
  "difficulty": "intermediate",
  "keywords": ["glass transition", "Tg", "viscosity"],
  "source": "textbook_Varshneya"
}
```

### Usage

```python
from datasets import load_dataset

dataset = load_dataset("yourusername/glass-expert-qa")

# Access train split
train_data = dataset['train']

# Example
print(train_data[0])
```

### Citation

```bibtex
@dataset{glass_expert_qa_2026,
  title={Glass Expert Q&A Dataset},
  author={Your Name},
  year={2026},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/yourusername/glass-expert-qa}
}
```

### License

CC BY 4.0 - Attribution required
```

### **Step 4: Upload to Hugging Face**

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Upload dataset
python scripts/export_to_huggingface.py
```

---

## 🗄️ PART 3: Database Hosting (Supabase)

### **Step 1: Create Supabase Project**

1. Go to [supabase.com](https://supabase.com)
2. Create new project
3. Note down:
   - Project URL
   - API Key (anon/public)
   - Database password

### **Step 2: Set Up Database**

```bash
# Install Supabase CLI
npm install -g supabase

# Login
supabase login

# Link project
supabase link --project-ref your-project-ref

# Run migrations
supabase db push

# Or manually run SQL
psql "postgresql://postgres:password@db.xxx.supabase.co:5432/postgres" < scripts/database/schema.sql
```

### **Step 3: Populate Database**

```python
# scripts/database/populate_supabase.py
import psycopg2
import pandas as pd
import os

# Connect to Supabase
conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cur = conn.cursor()

# Load SciGlass data
df = pd.read_csv('data/csv/sciglass.csv')

# Insert compositions
for _, row in df.iterrows():
    cur.execute("""
        INSERT INTO glass_compositions 
        (glass_id, sio2, na2o, cao, source)
        VALUES (%s, %s, %s, %s, %s)
    """, (row['id'], row['SiO2'], row['Na2O'], row['CaO'], 'SciGlass'))

conn.commit()
cur.close()
conn.close()
```

### **Step 4: Enable Public Access**

In Supabase Dashboard:
1. Go to **Settings** → **API**
2. Enable **Public API**
3. Set **Row Level Security (RLS)** policies:

```sql
-- Allow public read access
CREATE POLICY "Public read access" ON glass_compositions
FOR SELECT USING (true);

CREATE POLICY "Public read access" ON glass_properties
FOR SELECT USING (true);

-- Repeat for all tables
```

### **Step 5: Test API**

```bash
# Test REST API
curl "https://your-project.supabase.co/rest/v1/glass_compositions?select=*&limit=10" \
  -H "apikey: your-anon-key"
```

---

## 📚 PART 4: Documentation Website (GitHub Pages)

### **Step 1: Create Documentation Site**

```bash
# Install MkDocs
pip install mkdocs mkdocs-material

# Create mkdocs.yml
cat > mkdocs.yml << EOF
site_name: Glass Expert AI
site_description: Comprehensive glass science Q&A and database system
theme:
  name: material
  palette:
    primary: indigo
    accent: light blue

nav:
  - Home: index.md
  - Getting Started: getting-started.md
  - Q&A Generation: qa-generation.md
  - Database: database.md
  - API Reference: api-reference.md
  - Examples: examples.md
EOF

# Create docs directory
mkdir -p docs_site
cp docs/*.md docs_site/
```

### **Step 2: Deploy to GitHub Pages**

```bash
# Build documentation
mkdocs build

# Deploy to GitHub Pages
mkdocs gh-deploy
```

Your documentation will be available at:
`https://yourusername.github.io/glass-expert-ai/`

---

## 🚀 PART 5: API Deployment (FastAPI)

### **Step 1: Create FastAPI Application**

```python
# scripts/api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
import os

app = FastAPI(
    title="Glass Expert AI API",
    description="API for glass science data and Q&A",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection
def get_db():
    return psycopg2.connect(os.getenv('DATABASE_URL'))

@app.get("/")
def read_root():
    return {"message": "Glass Expert AI API", "version": "1.0.0"}

@app.get("/compositions")
def get_compositions(limit: int = 100, offset: int = 0):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM glass_compositions
        LIMIT %s OFFSET %s
    """, (limit, offset))
    results = cur.fetchall()
    cur.close()
    conn.close()
    return {"data": results}

@app.get("/qa-pairs")
def get_qa_pairs(category: str = None, difficulty: str = None, limit: int = 100):
    conn = get_db()
    cur = conn.cursor()
    
    query = "SELECT * FROM qa_pairs WHERE 1=1"
    params = []
    
    if category:
        query += " AND category = %s"
        params.append(category)
    
    if difficulty:
        query += " AND difficulty = %s"
        params.append(difficulty)
    
    query += " LIMIT %s"
    params.append(limit)
    
    cur.execute(query, params)
    results = cur.fetchall()
    cur.close()
    conn.close()
    
    return {"data": results}

@app.get("/search")
def search_glass(query: str):
    # Implement search logic
    return {"query": query, "results": []}
```

### **Step 2: Deploy to Cloud**

#### **Option A: Deploy to Render.com (Free)**

1. Create `render.yaml`:
```yaml
services:
  - type: web
    name: glass-expert-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn scripts.api.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: DATABASE_URL
        sync: false
```

2. Push to GitHub
3. Connect to Render.com
4. Deploy automatically

#### **Option B: Deploy to Railway.app (Free)**

1. Connect GitHub repository
2. Set environment variables
3. Deploy automatically

#### **Option C: Deploy to Vercel (Free)**

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
vercel
```

---

## 📊 PART 6: Monitoring & Analytics

### **Add Analytics to GitHub**

```bash
# Add GitHub badges to README.md
[![Stars](https://img.shields.io/github/stars/yourusername/glass-expert-ai)](https://github.com/yourusername/glass-expert-ai)
[![Downloads](https://img.shields.io/github/downloads/yourusername/glass-expert-ai/total)](https://github.com/yourusername/glass-expert-ai/releases)
[![License](https://img.shields.io/github/license/yourusername/glass-expert-ai)](LICENSE)
```

### **Track Hugging Face Downloads**

Hugging Face automatically tracks:
- Dataset downloads
- Model downloads
- Views and likes

### **Monitor API Usage**

```python
# Add logging to FastAPI
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)

@app.middleware("http")
async def log_requests(request, call_next):
    logging.info(f"{request.method} {request.url}")
    response = await call_next(request)
    return response
```

---

## ✅ DEPLOYMENT CHECKLIST

### **GitHub Repository**
- [ ] Repository created and public
- [ ] README.md with clear instructions
- [ ] LICENSE file (MIT recommended)
- [ ] .gitignore configured
- [ ] Sample data included
- [ ] GitHub Actions CI/CD set up
- [ ] Releases created

### **Hugging Face**
- [ ] Account created
- [ ] Dataset uploaded
- [ ] Dataset card completed
- [ ] Tags and categories set
- [ ] License specified
- [ ] Example code provided

### **Database (Supabase)**
- [ ] Project created
- [ ] Schema deployed
- [ ] Data populated
- [ ] Public API enabled
- [ ] RLS policies configured
- [ ] API tested

### **Documentation**
- [ ] Documentation site created
- [ ] All guides completed
- [ ] API reference documented
- [ ] Examples provided
- [ ] Deployed to GitHub Pages

### **API**
- [ ] FastAPI application created
- [ ] Endpoints implemented
- [ ] CORS configured
- [ ] Deployed to cloud
- [ ] API tested
- [ ] Rate limiting added

### **Publicity**
- [ ] Announced on social media
- [ ] Posted on relevant forums
- [ ] Added to awesome lists
- [ ] Submitted to directories

---

## 🌐 PUBLIC ACCESS URLS

After deployment, you'll have:

1. **GitHub Repository:**
   `https://github.com/yourusername/glass-expert-ai`

2. **Hugging Face Dataset:**
   `https://huggingface.co/datasets/yourusername/glass-expert-qa`

3. **Documentation:**
   `https://yourusername.github.io/glass-expert-ai/`

4. **Database API:**
   `https://your-project.supabase.co/rest/v1/`

5. **Custom API:**
   `https://glass-expert-api.onrender.com/`

---

## 📈 EXPECTED IMPACT

### **Target Metrics (6 months):**
- ✅ 100+ GitHub stars
- ✅ 1,000+ Hugging Face downloads
- ✅ 10,000+ API requests
- ✅ 50+ citations
- ✅ 10+ contributors

### **Use Cases:**
- AI/ML researchers training models
- Glass industry professionals
- Materials science students
- Academic researchers
- Software developers

---

**Your Glass Expert AI system is now publicly accessible and ready to help the glass science community!** 🎉
