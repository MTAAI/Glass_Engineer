-- Glass Expert AI Database Schema
-- PostgreSQL 15+ with pgvector extension

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- CORE TABLES
-- ============================================================================

-- Glass Compositions
CREATE TABLE glass_compositions (
    id SERIAL PRIMARY KEY,
    glass_id VARCHAR(100) UNIQUE,
    name VARCHAR(255),
    glass_type VARCHAR(100),  -- soda-lime, borosilicate, lead, etc.
    
    -- Major oxides (mol% or wt%)
    sio2 DECIMAL(6,3),
    na2o DECIMAL(6,3),
    cao DECIMAL(6,3),
    mgo DECIMAL(6,3),
    al2o3 DECIMAL(6,3),
    k2o DECIMAL(6,3),
    b2o3 DECIMAL(6,3),
    fe2o3 DECIMAL(6,3),
    pbo DECIMAL(6,3),
    
    -- Minor oxides
    tio2 DECIMAL(6,3),
    zno DECIMAL(6,3),
    bao DECIMAL(6,3),
    sro DECIMAL(6,3),
    li2o DECIMAL(6,3),
    p2o5 DECIMAL(6,3),
    
    -- Other components
    other_oxides JSONB,  -- For additional oxides
    
    -- Metadata
    composition_unit VARCHAR(20),  -- mol% or wt%
    source VARCHAR(255),
    reference TEXT,
    notes TEXT,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Glass Properties
CREATE TABLE glass_properties (
    id SERIAL PRIMARY KEY,
    composition_id INTEGER REFERENCES glass_compositions(id) ON DELETE CASCADE,
    
    property_name VARCHAR(100),  -- density, viscosity, Tg, etc.
    property_value DECIMAL(12,6),
    property_unit VARCHAR(50),
    
    -- Measurement conditions
    temperature DECIMAL(8,2),  -- in Celsius
    temperature_unit VARCHAR(10) DEFAULT '°C',
    pressure DECIMAL(10,3),  -- in MPa
    measurement_method VARCHAR(255),
    
    -- Quality indicators
    uncertainty DECIMAL(10,6),
    confidence_level DECIMAL(4,2),  -- 0-100%
    
    -- Metadata
    source VARCHAR(255),
    reference TEXT,
    measured_date DATE,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(composition_id, property_name, temperature)
);

-- Glass Processes
CREATE TABLE glass_processes (
    id SERIAL PRIMARY KEY,
    process_name VARCHAR(255),
    process_type VARCHAR(100),  -- melting, forming, annealing, tempering, coating
    
    description TEXT,
    detailed_steps TEXT,
    
    -- Process parameters (flexible JSON structure)
    parameters JSONB,
    
    -- Applicable glass types
    glass_types TEXT[],
    
    -- Equipment requirements
    equipment_required TEXT[],
    
    -- Quality control
    critical_parameters TEXT[],
    common_defects TEXT[],
    
    -- Metadata
    source VARCHAR(255),
    reference TEXT,
    industry_standard BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Glass Standards
CREATE TABLE glass_standards (
    id SERIAL PRIMARY KEY,
    standard_code VARCHAR(50) UNIQUE,  -- ASTM C1036, ISO 12543, etc.
    organization VARCHAR(100),  -- ASTM, ISO, EN, etc.
    
    title TEXT,
    description TEXT,
    scope TEXT,
    
    -- Standard details
    version VARCHAR(50),
    published_date DATE,
    status VARCHAR(50),  -- active, superseded, withdrawn
    
    -- Requirements (flexible JSON structure)
    requirements JSONB,
    test_methods JSONB,
    acceptance_criteria JSONB,
    
    -- Related standards
    related_standards TEXT[],
    supersedes VARCHAR(50),
    superseded_by VARCHAR(50),
    
    -- Metadata
    source VARCHAR(255),
    document_url TEXT,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Glass Defects
CREATE TABLE glass_defects (
    id SERIAL PRIMARY KEY,
    defect_name VARCHAR(255),
    defect_type VARCHAR(100),  -- bubble, stone, cord, stress, crack, etc.
    defect_category VARCHAR(50),  -- visual, structural, chemical
    
    description TEXT,
    appearance TEXT,
    
    -- Causes and remedies
    causes TEXT[],
    root_causes TEXT[],
    remedies TEXT[],
    prevention_methods TEXT[],
    
    -- Severity and impact
    severity_level VARCHAR(50),  -- minor, major, critical
    impact_on_quality TEXT,
    impact_on_performance TEXT,
    
    -- Applicable glass types and processes
    glass_types TEXT[],
    related_processes TEXT[],
    
    -- Detection methods
    detection_methods TEXT[],
    inspection_standards TEXT[],
    
    -- Metadata
    source VARCHAR(255),
    reference TEXT,
    images_url TEXT[],
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Research Papers
CREATE TABLE research_papers (
    id SERIAL PRIMARY KEY,
    paper_id VARCHAR(100) UNIQUE,
    
    title TEXT,
    authors TEXT[],
    author_affiliations TEXT[],
    
    -- Publication details
    journal VARCHAR(255),
    conference VARCHAR(255),
    year INTEGER,
    volume VARCHAR(50),
    issue VARCHAR(50),
    pages VARCHAR(50),
    
    -- Identifiers
    doi VARCHAR(255),
    arxiv_id VARCHAR(50),
    pmid VARCHAR(50),
    isbn VARCHAR(50),
    
    -- Content
    abstract TEXT,
    keywords TEXT[],
    
    -- Classification
    categories TEXT[],  -- composition, properties, processing, etc.
    glass_types TEXT[],
    topics TEXT[],
    
    -- Files and links
    file_path VARCHAR(500),
    pdf_url TEXT,
    external_links TEXT[],
    
    -- Metadata
    citation_count INTEGER,
    download_count INTEGER,
    quality_score DECIMAL(4,2),
    
    -- Timestamps
    published_date DATE,
    added_date DATE DEFAULT CURRENT_DATE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Q&A Pairs (for training data)
CREATE TABLE qa_pairs (
    id SERIAL PRIMARY KEY,
    qa_id VARCHAR(100) UNIQUE,
    
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    
    -- Classification
    category VARCHAR(100),
    subcategory VARCHAR(100),
    difficulty VARCHAR(50),  -- beginner, intermediate, advanced
    
    -- Keywords and concepts
    keywords TEXT[],
    related_concepts TEXT[],
    
    -- Source information
    source VARCHAR(255),
    source_type VARCHAR(50),  -- textbook, paper, database, synthetic
    source_id INTEGER,  -- Reference to source table
    
    -- Quality metrics
    validation_score DECIMAL(4,2),
    quality_score DECIMAL(4,2),
    human_verified BOOLEAN DEFAULT FALSE,
    
    -- Usage tracking
    used_in_training BOOLEAN DEFAULT FALSE,
    training_split VARCHAR(20),  -- train, validation, test
    
    -- Metadata
    generated_at TIMESTAMP,
    verified_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================================
-- EMBEDDINGS & VECTOR SEARCH
-- ============================================================================

-- Text Embeddings (for RAG)
CREATE TABLE text_embeddings (
    id SERIAL PRIMARY KEY,
    
    -- Source information
    source_type VARCHAR(50),  -- textbook, paper, qa_pair, etc.
    source_id INTEGER,
    source_text TEXT,
    
    -- Embedding
    embedding vector(1536),  -- OpenAI ada-002 dimension
    
    -- Metadata
    chunk_index INTEGER,
    total_chunks INTEGER,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================================
-- AUXILIARY TABLES
-- ============================================================================

-- Glass Applications
CREATE TABLE glass_applications (
    id SERIAL PRIMARY KEY,
    application_name VARCHAR(255),
    application_category VARCHAR(100),  -- architectural, automotive, optical, etc.
    
    description TEXT,
    requirements TEXT,
    
    -- Suitable glass types
    suitable_glass_types TEXT[],
    typical_compositions JSONB,
    
    -- Performance requirements
    required_properties JSONB,
    standards TEXT[],
    
    -- Market information
    market_size VARCHAR(100),
    growth_rate DECIMAL(5,2),
    key_manufacturers TEXT[],
    
    -- Metadata
    source VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Raw Materials
CREATE TABLE raw_materials (
    id SERIAL PRIMARY KEY,
    material_name VARCHAR(255),
    chemical_formula VARCHAR(100),
    cas_number VARCHAR(50),
    
    description TEXT,
    purpose TEXT,  -- glass former, flux, stabilizer, etc.
    
    -- Properties
    purity_grade VARCHAR(50),
    typical_purity DECIMAL(5,2),
    melting_point DECIMAL(8,2),
    
    -- Usage
    typical_usage_range JSONB,  -- {min: 0, max: 100, unit: "wt%"}
    compatible_glass_types TEXT[],
    
    -- Safety and handling
    hazards TEXT[],
    safety_precautions TEXT,
    
    -- Metadata
    source VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Manufacturers and Suppliers
CREATE TABLE manufacturers (
    id SERIAL PRIMARY KEY,
    company_name VARCHAR(255),
    country VARCHAR(100),
    
    company_type VARCHAR(50),  -- manufacturer, supplier, research
    
    -- Products and services
    products TEXT[],
    glass_types TEXT[],
    specializations TEXT[],
    
    -- Contact information
    website TEXT,
    contact_email VARCHAR(255),
    
    -- Metadata
    founded_year INTEGER,
    employee_count INTEGER,
    annual_revenue VARCHAR(100),
    
    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Glass Compositions
CREATE INDEX idx_compositions_type ON glass_compositions(glass_type);
CREATE INDEX idx_compositions_sio2 ON glass_compositions(sio2);
CREATE INDEX idx_compositions_source ON glass_compositions(source);

-- Glass Properties
CREATE INDEX idx_properties_composition ON glass_properties(composition_id);
CREATE INDEX idx_properties_name ON glass_properties(property_name);
CREATE INDEX idx_properties_temp ON glass_properties(temperature);

-- Glass Processes
CREATE INDEX idx_processes_type ON glass_processes(process_type);
CREATE INDEX idx_processes_glass_types ON glass_processes USING GIN(glass_types);

-- Glass Standards
CREATE INDEX idx_standards_code ON glass_standards(standard_code);
CREATE INDEX idx_standards_org ON glass_standards(organization);

-- Glass Defects
CREATE INDEX idx_defects_type ON glass_defects(defect_type);
CREATE INDEX idx_defects_category ON glass_defects(defect_category);
CREATE INDEX idx_defects_glass_types ON glass_defects USING GIN(glass_types);

-- Research Papers
CREATE INDEX idx_papers_year ON research_papers(year);
CREATE INDEX idx_papers_doi ON research_papers(doi);
CREATE INDEX idx_papers_categories ON research_papers USING GIN(categories);
CREATE INDEX idx_papers_keywords ON research_papers USING GIN(keywords);

-- Q&A Pairs
CREATE INDEX idx_qa_category ON qa_pairs(category);
CREATE INDEX idx_qa_difficulty ON qa_pairs(difficulty);
CREATE INDEX idx_qa_source ON qa_pairs(source);
CREATE INDEX idx_qa_split ON qa_pairs(training_split);

-- Text Embeddings (for vector search)
CREATE INDEX idx_embeddings_source ON text_embeddings(source_type, source_id);
CREATE INDEX idx_embeddings_vector ON text_embeddings USING ivfflat (embedding vector_cosine_ops);

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- Complete glass information (composition + properties)
CREATE VIEW glass_complete AS
SELECT 
    gc.id,
    gc.glass_id,
    gc.name,
    gc.glass_type,
    gc.sio2, gc.na2o, gc.cao, gc.mgo, gc.al2o3,
    json_agg(
        json_build_object(
            'property', gp.property_name,
            'value', gp.property_value,
            'unit', gp.property_unit,
            'temperature', gp.temperature
        )
    ) as properties
FROM glass_compositions gc
LEFT JOIN glass_properties gp ON gc.id = gp.composition_id
GROUP BY gc.id;

-- Q&A pairs with source information
CREATE VIEW qa_with_sources AS
SELECT 
    qa.id,
    qa.qa_id,
    qa.question,
    qa.answer,
    qa.category,
    qa.difficulty,
    qa.source,
    qa.validation_score,
    CASE 
        WHEN qa.source_type = 'paper' THEN rp.title
        ELSE qa.source
    END as source_title
FROM qa_pairs qa
LEFT JOIN research_papers rp ON qa.source_type = 'paper' AND qa.source_id = rp.id;

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Update timestamp function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_glass_compositions_updated_at BEFORE UPDATE ON glass_compositions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_glass_processes_updated_at BEFORE UPDATE ON glass_processes
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_glass_standards_updated_at BEFORE UPDATE ON glass_standards
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_glass_defects_updated_at BEFORE UPDATE ON glass_defects
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Vector similarity search function
CREATE OR REPLACE FUNCTION search_similar_text(
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.7,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    id integer,
    source_type varchar,
    source_id integer,
    source_text text,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        te.id,
        te.source_type,
        te.source_id,
        te.source_text,
        1 - (te.embedding <=> query_embedding) as similarity
    FROM text_embeddings te
    WHERE 1 - (te.embedding <=> query_embedding) > match_threshold
    ORDER BY te.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE glass_compositions IS 'Glass chemical compositions with oxide percentages';
COMMENT ON TABLE glass_properties IS 'Physical, chemical, and optical properties of glasses';
COMMENT ON TABLE glass_processes IS 'Manufacturing and processing methods for glass';
COMMENT ON TABLE glass_standards IS 'Industry standards (ASTM, ISO, EN) for glass products';
COMMENT ON TABLE glass_defects IS 'Common defects, causes, and remedies';
COMMENT ON TABLE research_papers IS 'Academic papers and research publications';
COMMENT ON TABLE qa_pairs IS 'Question-answer pairs for training AI models';
COMMENT ON TABLE text_embeddings IS 'Vector embeddings for semantic search and RAG';

-- ============================================================================
-- INITIAL DATA (Optional)
-- ============================================================================

-- Insert common glass types
INSERT INTO glass_compositions (glass_id, name, glass_type, sio2, na2o, cao, composition_unit, source)
VALUES 
    ('SODA_LIME_STANDARD', 'Standard Soda-Lime Glass', 'soda-lime', 72.0, 14.0, 10.0, 'wt%', 'Standard Composition'),
    ('BOROSILICATE_PYREX', 'Pyrex Borosilicate Glass', 'borosilicate', 80.6, 4.0, 0.0, 'wt%', 'Corning'),
    ('LEAD_CRYSTAL', 'Lead Crystal Glass', 'lead', 59.0, 0.0, 0.0, 'wt%', 'Standard Composition');

-- Insert common glass properties
-- (Will be populated by data extraction scripts)

COMMENT ON DATABASE current_database() IS 'Glass Expert AI - Comprehensive glass science database';
