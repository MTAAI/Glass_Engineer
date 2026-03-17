"""
Glass Expert AI - Pydantic Schemas
Merged: Arjun branch + glass-expert-ai branch
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional, Union

# ─── Base & Common Models ───────────────────────────────────────────────────

class SourceChunk(BaseModel):
    title: str = Field(..., description="Title of the source document.")
    source_type: str = Field(..., description="Type of the source (e.g., 'textbook', 'paper').")
    language: str = Field(..., description="Language of the chunk content.")
    similarity: float = Field(..., description="Similarity score of the chunk to the query.")
    content_preview: str = Field(..., description="A short preview of the chunk's content.")
    rerank_score: Optional[float] = Field(None, description="Cross-encoder reranking score.")

# ─── Query Endpoint ─────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=2000, description="The user's question about glass science.")
    language: str = Field("auto", description="Language for the answer ('en', 'fa', or 'auto').")
    mode: str = Field("simple", description="Query mode: 'simple', 'detailed', or 'research'.")
    top_k: int = Field(5, ge=1, le=20, description="Number of source chunks to retrieve.")
    source_type: Optional[str] = Field(None, description="Filter by source type (e.g. 'textbook', 'paper').")
    session_id: Optional[str] = Field(None, description="Conversation session ID for chat history.")

class QueryResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    question: str
    answer: str
    sources: List[SourceChunk]
    query_id: Optional[str] = None
    session_id: Optional[str] = None
    chat_id: Optional[str] = None
    model_used: str
    retrieval_time_ms: float
    generation_time_ms: Optional[float] = None
    language_detected: Optional[str] = None
    total_chunks_searched: Optional[int] = None

# ─── Health Endpoint ────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    database: str
    redis: str
    embedding_model: str = "BAAI/bge-m3"
    total_documents: int
    total_chunks: int
    version: str = "3.0.0"

# ─── Ingest Endpoint ────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Full text content of the document")
    source_type: str = Field("textbook", description="Type: textbook, paper, sop, standard, datasheet")
    language: str = Field("en", description="Language code: en or fa")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class IngestResponse(BaseModel):
    success: bool
    message: str
    chunks_created: int
    document_title: str

# ─── Analyze Endpoint ───────────────────────────────────────────────────────

class PropertyPrediction(BaseModel):
    property_name: str
    value: str
    unit: Optional[str] = None
    confidence: str = "medium"
    source: str = "LLM analysis"

class AnalyzeRequest(BaseModel):
    composition: Optional[Dict[str, float]] = None
    description: Optional[str] = None
    properties_of_interest: Optional[List[str]] = None

class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    composition_summary: Optional[str] = None
    analysis: str
    predicted_properties: List[PropertyPrediction]
    recommendations: List[str]
    sources: List[SourceChunk]
    model_used: str
    retrieval_time_ms: float

# ─── Design Endpoint ────────────────────────────────────────────────────────

class DesignRequest(BaseModel):
    target_properties: Dict[str, Union[float, str]]
    base_system: Optional[str] = None
    constraints: Optional[List[str]] = None

class DesignResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    suggested_compositions: List[Dict[str, Any]]
    design_rationale: str
    trade_offs: List[str]
    manufacturing_notes: str
    sources: List[SourceChunk]
    model_used: str
    retrieval_time_ms: float

# ─── Troubleshoot Endpoint ──────────────────────────────────────────────────

class RootCause(BaseModel):
    cause: str
    likelihood: str = "medium"
    explanation: str

class CorrectiveAction(BaseModel):
    action: str
    priority: str = "short-term"
    expected_outcome: str

class TroubleshootRequest(BaseModel):
    defect_description: str
    process_stage: Optional[str] = None
    glass_type: Optional[str] = None
    additional_context: Optional[str] = None

class TroubleshootResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    defect_classification: str
    root_causes: List[RootCause]
    corrective_actions: List[CorrectiveAction]
    preventive_measures: List[str]
    relevant_standards: List[str]
    sources: List[SourceChunk]
    model_used: str
    retrieval_time_ms: float

# ─── Feedback Endpoints ─────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: int = Field(..., ge=1, le=5)
    helpful: bool
    comment: Optional[str] = None
    query_id: str

class SourceFeedbackRequest(BaseModel):
    question: str
    source_title: str
    source_type: str
    relevant: bool
    comment: Optional[str] = None

class FeedbackResponse(BaseModel):
    success: bool
    message: str
    feedback_id: int

# ─── Conversations & Memory ─────────────────────────────────────────────────

class ConversationSummary(BaseModel):
    session_id: str
    title: str
    message_count: int
    last_message_at: str
    language: Optional[str] = None

class ConversationListResponse(BaseModel):
    conversations: List[ConversationSummary]

class ChatMessage(BaseModel):
    id: str
    role: str
    content: str
    sources: Optional[List[Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: str

class ConversationDetail(BaseModel):
    session_id: str
    title: str
    messages: List[ChatMessage]
    created_at: str
    last_message_at: str

class ConversationCreateRequest(BaseModel):
    title: Optional[str] = None

class ConversationCreateResponse(BaseModel):
    session_id: str
    title: str

class UserMemoryEntry(BaseModel):
    id: str
    memory_type: str
    key: str
    value: str
    updated_at: str

class UserMemoryListResponse(BaseModel):
    entries: List[UserMemoryEntry]

class UserMemorySaveRequest(BaseModel):
    memory_type: str = Field(..., description="Type: 'composition', 'furnace', 'topic', 'preference'")
    key: str
    value: str