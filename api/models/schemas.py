"""
Glass Expert AI - Pydantic Schemas

Defines the data models for API requests and responses.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional, Union

# --- Base & Common Models ---------------------------------------------------

class SourceChunk(BaseModel):
    """Represents a single chunk of a source document used for context."""
    title: str = Field(..., description="Title of the source document.")
    source_type: str = Field(..., description="Type of the source (e.g., 'textbook', 'paper').")
    language: str = Field(..., description="Language of the chunk content.")
    similarity: float = Field(..., description="Similarity score of the chunk to the query.")
    content_preview: str = Field(..., description="A short preview of the chunk's content.")

# --- Query Endpoint ---------------------------------------------------------

class QueryRequest(BaseModel):
    """Request model for the main /query endpoint."""
    question: str = Field(..., min_length=5, description="The user's question about glass science.")
    language: str = Field("auto", description="Language for the answer ('en', 'fa', or 'auto').")
    mode: str = Field("simple", description="Query mode: 'simple', 'detailed', or 'research'.")
    top_k: int = Field(5, ge=1, le=20, description="Number of source chunks to retrieve.")
    source_type: Optional[str] = Field(None, description="Filter by source type (e.g. 'textbook', 'paper').")
    session_id: Optional[str] = Field(None, description="Conversation session ID for multi-turn context.")

class QueryResponse(BaseModel):
    """Response model for the /query endpoint."""
    model_config = ConfigDict(protected_namespaces=())

    question: str = Field(..., description="The original question asked.")
    answer: str = Field(..., description="The generated answer to the user's question.")
    sources: List[SourceChunk] = Field(..., description="List of source chunks used for the answer.")
    query_id: Optional[str] = Field(None, description="Unique identifier for this query.")
    model_used: str = Field(..., description="The language model used to generate the answer.")
    retrieval_time_ms: float = Field(..., description="Time taken for document retrieval in milliseconds.")
    generation_time_ms: Optional[float] = Field(None, description="Time taken for answer generation in milliseconds.")
    language_detected: Optional[str] = Field(None, description="Detected language of the query.")
    total_chunks_searched: Optional[int] = Field(None, description="Total number of chunks retrieved.")
    session_id: Optional[str] = Field(None, description="Conversation session ID.")

# --- Health Endpoint --------------------------------------------------------

class HealthResponse(BaseModel):
    """Response model for the /health endpoint."""
    status: str
    database: str
    redis: str
    embedding_model: str
    total_documents: int
    total_chunks: int
    version: str

# --- Ingest Endpoint --------------------------------------------------------

class IngestRequest(BaseModel):
    """Request model for the /ingest endpoint."""
    file_path: str = Field(..., description="Path to the file to ingest.")
    source_type: str = Field("textbook", description="Type: textbook, paper, sop, standard, manual, qa_pair")

class IngestResponse(BaseModel):
    """Response model for the /ingest endpoint."""
    file_name: str
    source_type: str
    chunks_stored: int
    language: str
    status: str
    message: str

# --- Analyze Endpoint -------------------------------------------------------

class PropertyPrediction(BaseModel):
    """A structured prediction for a single glass property."""
    property_name: str = Field(..., description="Name of the predicted property.")
    value: str = Field(..., description="Predicted value or range.")
    unit: Optional[str] = Field(None, description="Unit of the predicted value.")
    confidence: str = Field("medium", description="Confidence level (high/medium/low).")
    source: str = Field("LLM analysis", description="Source of the prediction.")

class AnalyzeRequest(BaseModel):
    """Request model for the /analyze endpoint."""
    composition: Optional[Dict[str, float]] = Field(None, description="Glass composition as oxide weight percentages.")
    description: Optional[str] = Field(None, description="Free-text description of the glass sample.")
    properties_of_interest: Optional[List[str]] = Field(None, description="Specific properties of interest.")

class AnalyzeResponse(BaseModel):
    """Response model for the /analyze endpoint."""
    model_config = ConfigDict(protected_namespaces=())

    composition_summary: Optional[str] = Field(None, description="Brief summary of the glass system.")
    analysis: str = Field(..., description="Full technical analysis text.")
    predicted_properties: List[PropertyPrediction] = Field(..., description="Structured list of predicted properties.")
    recommendations: List[str] = Field(..., description="Actionable recommendations.")
    sources: List[SourceChunk] = Field(..., description="Source chunks used for analysis.")
    model_used: str = Field(..., description="Language model used.")
    retrieval_time_ms: float = Field(..., description="Retrieval time in milliseconds.")

# --- Design Endpoint --------------------------------------------------------

class DesignRequest(BaseModel):
    """Request model for the /design endpoint."""
    target_properties: Dict[str, Union[float, str]] = Field(..., description="Target properties and desired values.")
    base_system: Optional[str] = Field(None, description="Base glass system (e.g., 'soda-lime-silica').")
    constraints: Optional[List[str]] = Field(None, description="Design constraints (e.g., 'must be lead-free').")

class DesignResponse(BaseModel):
    """Response model for the /design endpoint."""
    model_config = ConfigDict(protected_namespaces=())

    suggested_compositions: List[Dict[str, Any]] = Field(..., description="Suggested glass compositions.")
    design_rationale: str = Field(..., description="Explanation of design choices.")
    trade_offs: List[str] = Field(..., description="Trade-offs considered.")
    manufacturing_notes: str = Field(..., description="Manufacturing considerations.")
    sources: List[SourceChunk] = Field(..., description="Source chunks used.")
    model_used: str = Field(..., description="Language model used.")
    retrieval_time_ms: float = Field(..., description="Retrieval time in milliseconds.")

# --- Troubleshoot Endpoint --------------------------------------------------

class RootCause(BaseModel):
    """A potential root cause for a manufacturing defect."""
    cause: str = Field(..., description="Description of the root cause.")
    likelihood: str = Field("medium", description="Likelihood (high/medium/low).")
    explanation: str = Field(..., description="How this cause leads to the defect.")

class CorrectiveAction(BaseModel):
    """A corrective action to address a root cause."""
    action: str = Field(..., description="Specific corrective action.")
    priority: str = Field("short-term", description="Priority (immediate/short-term/long-term).")
    expected_outcome: str = Field(..., description="Expected outcome of the action.")

class TroubleshootRequest(BaseModel):
    """Request model for the /troubleshoot endpoint."""
    defect_description: str = Field(..., description="Description of the observed defect.")
    process_stage: Optional[str] = Field(None, description="Manufacturing stage where defect is observed.")
    glass_type: Optional[str] = Field(None, description="Type of glass being produced.")
    additional_context: Optional[str] = Field(None, description="Other relevant context.")

class TroubleshootResponse(BaseModel):
    """Response model for the /troubleshoot endpoint."""
    model_config = ConfigDict(protected_namespaces=())

    defect_classification: str = Field(..., description="Classified defect type.")
    root_causes: List[RootCause] = Field(..., description="Ranked list of potential root causes.")
    corrective_actions: List[CorrectiveAction] = Field(..., description="Prioritized corrective actions.")
    preventive_measures: List[str] = Field(..., description="Measures to prevent recurrence.")
    relevant_standards: List[str] = Field(..., description="Applicable industry standards.")
    sources: List[SourceChunk] = Field(..., description="Source chunks used.")
    model_used: str = Field(..., description="Language model used.")
    retrieval_time_ms: float = Field(..., description="Retrieval time in milliseconds.")

# --- Feedback Endpoints -----------------------------------------------------

class FeedbackRequest(BaseModel):
    """Request model for submitting overall answer feedback."""
    question: str
    answer: str
    rating: int = Field(..., ge=1, le=5)
    helpful: bool
    comment: Optional[str] = None
    query_id: Optional[str] = None

class SourceFeedbackRequest(BaseModel):
    """Request model for submitting feedback on a specific source."""
    question: str
    source_title: str
    source_type: str
    relevant: bool
    comment: Optional[str] = None

class FeedbackResponse(BaseModel):
    """Generic response for feedback submission."""
    success: bool
    message: str
    feedback_id: int

# --- Conversation Endpoints ---------------------------------------------------

class ChatMessage(BaseModel):
    """A single message in a conversation."""
    id: str
    role: str
    content: str
    sources: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    created_at: str

class ConversationSummary(BaseModel):
    """Summary of a conversation for sidebar listing."""
    session_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0

class ConversationDetail(BaseModel):
    """Full conversation with messages."""
    session_id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[ChatMessage] = Field(default_factory=list)

class CreateConversationRequest(BaseModel):
    """Request model for creating a new conversation."""
    title: Optional[str] = Field(None, description="Optional title. Auto-generated from first message if omitted.")

class CreateConversationResponse(BaseModel):
    """Response for conversation creation."""
    session_id: str
    title: str

class RenameConversationRequest(BaseModel):
    """Request model for renaming a conversation."""
    title: str = Field(..., min_length=1, max_length=200)
