"""
Glass Expert AI - Pydantic Schemas

Defines the data models for API requests and responses.
This file centralizes all data structures used across the application for
consistency, validation, and clear API documentation.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional, Union

# ─── Base & Common Models ───────────────────────────────────────────────────

class SourceChunk(BaseModel):
    """Represents a single chunk of a source document used for context."""
    title: str = Field(..., description="Title of the source document.")
    source_type: str = Field(..., description="Type of the source (e.g., 'textbook', 'paper').")
    language: str = Field(..., description="Language of the chunk content.")
    similarity: float = Field(..., description="Similarity score of the chunk to the query.")
    content_preview: str = Field(..., description="A short preview of the chunk's content.")

# ─── Query Endpoint ─────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request model for the main /query endpoint."""
    question: str = Field(..., min_length=5, description="The user's question about glass science.")
    language: str = Field("auto", description="Language for the answer ('en', 'fa', or 'auto').")
    mode: str = Field("simple", description="Query mode: 'simple', 'detailed', or 'research'.")
    top_k: int = Field(5, ge=1, le=20, description="Number of source chunks to retrieve.")
    source_type: Optional[str] = Field(None, description="Filter by source type (e.g. 'textbook', 'paper').")

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

# ─── Health Endpoint ────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response model for the /health endpoint."""
    status: str
    database: str
    redis: str
    total_documents: int
    total_chunks: int

# ─── Ingest Endpoint ────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    """Request model for the /ingest endpoint."""
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Full text content of the document")
    source_type: str = Field("textbook", description="Type: textbook, paper, sop, standard, datasheet")
    language: str = Field("en", description="Language code: en or fa")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class IngestResponse(BaseModel):
    """Response model for the /ingest endpoint."""
    success: bool
    message: str
    chunks_created: int
    document_title: str

# ─── Analyze Endpoint ───────────────────────────────────────────────────────

class PropertyPrediction(BaseModel):
    """A structured prediction for a single glass property."""
    property_name: str = Field(..., description="Name of the predicted property (e.g., 'Glass Transition Temperature (Tg)').")
    value: str = Field(..., description="Predicted value or range.")
    unit: Optional[str] = Field(None, description="Unit of the predicted value (e.g., '°C').")
    confidence: str = Field("medium", description="Confidence level of the prediction (high/medium/low).")
    source: str = Field("LLM analysis", description="Source of the prediction (e.g., 'LLM analysis', 'Composition analysis').")

class AnalyzeRequest(BaseModel):
    """Request model for the /analyze endpoint."""
    composition: Optional[Dict[str, float]] = Field(None, description="Glass composition as a dictionary of oxide weight percentages.")
    description: Optional[str] = Field(None, description="Free-text description of the glass sample or problem.")
    properties_of_interest: Optional[List[str]] = Field(None, description="Specific properties the user is interested in.")

class AnalyzeResponse(BaseModel):
    """Response model for the /analyze endpoint."""
    model_config = ConfigDict(protected_namespaces=())

    composition_summary: Optional[str] = Field(None, description="A brief summary of the glass system.")
    analysis: str = Field(..., description="The full technical analysis text generated by the LLM.")
    predicted_properties: List[PropertyPrediction] = Field(..., description="A structured list of predicted properties.")
    recommendations: List[str] = Field(..., description="A list of actionable recommendations.")
    sources: List[SourceChunk] = Field(..., description="List of source chunks used for the analysis.")
    model_used: str = Field(..., description="The language model used for the analysis.")
    retrieval_time_ms: float = Field(..., description="Time taken for document retrieval in milliseconds.")

# ─── Design Endpoint ────────────────────────────────────────────────────────

class DesignRequest(BaseModel):
    """Request model for the /design endpoint."""
    target_properties: Dict[str, Union[float, str]] = Field(..., description="Dictionary of target properties and their desired values/ranges.")
    base_system: Optional[str] = Field(None, description="The base glass system to start from (e.g., 'soda-lime-silica').")
    constraints: Optional[List[str]] = Field(None, description="A list of constraints for the design (e.g., 'must be lead-free').")

class DesignResponse(BaseModel):
    """Response model for the /design endpoint."""
    model_config = ConfigDict(protected_namespaces=())

    suggested_compositions: List[Dict[str, Any]] = Field(..., description="A list of suggested glass compositions (oxide wt%).")
    design_rationale: str = Field(..., description="The full text explaining the design choices.")
    trade_offs: List[str] = Field(..., description="A list of trade-offs considered in the design.")
    manufacturing_notes: str = Field(..., description="Notes on manufacturing considerations for the suggested compositions.")
    sources: List[SourceChunk] = Field(..., description="List of source chunks used for the design.")
    model_used: str = Field(..., description="The language model used for the design.")
    retrieval_time_ms: float = Field(..., description="Time taken for document retrieval in milliseconds.")

# ─── Troubleshoot Endpoint ──────────────────────────────────────────────────

class RootCause(BaseModel):
    """A potential root cause for a manufacturing defect."""
    cause: str = Field(..., description="The description of the root cause.")
    likelihood: str = Field("medium", description="Likelihood of this being the true cause (high/medium/low).")
    explanation: str = Field(..., description="Explanation of how this cause leads to the defect.")

class CorrectiveAction(BaseModel):
    """A corrective action to address a root cause."""
    action: str = Field(..., description="The specific corrective action to be taken.")
    priority: str = Field("short-term", description="Priority of the action (immediate/short-term/long-term).")
    expected_outcome: str = Field(..., description="The expected outcome of implementing the action.")

class TroubleshootRequest(BaseModel):
    """Request model for the /troubleshoot endpoint."""
    defect_description: str = Field(..., description="Detailed description of the observed defect.")
    process_stage: Optional[str] = Field(None, description="The manufacturing stage where the defect is observed.")
    glass_type: Optional[str] = Field(None, description="The type of glass being produced.")
    additional_context: Optional[str] = Field(None, description="Any other relevant context or recent changes.")

class TroubleshootResponse(BaseModel):
    """Response model for the /troubleshoot endpoint."""
    model_config = ConfigDict(protected_namespaces=())

    defect_classification: str = Field(..., description="The classified type of the defect.")
    root_causes: List[RootCause] = Field(..., description="A ranked list of potential root causes.")
    corrective_actions: List[CorrectiveAction] = Field(..., description="A prioritized list of corrective actions.")
    preventive_measures: List[str] = Field(..., description="A list of measures to prevent recurrence.")
    relevant_standards: List[str] = Field(..., description="A list of relevant industry standards (e.g., ASTM, ISO).")
    sources: List[SourceChunk] = Field(..., description="List of source chunks used for the analysis.")
    model_used: str = Field(..., description="The language model used for the analysis.")
    retrieval_time_ms: float = Field(..., description="Time taken for document retrieval in milliseconds.")

# ─── Feedback Endpoints ─────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    """Request model for submitting overall answer feedback."""
    question: str
    answer: str
    rating: int = Field(..., ge=1, le=5)
    helpful: bool
    comment: Optional[str] = None
    query_id: str

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
