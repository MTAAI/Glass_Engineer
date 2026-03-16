export interface SourceChunk {
  title: string
  source_type: string
  language: string
  similarity: number
  content_preview: string
  rerank_score?: number
}

export interface QueryResponse {
  answer: string
  sources: SourceChunk[]
  language_detected: string
  model_used: string
  retrieval_time_ms: number
  total_chunks_searched: number
  session_id?: string
  chat_id?: string  // UUID of assistant message in chat_history (for feedback)
}

// ── Conversations ─────────────────────────────────────────────────────────────

export interface ConversationSummary {
  session_id: string
  title: string
  message_count: number
  last_message_at: string
}

export interface ConversationListResponse {
  conversations: ConversationSummary[]
}

export interface HealthResponse {
  status: string
  database: string
  redis: string
  embedding_model: string
  total_documents: number
  total_chunks: number
  version: string
}

export interface FeedbackPayload {
  chat_id: string   // UUID of the assistant message in chat_history
  rating: 1 | -1    // thumbs up (+1) or thumbs down (-1)
  corrected_text?: string
  comment?: string
}

export interface SourceFeedbackPayload {
  feedback_id: string   // UUID of the parent feedback entry
  document_id: string   // UUID of the document being rated
  is_relevant: boolean
}

// ── Auth ──────────────────────────────────────────────────────────────────────
export interface AuthToken {
  access_token: string
  token_type: string
  user_id: string
  email: string
  role: string
  full_name?: string
}

export interface AuthUser {
  id: string
  email: string
  full_name?: string
  role: string
  plant_location?: string
  language_pref: string
  is_active: boolean
}

export type MessageRole = 'user' | 'assistant'

export interface Message {
  id: string
  role: MessageRole
  content: string
  sources?: SourceChunk[]
  meta?: {
    retrieval_time_ms: number
    language_detected: string
    model_used: string
    total_chunks_searched: number
  }
  chatId?: string  // UUID from chat_history (for feedback)
  feedbackSent?: boolean
  timestamp: Date
}
