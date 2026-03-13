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
  question: string
  answer: string
  helpful: boolean
  rating: number
  comment?: string
}

export interface SourceFeedbackPayload {
  question: string
  source_title: string
  source_type: string
  relevant: boolean
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
  feedbackSent?: boolean
  timestamp: Date
}
