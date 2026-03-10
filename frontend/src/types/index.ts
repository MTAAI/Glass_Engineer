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

export interface Conversation {
  session_id: string
  title: string
  created_at: string
  updated_at: string
  message_count: number
}

export interface ConversationDetail {
  session_id: string
  title: string
  created_at: string
  updated_at: string
  messages: {
    id: string
    role: string
    content: string
    sources?: Record<string, unknown>[]
    metadata?: Record<string, unknown>
    created_at: string
  }[]
}
