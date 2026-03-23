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

/** Alias used in App.tsx sidebar. */
export type Conversation = ConversationSummary

export interface ConversationDetail {
  session_id: string
  title: string
  messages: ChatMessage[]
  created_at: string
  last_message_at: string
}

export interface ChatMessage {
  id: string
  role: string
  content: string
  sources?: SourceChunk[]
  metadata?: Record<string, unknown>
  created_at: string
}

export interface ConversationCreateResponse {
  session_id: string
  title: string
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

// ── User Memory ──────────────────────────────────────────────────────────────

export interface UserMemoryEntry {
  id: string
  memory_type: string
  key: string
  value: string
  updated_at: string
}

export interface UserMemoryListResponse {
  entries: UserMemoryEntry[]
}

// ── Conversation Search ──────────────────────────────────────────────────────

export interface ConversationSearchResult {
  session_id: string
  role: string
  snippet: string
  created_at: string
  title: string
}

export interface ConversationSearchResponse {
  query: string
  results: ConversationSearchResult[]
  total: number
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
