import axios from 'axios'
import type { QueryResponse, HealthResponse, FeedbackPayload, SourceFeedbackPayload, Conversation, ConversationDetail } from '../types'

const API = axios.create({
  baseURL: '/api/v1',
  timeout: 120_000,
})

export async function fetchHealth(): Promise<HealthResponse> {
  const { data } = await API.get<HealthResponse>('/health')
  return data
}

export async function queryKnowledgeBase(
  question: string,
  topK: number = 5,
  sourceType?: string,
  language?: string,
  sessionId?: string,
): Promise<QueryResponse> {
  const payload: Record<string, unknown> = { question, top_k: topK }
  if (sourceType && sourceType !== 'all') payload.source_type = sourceType
  if (language && language !== 'auto') payload.language = language
  if (sessionId) payload.session_id = sessionId
  const { data } = await API.post<QueryResponse>('/query', payload)
  return data
}

export async function submitFeedback(payload: FeedbackPayload): Promise<void> {
  await API.post('/feedback', payload)
}

export async function submitSourceFeedback(payload: SourceFeedbackPayload): Promise<void> {
  await API.post('/feedback/source', payload)
}

// ── Conversations ─────────────────────────────────────────────────────────────

export async function createConversation(title?: string): Promise<{ session_id: string; title: string }> {
  const { data } = await API.post('/conversations', title ? { title } : {})
  return data
}

export async function listConversations(): Promise<Conversation[]> {
  const { data } = await API.get<Conversation[]>('/conversations')
  return data
}

export async function getConversation(sessionId: string): Promise<ConversationDetail> {
  const { data } = await API.get<ConversationDetail>(`/conversations/${sessionId}`)
  return data
}

export async function renameConversation(sessionId: string, title: string): Promise<void> {
  await API.patch(`/conversations/${sessionId}`, { title })
}

export async function deleteConversation(sessionId: string): Promise<void> {
  await API.delete(`/conversations/${sessionId}`)
}
