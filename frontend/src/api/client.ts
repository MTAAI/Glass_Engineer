import axios from 'axios'
import type {
  QueryResponse, HealthResponse, FeedbackPayload, SourceFeedbackPayload,
  AuthToken, ConversationListResponse, ConversationDetail, ConversationCreateResponse,
  Conversation, ConversationSearchResponse, UserMemoryEntry, UserMemoryListResponse,
} from '../types'

const API = axios.create({
  baseURL: '/api/v1',
  timeout: 120_000,
})

// ── Auth token management ────────────────────────────────────────────────────

const TOKEN_KEY = 'glass_expert_token'

export function getStoredToken(): AuthToken | null {
  try {
    const raw = localStorage.getItem(TOKEN_KEY)
    return raw ? JSON.parse(raw) : null
  } catch {
    return null
  }
}

export function storeToken(token: AuthToken): void {
  localStorage.setItem(TOKEN_KEY, JSON.stringify(token))
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY)
}

// ── Auth interceptor — attach JWT to every request ───────────────────────────

API.interceptors.request.use((config) => {
  const token = getStoredToken()
  if (token?.access_token) {
    config.headers.Authorization = `Bearer ${token.access_token}`
  }
  return config
})

API.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      clearToken()
      window.dispatchEvent(new Event('auth:logout'))
    }
    return Promise.reject(err)
  }
)

// ── Auth endpoints ───────────────────────────────────────────────────────────

export async function login(email: string, password: string): Promise<AuthToken> {
  const form = new URLSearchParams()
  form.append('username', email) // OAuth2 form uses 'username'
  form.append('password', password)
  const { data } = await API.post<AuthToken>('/auth/login', form, {
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
  })
  storeToken(data)
  return data
}

export async function register(
  email: string,
  password: string,
  fullName?: string,
  languagePref: string = 'en'
): Promise<AuthToken> {
  const { data } = await API.post<AuthToken>('/auth/register', {
    email,
    password,
    full_name: fullName,
    language_pref: languagePref,
  })
  storeToken(data)
  return data
}

export function logout(): void {
  clearToken()
  window.dispatchEvent(new Event('auth:logout'))
}

// ── Existing endpoints ───────────────────────────────────────────────────────

export async function fetchHealth(): Promise<HealthResponse> {
  const { data } = await API.get<HealthResponse>('/health')
  return data
}

export async function queryKnowledgeBase(
  question: string,
  topK: number = 5,
  sourceType?: string,
  language?: string,
  sessionId?: string
): Promise<QueryResponse> {
  const payload: Record<string, unknown> = { question, top_k: topK }
  if (sourceType && sourceType !== 'all') payload.source_type = sourceType
  if (language && language !== 'auto') payload.language = language
  if (sessionId) payload.session_id = sessionId
  const { data } = await API.post<QueryResponse>('/query', payload)
  return data
}

// ── Conversations ─────────────────────────────────────────────────────────────

export async function createConversation(title?: string): Promise<ConversationCreateResponse> {
  const { data } = await API.post<ConversationCreateResponse>('/conversations', { title })
  return data
}

export async function listConversations(): Promise<Conversation[]> {
  const { data } = await API.get<ConversationListResponse>('/conversations')
  return data.conversations
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

export async function submitFeedback(payload: FeedbackPayload): Promise<{ feedback_id: string }> {
  const { data } = await API.post('/feedback', payload)
  return data
}

export async function submitSourceFeedback(payload: SourceFeedbackPayload): Promise<void> {
  await API.post('/feedback/source', payload)
}

// ── File Upload ──────────────────────────────────────────────────────────────

export interface UploadResponse {
  status: string
  file_name: string
  source_type: string
  chunks_stored: number
  message: string
}

// ── Admin Dashboard ─────────────────────────────────────────────────────────

export interface AdminStats {
  total_chunks: number
  chunks_by_type: { source_type: string; count: number }[]
  chunks_by_language: { language: string; count: number }[]
  total_users: number
  active_users_7d: number
  total_conversations: number
  total_messages: number
  messages_7d: number
  feedback: { total: number; positive: number; negative: number; satisfaction_pct: number }
  top_queries: { question: string; count: number }[]
  recent_ingestions: { file_name: string; source_type: string; language: string; chunk_count: number; status: string; created_at: string }[]
  model_usage: { model: string; count: number }[]
}

export async function fetchAdminStats(): Promise<AdminStats> {
  const { data } = await API.get<AdminStats>('/admin/stats')
  return data
}

// ── Export Conversation ──────────────────────────────────────────────────────

export async function exportConversation(sessionId: string): Promise<void> {
  const response = await API.get(`/export/${sessionId}`, { responseType: 'blob' })
  const url = window.URL.createObjectURL(new Blob([response.data]))
  const a = document.createElement('a')
  a.href = url
  const disposition = response.headers['content-disposition'] || ''
  const match = disposition.match(/filename="?(.+?)"?$/)
  a.download = match?.[1] || `glass_expert_report_${sessionId.slice(0, 8)}.docx`
  document.body.appendChild(a)
  a.click()
  a.remove()
  window.URL.revokeObjectURL(url)
}

// ── Conversation Search ──────────────────────────────────────────────────────

export async function searchConversations(query: string, limit: number = 20): Promise<ConversationSearchResponse> {
  const { data } = await API.get<ConversationSearchResponse>('/conversations/search', {
    params: { q: query, limit },
  })
  return data
}

// ── User Memory ─────────────────────────────────────────────────────────────

export async function getUserMemory(): Promise<UserMemoryEntry[]> {
  const { data } = await API.get<UserMemoryListResponse>('/user/memory')
  return data.entries
}

export async function saveUserMemory(
  key: string,
  value: string,
  memoryType: string = 'preference',
): Promise<void> {
  await API.post('/user/memory', { key, value, memory_type: memoryType })
}

export async function deleteUserMemory(key: string): Promise<void> {
  await API.delete(`/user/memory/${encodeURIComponent(key)}`)
}

// ── File Upload ──────────────────────────────────────────────────────────────

export async function uploadDocument(
  file: File,
  sourceType: string = 'paper',
): Promise<UploadResponse> {
  const form = new FormData()
  form.append('file', file)
  form.append('source_type', sourceType)
  const { data } = await API.post<UploadResponse>('/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 300_000, // 5 min for large files
  })
  return data
}
