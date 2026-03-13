import axios from 'axios'
import type { QueryResponse, HealthResponse, FeedbackPayload, SourceFeedbackPayload, AuthToken, ConversationListResponse } from '../types'

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

export async function listConversations(): Promise<ConversationListResponse> {
  const { data } = await API.get<ConversationListResponse>('/conversations')
  return data
}

export async function deleteConversation(sessionId: string): Promise<void> {
  await API.delete(`/conversations/${sessionId}`)
}

export async function submitFeedback(payload: FeedbackPayload): Promise<void> {
  await API.post('/feedback', payload)
}

export async function submitSourceFeedback(payload: SourceFeedbackPayload): Promise<void> {
  await API.post('/feedback/source', payload)
}
