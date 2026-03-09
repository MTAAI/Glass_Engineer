import axios from 'axios'
import type { QueryResponse, HealthResponse, FeedbackPayload, SourceFeedbackPayload } from '../types'

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
  language?: string
): Promise<QueryResponse> {
  const payload: Record<string, unknown> = { question, top_k: topK }
  if (sourceType && sourceType !== 'all') payload.source_type = sourceType
  if (language && language !== 'auto') payload.language = language
  const { data } = await API.post<QueryResponse>('/query', payload)
  return data
}

export async function submitFeedback(payload: FeedbackPayload): Promise<void> {
  await API.post('/feedback', payload)
}

export async function submitSourceFeedback(payload: SourceFeedbackPayload): Promise<void> {
  await API.post('/feedback/source', payload)
}
