import { useState, useEffect, useRef, useCallback } from 'react'
import { v4 as uuidv4 } from 'uuid'
import ReactMarkdown from 'react-markdown'
import {
  Microscope, Send, Trash2, ChevronDown, ChevronUp,
  ThumbsUp, ThumbsDown, CheckCircle, AlertCircle,
  BookOpen, FileText, FlaskConical, Layers, Star, LogOut,
  Sparkles, ShieldCheck, Database, Globe2, Gauge, Settings2,
  Plus, MessageSquare, X, Menu
} from 'lucide-react'
import {
  fetchHealth,
  queryKnowledgeBase,
  submitFeedback,
  submitSourceFeedback,
  login,
  register,
  logout,
  getStoredToken,
  listConversations,
  deleteConversation
} from './api/client'
import type { Message, SourceChunk, HealthResponse, AuthToken, ConversationSummary } from './types'

// ── Utility ────────────────────────────────────────────────────────────────
function formatMs(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`
  return `${(ms / 1000).toFixed(1)}s`
}
function langFlag(lang: string): string {
  return lang === 'fa' ? '🇮🇷' : '🇬🇧'
}
function isRtlText(text: string): boolean {
  const rtlChars = text.match(/[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]/g)
  return !!rtlChars && rtlChars.length > text.length * 0.3
}
function sourceIcon(type: string) {
  switch (type) {
    case 'textbook': return <BookOpen size={13} className="text-blue-300" />
    case 'paper': return <FileText size={13} className="text-fuchsia-300" />
    case 'sop': return <Layers size={13} className="text-amber-300" />
    case 'standard': return <Star size={13} className="text-orange-300" />
    default: return <FlaskConical size={13} className="text-cyan-300" />
  }
}
function timeAgo(dateStr: string): string {
  const d = new Date(dateStr)
  const now = new Date()
  const diffMs = now.getTime() - d.getTime()
  const mins = Math.floor(diffMs / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  const days = Math.floor(hrs / 24)
  return `${days}d ago`
}
function panelClass(extra = '') {
  return `glass-panel rounded-2xl ${extra}`
}

// ── Ambient Shell ──────────────────────────────────────────────────────────
function AmbientShell() {
  return (
    <>
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute -left-20 top-10 h-72 w-72 rounded-full bg-blue-500/10 blur-3xl" />
        <div className="absolute right-0 top-0 h-80 w-80 rounded-full bg-cyan-400/10 blur-3xl" />
        <div className="absolute bottom-0 left-1/3 h-96 w-96 rounded-full bg-indigo-500/10 blur-3xl" />
      </div>
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(to_right,rgba(255,255,255,0.025)_1px,transparent_1px),linear-gradient(to_bottom,rgba(255,255,255,0.025)_1px,transparent_1px)] bg-[size:32px_32px] opacity-[0.08]" />
    </>
  )
}

// ── Small UI Blocks ────────────────────────────────────────────────────────
function StatPill({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode
  label: string
  value: string
}) {
  return (
    <div className="flex items-center gap-2 rounded-xl border border-slate-700/60 bg-slate-900/40 px-3 py-2">
      <div className="text-blue-300">{icon}</div>
      <div className="min-w-0">
        <div className="text-[10px] uppercase tracking-[0.16em] text-slate-500">{label}</div>
        <div className="truncate text-xs font-medium text-slate-200">{value}</div>
      </div>
    </div>
  )
}

// ── Source Card ─────────────────────────────────────────────────────────────
interface SourceCardProps {
  source: SourceChunk
  index: number
  question: string
}
function SourceCard({ source, index, question }: SourceCardProps) {
  const [voted, setVoted] = useState<'up' | 'down' | null>(null)
  const pct = Math.round(source.similarity * 100)
  const handleVote = async (relevant: boolean) => {
    setVoted(relevant ? 'up' : 'down')
    try {
      await submitSourceFeedback({
        question,
        source_title: source.title,
        source_type: source.source_type,
        relevant,
      })
    } catch { /* silent */ }
  }
  return (
    <div className="rounded-2xl border border-slate-700/70 bg-slate-950/40 p-4 shadow-[0_10px_25px_rgba(2,6,23,0.22)]">
      <div className="mb-2 flex items-start gap-3">
        <div className="mt-0.5 rounded-lg bg-slate-900/70 p-2 ring-1 ring-slate-700/70">
          {sourceIcon(source.source_type)}
        </div>
        <div className="min-w-0 flex-1">
          <div className="mb-1 flex items-center gap-2">
            <span className="rounded-full bg-blue-500/10 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.16em] text-blue-300">
              Source {index + 1}
            </span>
            <span className="text-[10px] text-slate-500">{langFlag(source.language)}</span>
          </div>
          <div className="truncate text-sm font-semibold text-slate-100">{source.title}</div>
          <div className="mt-1 flex flex-wrap gap-2 text-[11px] text-slate-400">
            <span className="rounded-full bg-slate-900/60 px-2 py-1">{source.source_type}</span>
            <span className="rounded-full bg-slate-900/60 px-2 py-1">{pct}% similarity</span>
          </div>
        </div>
      </div>
      <div className="mb-3 h-1.5 rounded-full bg-slate-900">
        <div
          className="h-1.5 rounded-full bg-gradient-to-r from-blue-500 via-cyan-400 to-sky-300"
          style={{ width: `${pct}%` }}
        />
      </div>
      <p className="text-xs leading-6 text-slate-400">
        {source.content_preview}
      </p>
      <div className="mt-3 flex items-center gap-3 border-t border-slate-800/70 pt-3">
        <span className="text-[11px] text-slate-500">Was this source useful?</span>
        {voted === null ? (
          <>
            <button
              onClick={() => handleVote(true)}
              className="rounded-lg p-2 text-slate-400 transition hover:bg-emerald-500/10 hover:text-emerald-300"
              title="Relevant"
            >
              <ThumbsUp size={14} />
            </button>
            <button
              onClick={() => handleVote(false)}
              className="rounded-lg p-2 text-slate-400 transition hover:bg-red-500/10 hover:text-red-300"
              title="Not relevant"
            >
              <ThumbsDown size={14} />
            </button>
          </>
        ) : (
          <span className="flex items-center gap-1 text-xs text-emerald-400">
            <CheckCircle size={12} />
            Recorded
          </span>
        )}
      </div>
    </div>
  )
}

// ── Citations Panel ─────────────────────────────────────────────────────────
interface CitationsPanelProps {
  sources: SourceChunk[]
  question: string
}
function CitationsPanel({ sources, question }: CitationsPanelProps) {
  const [open, setOpen] = useState(false)
  if (!sources?.length) return null
  return (
    <div className="mt-4 rounded-2xl border border-slate-800/70 bg-slate-950/30 p-3">
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between gap-3 text-left"
      >
        <div className="flex items-center gap-2">
          <div className="rounded-lg bg-blue-500/10 p-2 text-blue-300">
            <BookOpen size={14} />
          </div>
          <div>
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
              Evidence
            </div>
            <div className="text-sm text-slate-200">
              {sources.length} source{sources.length > 1 ? 's' : ''} retrieved
            </div>
          </div>
        </div>
        <div className="text-slate-400">
          {open ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </div>
      </button>
      {open && (
        <div className="mt-4 space-y-3">
          {sources.map((src, i) => (
            <SourceCard key={`${src.title}-${i}`} source={src} index={i} question={question} />
          ))}
        </div>
      )}
    </div>
  )
}

// ── Feedback Bar ────────────────────────────────────────────────────────────
interface FeedbackBarProps {
  chatId?: string  // UUID from chat_history — required for feedback
}
function FeedbackBar({ chatId }: FeedbackBarProps) {
  const [sent, setSent] = useState<'helpful' | 'not_helpful' | null>(null)
  const handleFeedback = async (helpful: boolean) => {
    if (!chatId) return  // can't submit feedback without a chat_id
    setSent(helpful ? 'helpful' : 'not_helpful')
    try {
      await submitFeedback({
        chat_id: chatId,
        rating: helpful ? 1 : -1,
      })
    } catch { /* silent */ }
  }
  if (sent !== null) {
    return (
      <div className="mt-4 flex items-center gap-2 rounded-xl border border-emerald-500/20 bg-emerald-500/8 px-3 py-2 text-xs text-emerald-300">
        <CheckCircle size={13} />
        <span>{sent === 'helpful' ? 'Thanks — that helps us improve.' : 'Feedback recorded for review.'}</span>
      </div>
    )
  }
  return (
    <div className="mt-4 flex items-center gap-3 border-t border-slate-800/70 pt-3">
      <span className="text-xs text-slate-500">Was this answer useful?</span>
      <button
        onClick={() => handleFeedback(true)}
        className="flex items-center gap-1 rounded-lg px-2.5 py-1.5 text-xs text-slate-400 transition hover:bg-emerald-500/10 hover:text-emerald-300"
      >
        <ThumbsUp size={13} /> Yes
      </button>
      <button
        onClick={() => handleFeedback(false)}
        className="flex items-center gap-1 rounded-lg px-2.5 py-1.5 text-xs text-slate-400 transition hover:bg-red-500/10 hover:text-red-300"
      >
        <ThumbsDown size={13} /> No
      </button>
    </div>
  )
}

// ── Message Bubble ──────────────────────────────────────────────────────────
interface MessageBubbleProps {
  message: Message
  prevQuestion?: string
}
function MessageBubble({ message, prevQuestion }: MessageBubbleProps) {
  const isUser = message.role === 'user'
  const rtl = isRtlText(message.content)
  const isResponseRtl = !isUser && message.meta?.language_detected === 'fa'
  const dirProps = (rtl || isResponseRtl) ? { dir: 'rtl' as const } : {}
  if (isUser) {
    return (
      <div className={`mb-5 flex ${rtl ? 'justify-start' : 'justify-end'}`}>
        <div
          {...dirProps}
          className={`max-w-[82%] rounded-3xl border border-blue-400/20 bg-gradient-to-br from-blue-600/80 to-sky-700/70 px-5 py-4 text-sm leading-7 text-white shadow-[0_15px_35px_rgba(37,99,235,0.22)] ${rtl ? 'rounded-tl-md text-right' : 'rounded-tr-md text-left'}`}
        >
          {message.content}
        </div>
      </div>
    )
  }
  return (
    <div className={`mb-5 flex ${isResponseRtl ? 'justify-end' : 'justify-start'}`}>
      <div
        {...dirProps}
        className={`max-w-[88%] rounded-3xl border border-slate-700/70 bg-slate-950/45 px-5 py-4 shadow-[0_15px_35px_rgba(2,6,23,0.25)] ${isResponseRtl ? 'rounded-tr-md text-right' : 'rounded-tl-md text-left'}`}
      >
        <div className="mb-3 flex items-center gap-2">
          <div className="rounded-xl bg-blue-500/10 p-2 text-blue-300 ring-1 ring-blue-500/20">
            <Microscope size={14} />
          </div>
          <div>
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
              Glass Expert AI
            </div>
            <div className="text-[11px] text-slate-600">
              {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </div>
          </div>
        </div>
        <div className="prose prose-invert prose-sm max-w-none text-sm leading-7 text-slate-200">
          <ReactMarkdown>{message.content}</ReactMarkdown>
        </div>
        {message.meta && (
          <div className="mt-4 flex flex-wrap gap-2">
            <span className="rounded-full border border-slate-700/70 bg-slate-900/70 px-2.5 py-1 text-[11px] text-slate-400">
              ⏱ {formatMs(message.meta.retrieval_time_ms)}
            </span>
            <span className="rounded-full border border-slate-700/70 bg-slate-900/70 px-2.5 py-1 text-[11px] text-slate-400">
              🌐 {message.meta.language_detected?.toUpperCase()}
            </span>
            <span className="rounded-full border border-slate-700/70 bg-slate-900/70 px-2.5 py-1 text-[11px] text-slate-400">
              🤖 {message.meta.model_used}
            </span>
            <span className="rounded-full border border-slate-700/70 bg-slate-900/70 px-2.5 py-1 text-[11px] text-slate-400">
              📚 {message.meta.total_chunks_searched} chunks
            </span>
          </div>
        )}
        {message.sources && message.sources.length > 0 && (
          <CitationsPanel sources={message.sources} question={prevQuestion || message.content} />
        )}
        {message.content && message.content !== 'No answer generated.' && (
          <FeedbackBar chatId={message.chatId} />
        )}
      </div>
    </div>
  )
}

// ── Health Badge ─────────────────────────────────────────────────────────────
function HealthBadge({ health }: { health: HealthResponse | null }) {
  if (!health) {
    return (
      <div className="flex items-center gap-2 rounded-xl border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-300">
        <AlertCircle size={14} />
        API offline
      </div>
    )
  }
  const ok = health.status === 'healthy'
  return (
    <div className={`flex items-center gap-2 rounded-xl border px-3 py-2 text-xs ${ok ? 'border-emerald-500/20 bg-emerald-500/10 text-emerald-300' : 'border-amber-500/20 bg-amber-500/10 text-amber-300'}`}>
      <CheckCircle size={14} />
      <span>{ok ? 'Healthy' : health.status}</span>
    </div>
  )
}

// ── Login Screen ─────────────────────────────────────────────────────────────
function LoginScreen({ onAuth }: { onAuth: (token: AuthToken) => void }) {
  const [isRegister, setIsRegister] = useState(false)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [fullName, setFullName] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const token = isRegister
        ? await register(email, password, fullName || undefined)
        : await login(email, password)
      onAuth(token)
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(msg || (isRegister ? 'Registration failed' : 'Invalid email or password'))
    } finally {
      setLoading(false)
    }
  }
  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden px-4">
      <AmbientShell />
      <div className="grid w-full max-w-6xl gap-8 lg:grid-cols-[1.2fr_0.9fr]">
        <div className="hidden flex-col justify-center lg:flex">
          <div className="max-w-2xl">
            <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-blue-400/20 bg-blue-500/10 px-4 py-2 text-xs font-semibold uppercase tracking-[0.18em] text-blue-300">
              <Sparkles size={14} />
              Glass Engineering Intelligence
            </div>
            <h1 className="mb-4 text-5xl font-extrabold leading-tight text-white">
              The most elegant way to search,
              <span className="bg-gradient-to-r from-blue-300 via-cyan-300 to-sky-400 bg-clip-text text-transparent"> reason, and cite</span>
              {' '}glass knowledge.
            </h1>
            <p className="mb-8 max-w-xl text-base leading-8 text-slate-400">
              Ask in English or فارسی. Retrieve grounded knowledge from books, papers, SOPs, and technical references — with a premium research workflow designed for engineers.
            </p>
            <div className="grid grid-cols-2 gap-4">
              <div className={panelClass('p-4')}>
                <div className="mb-2 text-blue-300"><Database size={18} /></div>
                <div className="text-sm font-semibold text-white">Grounded answers</div>
                <div className="mt-1 text-xs leading-6 text-slate-400">Every response is tied to retrieved technical evidence.</div>
              </div>
              <div className={panelClass('p-4')}>
                <div className="mb-2 text-cyan-300"><Globe2 size={18} /></div>
                <div className="text-sm font-semibold text-white">Bilingual workflow</div>
                <div className="mt-1 text-xs leading-6 text-slate-400">English + Persian interaction with source-backed responses.</div>
              </div>
              <div className={panelClass('p-4')}>
                <div className="mb-2 text-emerald-300"><ShieldCheck size={18} /></div>
                <div className="text-sm font-semibold text-white">Private by design</div>
                <div className="mt-1 text-xs leading-6 text-slate-400">Local-first architecture with policy-based fallback controls.</div>
              </div>
              <div className={panelClass('p-4')}>
                <div className="mb-2 text-fuchsia-300"><Gauge size={18} /></div>
                <div className="text-sm font-semibold text-white">Engineering speed</div>
                <div className="mt-1 text-xs leading-6 text-slate-400">Ask, analyze, design, and troubleshoot without breaking flow.</div>
              </div>
            </div>
          </div>
        </div>
        <div className={`${panelClass('mx-auto w-full max-w-md p-8 soft-ring')} relative`}>
          <div className="mb-8 text-center">
            <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-500/20 to-cyan-400/20 ring-1 ring-blue-400/20">
              <Microscope size={28} className="text-blue-300" />
            </div>
            <h2 className="text-2xl font-bold text-white">Glass Expert AI</h2>
            <p className="mt-2 text-sm text-slate-400">
              {isRegister ? 'Create your account to start exploring the knowledge base.' : 'Sign in to continue your research workspace.'}
            </p>
          </div>
          <form onSubmit={handleSubmit} className="space-y-4">
            {isRegister && (
              <input
                type="text"
                placeholder="Full name"
                value={fullName}
                onChange={e => setFullName(e.target.value)}
                className="w-full rounded-xl border border-slate-700/70 bg-slate-950/40 px-4 py-3 text-sm text-slate-200 placeholder-slate-600 outline-none transition focus:border-blue-500"
              />
            )}
            <input
              type="email"
              placeholder="Email address"
              value={email}
              onChange={e => setEmail(e.target.value)}
              required
              className="w-full rounded-xl border border-slate-700/70 bg-slate-950/40 px-4 py-3 text-sm text-slate-200 placeholder-slate-600 outline-none transition focus:border-blue-500"
            />
            <input
              type="password"
              placeholder="Password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              required
              minLength={6}
              className="w-full rounded-xl border border-slate-700/70 bg-slate-950/40 px-4 py-3 text-sm text-slate-200 placeholder-slate-600 outline-none transition focus:border-blue-500"
            />
            {error && (
              <div className="flex items-center gap-2 rounded-xl border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-300">
                <AlertCircle size={14} />
                {error}
              </div>
            )}
            <button
              type="submit"
              disabled={loading}
              className="w-full rounded-xl bg-gradient-to-r from-blue-600 to-cyan-500 px-4 py-3 text-sm font-semibold text-white shadow-[0_10px_25px_rgba(37,99,235,0.25)] transition hover:from-blue-500 hover:to-cyan-400 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {loading ? 'Please wait…' : isRegister ? 'Create account' : 'Sign in'}
            </button>
          </form>
          <div className="mt-5 text-center text-xs text-slate-500">
            {isRegister ? 'Already have an account?' : "Don't have an account?"}{' '}
            <button
              onClick={() => {
                setIsRegister(!isRegister)
                setError('')
              }}
              className="font-medium text-blue-300 transition hover:text-blue-200"
            >
              {isRegister ? 'Sign in' : 'Register'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [authToken, setAuthToken] = useState<AuthToken | null>(getStoredToken)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [topK, setTopK] = useState(5)
  const [sourceFilter, setSourceFilter] = useState('all')
  const bottomRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Conversation state
  const [sessionId, setSessionId] = useState<string>(uuidv4())
  const [conversations, setConversations] = useState<ConversationSummary[]>([])
  const [showHistory, setShowHistory] = useState(true)
  const [sidebarOpen, setSidebarOpen] = useState(false)

  useEffect(() => {
    const handler = () => setAuthToken(null)
    window.addEventListener('auth:logout', handler)
    return () => window.removeEventListener('auth:logout', handler)
  }, [])

  useEffect(() => {
    if (!authToken) return
    const poll = async () => {
      try {
        const h = await fetchHealth()
        setHealth(h)
      } catch {
        setHealth(null)
      }
    }
    poll()
    const id = setInterval(poll, 30_000)
    return () => clearInterval(id)
  }, [authToken])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = '0px'
    el.style.height = `${Math.min(el.scrollHeight, 140)}px`
  }, [input])

  // Load conversation list
  const refreshConversations = useCallback(async () => {
    if (!authToken) return
    try { setConversations((await listConversations()).conversations) } catch { /* not critical */ }
  }, [authToken])

  useEffect(() => { refreshConversations() }, [refreshConversations])

  const startNewConversation = useCallback(() => {
    setSessionId(uuidv4())
    setMessages([])
    setSidebarOpen(false)
  }, [])

  const switchConversation = useCallback(async (convSessionId: string) => {
    try {
      const { default: axios } = await import('axios')
      const token = getStoredToken()
      const { data } = await axios.get(`/api/v1/conversations/${convSessionId}`, {
        headers: token ? { Authorization: `Bearer ${token.access_token}` } : {},
      })
      setSessionId(convSessionId)
      setSidebarOpen(false)
      setMessages(
        data.messages.map((m: { id: string; role: string; content: string; sources?: SourceChunk[]; metadata?: Record<string, unknown>; created_at: string }) => ({
          id: m.id,
          role: m.role,
          content: m.content,
          sources: m.sources || undefined,
          meta: m.metadata && m.metadata.model_used ? {
            retrieval_time_ms: (m.metadata.retrieval_time_ms as number) || 0,
            language_detected: (m.metadata.language as string) || 'en',
            model_used: (m.metadata.model_used as string) || '',
            total_chunks_searched: 0,
          } : undefined,
          timestamp: new Date(m.created_at),
        }))
      )
    } catch {
      setSessionId(convSessionId)
      setMessages([])
    }
  }, [])

  const handleDeleteConversation = useCallback(async (convSessionId: string) => {
    try {
      await deleteConversation(convSessionId)
      setConversations(prev => prev.filter(c => c.session_id !== convSessionId))
      if (convSessionId === sessionId) startNewConversation()
    } catch { /* silent */ }
  }, [sessionId, startNewConversation])

  const sendMessage = useCallback(async (text: string) => {
    if (!text.trim() || loading) return
    setInput('')
    setLoading(true)

    const userMsg: Message = {
      id: uuidv4(),
      role: 'user',
      content: text.trim(),
      timestamp: new Date(),
    }
    setMessages(prev => [...prev, userMsg])

    try {
      const res = await queryKnowledgeBase(text.trim(), topK, sourceFilter, undefined, sessionId)
      if (res.session_id && res.session_id !== sessionId) setSessionId(res.session_id)
      const assistantMsg: Message = {
        id: uuidv4(),
        role: 'assistant',
        content: res.answer || 'No answer generated.',
        sources: res.sources,
        meta: {
          retrieval_time_ms: res.retrieval_time_ms,
          language_detected: res.language_detected,
          model_used: res.model_used,
          total_chunks_searched: res.total_chunks_searched,
        },
        chatId: res.chat_id,  // DB UUID for feedback
        timestamp: new Date(),
      }
      setMessages(prev => [...prev, assistantMsg])
      setTimeout(refreshConversations, 500)
    } catch (err: unknown) {
      const errorMsg: Message = {
        id: uuidv4(),
        role: 'assistant',
        content: `Error: ${err instanceof Error ? err.message : 'Request failed'}`,
        timestamp: new Date(),
      }
      setMessages(prev => [...prev, errorMsg])
    } finally {
      setLoading(false)
    }
  }, [loading, topK, sourceFilter, sessionId, refreshConversations])

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage(input)
    }
  }

  const suggestions = [
    'What is the glass transition temperature of borosilicate glass?',
    'What causes devitrification in glass manufacturing?',
    'Compare soda-lime and borosilicate glass for thermal shock resistance.',
    'دمای انتقال شیشه\u200cای بوروسیلیکات چقدر است؟',
    'علت ایجاد حباب در تولید شیشه چیست؟',
    'شبکه\u200cساز و اصلاح\u200cکننده در شیشه چه تفاوتی دارند؟',
  ]

  if (!authToken) {
    return <LoginScreen onAuth={(token) => setAuthToken(token)} />
  }

  return (
    <div className="relative flex h-screen overflow-hidden">
      <AmbientShell />

      {/* Mobile overlay (only on small screens where sidebar overlaps) */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-20 bg-black/50 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar — inline from md+ (768px), overlay only on small phones */}
      <aside className={`fixed inset-y-0 left-0 z-30 w-[310px] shrink-0 border-r border-slate-800/70 bg-slate-950/95 p-5 flex flex-col transition-transform duration-300 md:relative md:z-10 md:translate-x-0 md:bg-slate-950/35 ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}`}>
        <div className="mb-5">
          <div className="mb-3 flex items-center gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-500/20 to-cyan-400/20 ring-1 ring-blue-400/20">
              <Microscope size={22} className="text-blue-300" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-white">Glass Expert AI</h1>
              <p className="text-xs text-slate-500">Research-grade glass engineering copilot</p>
            </div>
          </div>
          <div className="grid gap-2">
            <StatPill icon={<Database size={14} />} label="Retriever" value="pgvector + BGE" />
            <StatPill icon={<Globe2 size={14} />} label="Languages" value="English + Persian" />
            <StatPill icon={<ShieldCheck size={14} />} label="Mode" value="Grounded answers only" />
          </div>
        </div>

        {/* New Chat button */}
        <button
          onClick={startNewConversation}
          className="mb-4 flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-blue-600 to-cyan-500 px-4 py-2.5 text-sm font-semibold text-white shadow-[0_10px_25px_rgba(37,99,235,0.20)] transition hover:from-blue-500 hover:to-cyan-400"
        >
          <Plus size={15} />
          New Chat
        </button>

        {/* Conversation History */}
        <div className={`${panelClass('p-4 mb-4')} flex-1 overflow-hidden flex flex-col`}>
          <button
            onClick={() => setShowHistory(!showHistory)}
            className="mb-3 flex w-full items-center justify-between text-left"
          >
            <div className="flex items-center gap-2">
              <MessageSquare size={14} className="text-blue-300" />
              <div>
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">History</div>
                <div className="text-sm text-slate-200">{conversations.length} conversation{conversations.length !== 1 ? 's' : ''}</div>
              </div>
            </div>
            <div className="text-slate-400">
              {showHistory ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            </div>
          </button>

          {showHistory && (
            <div className="flex-1 overflow-y-auto space-y-1">
              {conversations.map(conv => (
                <div
                  key={conv.session_id}
                  className={`group flex items-center gap-2 rounded-xl px-3 py-2 cursor-pointer text-xs transition ${conv.session_id === sessionId ? 'border border-blue-500/20 bg-blue-500/10 text-blue-300' : 'text-slate-400 hover:bg-slate-900/60 hover:text-slate-300'}`}
                  onClick={() => switchConversation(conv.session_id)}
                >
                  <MessageSquare size={12} className="flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="truncate font-medium">{conv.title}</p>
                    <p className="text-slate-600 text-[10px]">{timeAgo(conv.last_message_at)} · {conv.message_count} msgs</p>
                  </div>
                  <button
                    onClick={(e) => { e.stopPropagation(); handleDeleteConversation(conv.session_id) }}
                    className="opacity-0 group-hover:opacity-100 rounded-lg p-1 text-slate-600 hover:text-red-400 hover:bg-red-500/10 transition-all"
                    title="Delete"
                  >
                    <X size={12} />
                  </button>
                </div>
              ))}
              {conversations.length === 0 && (
                <p className="text-xs text-slate-600 text-center py-4">No conversations yet</p>
              )}
            </div>
          )}
        </div>

        {/* System Health */}
        <div className={`${panelClass('p-4 mb-4')}`}>
          <div className="mb-3 flex items-center justify-between">
            <div>
              <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">System health</div>
              <div className="mt-1 text-sm text-slate-200">Operational status</div>
            </div>
            <HealthBadge health={health} />
          </div>
          {health && (
            <div className="grid gap-2">
              <StatPill icon={<Gauge size={13} />} label="Chunks" value={health.total_chunks.toLocaleString()} />
              <StatPill icon={<BookOpen size={13} />} label="Documents" value={String(health.total_documents)} />
              <StatPill icon={<Settings2 size={13} />} label="Version" value={`v${health.version}`} />
            </div>
          )}
        </div>

        {/* Retrieval Controls */}
        <div className={`${panelClass('p-4 mb-4')}`}>
          <div className="mb-3 text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Retrieval controls</div>
          <label className="mb-2 block text-xs text-slate-400">
            Top-K sources: <span className="font-semibold text-slate-200">{topK}</span>
          </label>
          <input
            type="range"
            min={1}
            max={10}
            value={topK}
            onChange={e => setTopK(Number(e.target.value))}
            className="mb-4 w-full accent-blue-500"
          />
          <label className="mb-2 block text-xs text-slate-400">Source type</label>
          <select
            value={sourceFilter}
            onChange={e => setSourceFilter(e.target.value)}
            className="w-full rounded-xl border border-slate-700/70 bg-slate-950/50 px-3 py-2 text-sm text-slate-200 outline-none transition focus:border-blue-500"
          >
            <option value="all">All sources</option>
            <option value="textbook">Textbook</option>
            <option value="paper">Paper</option>
            <option value="sop">SOP</option>
            <option value="standard">Standard</option>
            <option value="manual">Manual</option>
          </select>
        </div>

        {/* Account */}
        <div className={`${panelClass('p-4')}`}>
          <div className="mb-2 text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Account</div>
          <div className="text-sm font-semibold text-white">{authToken.full_name || 'User'}</div>
          <div className="mt-1 truncate text-xs text-slate-500" title={authToken.email}>
            {authToken.email}
          </div>
          <div className="mt-4 flex items-center gap-2">
            <button
              onClick={() => setMessages([])}
              className="inline-flex items-center gap-2 rounded-xl border border-slate-700/70 bg-slate-950/45 px-3 py-2 text-xs text-slate-300 transition hover:border-red-400/40 hover:text-red-300"
            >
              <Trash2 size={13} />
              Clear chat
            </button>
            <button
              onClick={() => { logout(); setAuthToken(null) }}
              className="inline-flex items-center gap-2 rounded-xl border border-slate-700/70 bg-slate-950/45 px-3 py-2 text-xs text-slate-300 transition hover:border-red-400/40 hover:text-red-300"
            >
              <LogOut size={13} />
              Logout
            </button>
          </div>
        </div>
      </aside>

      {/* Main */}
      <main className="relative z-10 flex flex-1 flex-col overflow-hidden">
        {/* Header */}
        <header className="border-b border-slate-800/70 bg-slate-950/30 px-4 py-4 backdrop-blur md:px-6">
          <div className="mx-auto flex max-w-6xl items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="rounded-xl border border-slate-700/70 bg-slate-900/50 p-2 text-slate-400 transition hover:text-white md:hidden"
              >
                <Menu size={18} />
              </button>
              <div>
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
                  Glass research workspace
                </div>
                <h2 className="text-lg font-bold text-white">Ask, analyze, design, and troubleshoot</h2>
              </div>
            </div>
            <div className="hidden md:flex md:items-center md:gap-2">
              <span className="rounded-full border border-blue-500/20 bg-blue-500/10 px-3 py-1.5 text-xs text-blue-300">
                BAAI/bge-large-en-v1.5 + pgvector
              </span>
              <span className="rounded-full border border-slate-700/70 bg-slate-900/60 px-3 py-1.5 text-xs text-slate-400">
                {health ? `${health.total_documents} docs` : 'Loading…'}
              </span>
            </div>
          </div>
        </header>

        {/* Messages area */}
        <div className="flex-1 overflow-y-auto px-4 py-6 md:px-6">
          <div className="mx-auto max-w-5xl">
            {messages.length === 0 ? (
              <div className="flex min-h-[70vh] flex-col items-center justify-center">
                <div className="mb-6 flex h-20 w-20 items-center justify-center rounded-[28px] bg-gradient-to-br from-blue-500/20 to-cyan-400/20 ring-1 ring-blue-400/20">
                  <Microscope size={34} className="text-blue-300" />
                </div>
                <div className="mb-3 text-center">
                  <h3 className="mb-2 text-3xl font-extrabold text-white">Glass Expert AI</h3>
                  <p className="mx-auto max-w-2xl text-sm leading-7 text-slate-400">
                    A premium engineering copilot for glass science, process reasoning, composition analysis, and source-grounded answers.
                  </p>
                  <p dir="rtl" className="mx-auto mt-2 max-w-2xl text-sm leading-7 text-slate-500">
                    دستیار هوشمند تخصصی برای علم شیشه، تحلیل ترکیب، عیب‌یابی و پاسخ‌های مستند
                  </p>
                </div>
                <div className="grid w-full max-w-4xl gap-3 md:grid-cols-2 xl:grid-cols-3">
                  {suggestions.map((s, i) => {
                    const rtlSuggestion = isRtlText(s)
                    return (
                      <button
                        key={i}
                        onClick={() => sendMessage(s)}
                        dir={rtlSuggestion ? 'rtl' : undefined}
                        className={`${panelClass('p-4 text-left transition hover:border-blue-400/30 hover:bg-slate-900/65')} ${rtlSuggestion ? 'text-right' : 'text-left'}`}
                      >
                        <div className="mb-2 flex items-center gap-2 text-blue-300">
                          <Sparkles size={14} />
                          <span className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-500">
                            Suggested prompt
                          </span>
                        </div>
                        <div className="text-sm leading-7 text-slate-200">{s}</div>
                      </button>
                    )
                  })}
                </div>
              </div>
            ) : (
              <div className="space-y-1">
                {messages.map((msg, idx) => {
                  const prev = idx > 0 ? messages[idx - 1] : undefined
                  const prevQ = msg.role === 'assistant' && prev?.role === 'user' ? prev.content : undefined
                  return <MessageBubble key={msg.id} message={msg} prevQuestion={prevQ} />
                })}
              </div>
            )}
            {loading && (
              <div className="mt-3 flex justify-start">
                <div className={`${panelClass('max-w-sm rounded-3xl rounded-tl-md px-5 py-4')}`}>
                  <div className="mb-2 flex items-center gap-2">
                    <div className="rounded-xl bg-blue-500/10 p-2 text-blue-300 ring-1 ring-blue-500/20">
                      <Microscope size={14} className="animate-pulse" />
                    </div>
                    <span className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Reasoning</span>
                  </div>
                  <div className="flex items-center gap-2 text-sm text-slate-300">
                    <span>Searching the knowledge base</span>
                    <div className="flex gap-1">
                      <span className="h-2 w-2 rounded-full bg-blue-400 animate-bounce" style={{ animationDelay: '0ms' }} />
                      <span className="h-2 w-2 rounded-full bg-cyan-400 animate-bounce" style={{ animationDelay: '150ms' }} />
                      <span className="h-2 w-2 rounded-full bg-sky-300 animate-bounce" style={{ animationDelay: '300ms' }} />
                    </div>
                  </div>
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>
        </div>

        {/* Composer */}
        <div className="border-t border-slate-800/70 bg-slate-950/30 px-4 py-4 backdrop-blur md:px-6">
          <div className="mx-auto max-w-5xl">
            <div className={`${panelClass('rounded-[28px] p-3 shadow-[0_18px_40px_rgba(2,6,23,0.35)]')}`}>
              <div className="flex items-end gap-3">
                <textarea
                  ref={textareaRef}
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  dir={isRtlText(input) ? 'rtl' : 'ltr'}
                  placeholder="Ask a glass science question… / سوال خود را بپرسید…"
                  rows={1}
                  className="max-h-[140px] min-h-[52px] flex-1 resize-none bg-transparent px-3 py-3 text-sm leading-7 text-slate-100 placeholder-slate-600 outline-none"
                />
                <button
                  onClick={() => sendMessage(input)}
                  disabled={loading || !input.trim()}
                  className="rounded-2xl bg-gradient-to-r from-blue-600 to-cyan-500 p-3.5 text-white shadow-[0_12px_24px_rgba(37,99,235,0.22)] transition hover:from-blue-500 hover:to-cyan-400 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  <Send size={18} />
                </button>
              </div>
              <div className="mt-2 flex flex-wrap items-center justify-between gap-3 border-t border-slate-800/70 px-3 pt-3">
                <div className="flex flex-wrap gap-2 text-[11px] text-slate-500">
                  <span className="rounded-full bg-slate-900/70 px-2.5 py-1">Grounded responses</span>
                  <span className="rounded-full bg-slate-900/70 px-2.5 py-1">Bilingual flow</span>
                  <span className="rounded-full bg-slate-900/70 px-2.5 py-1">Source-aware answers</span>
                </div>
                <div className="text-[11px] text-slate-600">
                  Glass Expert AI · v3.0.0
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
