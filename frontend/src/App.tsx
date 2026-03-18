import { useState, useEffect, useRef, useCallback, Component, type ReactNode, type ErrorInfo } from 'react'
import { v4 as uuidv4 } from 'uuid'
import ReactMarkdown from 'react-markdown'
import rehypeSanitize from 'rehype-sanitize'
import {
  Microscope, Send, Trash2, ChevronDown, ChevronUp,
  ThumbsUp, ThumbsDown, CheckCircle, AlertCircle,
  BookOpen, FileText, FlaskConical, Layers, Star,
  MessageSquarePlus, MessageSquare, Pencil, X, Check,
  RotateCcw, Cpu, Globe, LogOut,
} from 'lucide-react'
import {
  fetchHealth, queryKnowledgeBase, submitFeedback, submitSourceFeedback,
  createConversation, listConversations, getConversation, renameConversation, deleteConversation,
  login, register, logout, getStoredToken,
} from './api/client'
import type { Message, SourceChunk, HealthResponse, Conversation, AuthToken } from './types'

// ── Error Boundary ────────────────────────────────────────────────────────────
interface ErrorBoundaryProps { children: ReactNode }
interface ErrorBoundaryState { hasError: boolean; error: Error | null }

class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('Glass Expert AI crashed:', error, info.componentStack)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex items-center justify-center h-screen bg-[#0f1117] text-slate-200">
          <div className="text-center max-w-md p-6">
            <AlertCircle size={48} className="text-red-400 mx-auto mb-4" />
            <h2 className="text-xl font-bold mb-2">Something went wrong</h2>
            <p className="text-slate-400 text-sm mb-4">
              {this.state.error?.message || 'An unexpected error occurred.'}
            </p>
            <button
              onClick={() => { this.setState({ hasError: false, error: null }); window.location.reload() }}
              className="flex items-center gap-2 mx-auto bg-blue-600 hover:bg-blue-500 text-white rounded-lg px-4 py-2 text-sm transition-colors"
            >
              <RotateCcw size={14} /> Reload App
            </button>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}

// ── Utility ────────────────────────────────────────────────────────────────────
function formatMs(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

function langFlag(lang: string): string {
  return lang === 'fa' ? '🇮🇷' : '🇬🇧'
}

/** Check if text is predominantly RTL (Persian/Arabic). */
function isRtlText(text: string): boolean {
  const rtlChars = text.match(/[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]/g)
  return !!rtlChars && rtlChars.length > text.length * 0.3
}

function sourceIcon(type: string) {
  switch (type) {
    case 'textbook': return <BookOpen size={13} className="text-blue-400" />
    case 'paper': return <FileText size={13} className="text-purple-400" />
    case 'sop': return <Layers size={13} className="text-yellow-400" />
    case 'standard': return <Star size={13} className="text-orange-400" />
    default: return <FlaskConical size={13} className="text-cyan-400" />
  }
}

/** Format rerank score as a quality tier badge. */
function rerankBadge(score?: number) {
  if (score === undefined || score === null) return null
  const s = Number(score)
  if (s > 0.8) return <span className="text-[10px] bg-green-900/50 text-green-400 px-1.5 py-0.5 rounded font-medium">High</span>
  if (s > 0.4) return <span className="text-[10px] bg-yellow-900/50 text-yellow-400 px-1.5 py-0.5 rounded font-medium">Mid</span>
  return <span className="text-[10px] bg-slate-800 text-slate-500 px-1.5 py-0.5 rounded font-medium">Low</span>
}

/** Shorten model name for display. */
function modelLabel(model: string): string {
  if (model.includes('fallback')) return 'GPT-4o-mini'
  if (model.includes('llama') || model.includes('Llama')) return 'Llama 3.1 8B'
  if (model.includes('qwen') || model.includes('Qwen')) return 'Qwen 14B'
  if (model === 'retrieval-only') return 'Retrieval only'
  if (model === 'no-retrieval') return 'No results'
  return model.split('/').pop() || model
}

// ── Source Card ────────────────────────────────────────────────────────────────
interface SourceCardProps {
  source: SourceChunk
  index: number
  question: string
}

function SourceCard({ source, index, question }: SourceCardProps) {
  const [voted, setVoted] = useState<'up' | 'down' | null>(null)
  const pct = Math.round(source.similarity * 100)
  const rtl = isRtlText(source.content_preview || '')

  const handleVote = async (relevant: boolean) => {
    setVoted(relevant ? 'up' : 'down')
    // Note: per-source feedback requires feedback_id + document_id chain.
    // For now, just record the UI state. Full source feedback is
    // submitted after the main answer feedback creates a feedback_id.
  }

  return (
    <div className="bg-[#1a2332] border border-[#2d4a6e] rounded-lg p-3 mb-2 hover:border-[#3d5a7e] transition-colors">
      <div className="flex items-center gap-2 mb-1">
        {sourceIcon(source.source_type)}
        <span className="text-blue-400 font-medium text-sm truncate flex-1">[{index + 1}] {source.title}</span>
        <span className="text-[10px] bg-[#1e3a5f] text-slate-300 px-2 py-0.5 rounded">
          {source.source_type}
        </span>
        {rerankBadge(source.rerank_score)}
        <span className="text-[10px] text-slate-400">{langFlag(source.language)}</span>
        <span className="text-green-400 text-xs font-semibold">{pct}%</span>
      </div>

      {/* Similarity bar */}
      <div className="h-1 bg-[#0f1117] rounded-full mb-2">
        <div
          className="h-1 rounded-full bg-gradient-to-r from-blue-500 to-cyan-400 transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>

      {/* Content preview */}
      <p
        className="text-slate-500 text-xs italic leading-relaxed line-clamp-2"
        dir={rtl ? 'rtl' : 'ltr'}
      >
        {source.content_preview}
      </p>

      {/* Per-source feedback */}
      <div className="flex items-center gap-2 mt-2">
        <span className="text-slate-600 text-xs">Relevant?</span>
        {voted === null ? (
          <>
            <button
              onClick={() => handleVote(true)}
              className="text-slate-500 hover:text-green-400 transition-colors p-0.5"
              title="Yes, relevant"
            >
              <ThumbsUp size={13} />
            </button>
            <button
              onClick={() => handleVote(false)}
              className="text-slate-500 hover:text-red-400 transition-colors p-0.5"
              title="Not relevant"
            >
              <ThumbsDown size={13} />
            </button>
          </>
        ) : (
          <span className={`text-xs flex items-center gap-1 ${voted === 'up' ? 'text-green-400' : 'text-orange-400'}`}>
            <CheckCircle size={11} /> {voted === 'up' ? 'Noted' : 'Recorded'}
          </span>
        )}
      </div>
    </div>
  )
}

// ── Citations Panel ────────────────────────────────────────────────────────────
interface CitationsPanelProps {
  sources: SourceChunk[]
  question: string
}

function CitationsPanel({ sources, question }: CitationsPanelProps) {
  const [open, setOpen] = useState(false)
  if (!sources || sources.length === 0) return null

  return (
    <div className="mt-2">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 text-xs text-slate-400 hover:text-blue-400 transition-colors"
      >
        <BookOpen size={13} />
        <span>{sources.length} source{sources.length > 1 ? 's' : ''} cited</span>
        {open ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
      </button>

      {open && (
        <div className="mt-2 border-l-2 border-[#2d4a6e] pl-3 animate-in slide-in-from-top-2">
          {sources.map((src, i) => (
            <SourceCard key={i} source={src} index={i} question={question} />
          ))}
        </div>
      )}
    </div>
  )
}

// ── Answer Feedback Bar ────────────────────────────────────────────────────────
interface FeedbackBarProps {
  question: string
  answer: string
  messageId: string
  chatId?: string
}

function FeedbackBar({ question, answer, messageId, chatId }: FeedbackBarProps) {
  const [sent, setSent] = useState<'helpful' | 'not_helpful' | null>(null)
  const [showComment, setShowComment] = useState(false)
  const [comment, setComment] = useState('')

  const handleFeedback = async (helpful: boolean) => {
    setSent(helpful ? 'helpful' : 'not_helpful')
    // Only submit if we have a chat_id (requires auth + DB persistence)
    if (!chatId) return
    try {
      await submitFeedback({
        chat_id: chatId,
        rating: helpful ? 1 : -1,
        comment: comment || undefined,
      })
    } catch { /* silent */ }
  }

  if (sent !== null) {
    return (
      <div className="flex items-center gap-1 mt-2 text-xs text-green-400">
        <CheckCircle size={12} />
        <span>{sent === 'helpful' ? 'Glad it helped!' : 'Thanks — we\'ll improve.'}</span>
      </div>
    )
  }

  return (
    <div className="mt-3 pt-2 border-t border-[#2d3748]">
      <div className="flex items-center gap-3">
        <span className="text-xs text-slate-500">Was this helpful?</span>
        <button
          onClick={() => handleFeedback(true)}
          className="flex items-center gap-1 text-xs text-slate-400 hover:text-green-400 transition-colors"
        >
          <ThumbsUp size={13} /> Yes
        </button>
        <button
          onClick={() => {
            if (!showComment) {
              setShowComment(true)
            } else {
              handleFeedback(false)
            }
          }}
          className="flex items-center gap-1 text-xs text-slate-400 hover:text-red-400 transition-colors"
        >
          <ThumbsDown size={13} /> No
        </button>
      </div>
      {showComment && (
        <div className="flex items-center gap-2 mt-2">
          <input
            autoFocus
            value={comment}
            onChange={e => setComment(e.target.value)}
            placeholder="What could be better? (optional)"
            className="flex-1 text-xs bg-[#0f1117] border border-[#2d3748] rounded px-2 py-1 text-slate-300 placeholder-slate-600 outline-none focus:border-blue-500"
            onKeyDown={e => { if (e.key === 'Enter') handleFeedback(false) }}
          />
          <button
            onClick={() => handleFeedback(false)}
            className="text-xs text-blue-400 hover:text-blue-300"
          >
            Send
          </button>
        </div>
      )}
    </div>
  )
}

// ── Message Bubble ─────────────────────────────────────────────────────────────
interface MessageBubbleProps {
  message: Message
  prevQuestion?: string
}

function MessageBubble({ message, prevQuestion }: MessageBubbleProps) {
  const isUser = message.role === 'user'
  const rtl = isRtlText(message.content)
  const isResponseRtl = !isUser && message.meta?.language_detected === 'fa'
  const useRtl = rtl || isResponseRtl
  const dirProps = useRtl ? { dir: 'rtl' as const } : {}

  if (isUser) {
    return (
      <div className={`flex ${rtl ? 'justify-start' : 'justify-end'} mb-4`}>
        <div
          {...dirProps}
          className={`max-w-[80%] bg-[#1e3a5f] rounded-2xl ${rtl ? 'rounded-tl-sm' : 'rounded-tr-sm'} px-4 py-3 text-[#e8f4fd] text-sm shadow-lg shadow-blue-900/10`}
        >
          {message.content}
        </div>
      </div>
    )
  }

  const isError = message.content.startsWith('❌')

  return (
    <div className={`flex ${isResponseRtl ? 'justify-end' : 'justify-start'} mb-4`}>
      <div
        {...dirProps}
        className={`max-w-[85%] bg-[#1a1f2e] border ${isError ? 'border-red-900/50' : 'border-[#2d3748]'} rounded-2xl ${isResponseRtl ? 'rounded-tr-sm' : 'rounded-tl-sm'} px-4 py-3 shadow-lg shadow-black/20`}
      >
        {/* Header */}
        <div className={`flex items-center gap-2 mb-2 ${useRtl ? 'flex-row-reverse' : ''}`}>
          <Microscope size={14} className="text-blue-400" />
          <span className="text-xs text-slate-500">Glass Expert AI</span>
          {message.meta?.model_used && (
            <span className="text-[10px] bg-[#1e293b] text-slate-400 px-1.5 py-0.5 rounded flex items-center gap-1">
              <Cpu size={9} /> {modelLabel(message.meta.model_used)}
            </span>
          )}
          <span className="text-xs text-slate-600 ml-auto">
            {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </span>
        </div>

        {/* Answer content */}
        <div className={`text-slate-200 text-sm prose prose-invert prose-sm max-w-none ${useRtl ? 'text-right' : ''}`}>
          <ReactMarkdown rehypePlugins={[rehypeSanitize]}>{message.content}</ReactMarkdown>
        </div>

        {/* Metadata row */}
        {message.meta && !isError && (
          <div className={`flex flex-wrap gap-3 mt-2 text-xs text-slate-600 ${useRtl ? 'flex-row-reverse' : ''}`}>
            <span className="flex items-center gap-1">⏱ {formatMs(message.meta.retrieval_time_ms)}</span>
            <span className="flex items-center gap-1">
              <Globe size={10} /> {message.meta.language_detected?.toUpperCase()}
            </span>
            <span>📊 {message.meta.total_chunks_searched} sources</span>
          </div>
        )}

        {/* Citations panel */}
        {message.sources && message.sources.length > 0 && (
          <CitationsPanel sources={message.sources} question={prevQuestion || message.content} />
        )}

        {/* Answer feedback */}
        {!isError && message.content && message.content !== 'No answer generated.' && (
          <FeedbackBar
            question={prevQuestion || ''}
            answer={message.content}
            messageId={message.id}
            chatId={message.chatId}
          />
        )}
      </div>
    </div>
  )
}

// ── Health Badge ───────────────────────────────────────────────────────────────
function HealthBadge({ health }: { health: HealthResponse | null }) {
  if (!health) {
    return (
      <div className="flex items-center gap-1 text-xs text-red-400">
        <AlertCircle size={12} /> API offline
      </div>
    )
  }
  const ok = health.status === 'healthy'
  return (
    <div className={`flex items-center gap-1 text-xs ${ok ? 'text-green-400' : 'text-yellow-400'}`}>
      <CheckCircle size={12} />
      <span>{health.total_chunks.toLocaleString()} chunks</span>
    </div>
  )
}

// ── Loading Indicator ─────────────────────────────────────────────────────────
function LoadingBubble() {
  const [phase, setPhase] = useState(0)
  const phases = ['Searching knowledge base...', 'Ranking sources...', 'Generating answer...']

  useEffect(() => {
    const t1 = setTimeout(() => setPhase(1), 1200)
    const t2 = setTimeout(() => setPhase(2), 3000)
    return () => { clearTimeout(t1); clearTimeout(t2) }
  }, [])

  return (
    <div className="flex justify-start mb-4">
      <div className="bg-[#1a1f2e] border border-[#2d3748] rounded-2xl rounded-tl-sm px-4 py-3 shadow-lg shadow-black/20">
        <div className="flex items-center gap-2">
          <Microscope size={14} className="text-blue-400 animate-pulse" />
          <span className="text-xs text-slate-500">{phases[phase]}</span>
          <div className="flex gap-1">
            <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
            <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
            <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
          </div>
        </div>
        {/* Progress bar */}
        <div className="mt-2 h-0.5 bg-[#0f1117] rounded-full overflow-hidden w-48">
          <div
            className="h-full bg-gradient-to-r from-blue-500 to-cyan-400 rounded-full transition-all duration-1000 ease-out"
            style={{ width: `${(phase + 1) * 33}%` }}
          />
        </div>
      </div>
    </div>
  )
}

// ── Login / Register Screen ──────────────────────────────────────────────────
interface AuthScreenProps {
  onAuth: (token: AuthToken) => void
}

function AuthScreen({ onAuth }: AuthScreenProps) {
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [fullName, setFullName] = useState('')
  const [langPref, setLangPref] = useState('en')
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setSubmitting(true)
    try {
      let token: AuthToken
      if (mode === 'login') {
        token = await login(email, password)
      } else {
        token = await register(email, password, fullName || undefined, langPref)
      }
      onAuth(token)
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(msg || (mode === 'login' ? 'Invalid email or password' : 'Registration failed'))
    }
    setSubmitting(false)
  }

  return (
    <div className="flex items-center justify-center min-h-screen bg-[#0f1117]">
      <div className="w-full max-w-sm mx-4">
        {/* Logo */}
        <div className="text-center mb-8">
          <Microscope size={48} className="text-blue-400 mx-auto mb-3" />
          <h1 className="text-2xl font-bold text-white">Glass Expert AI</h1>
          <p className="text-slate-500 text-sm mt-1">Glass science knowledge assistant</p>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="bg-[#1a1f2e] border border-[#2d3748] rounded-xl p-6 space-y-4">
          <h2 className="text-lg font-semibold text-white text-center">
            {mode === 'login' ? 'Sign In' : 'Create Account'}
          </h2>

          {error && (
            <div className="flex items-center gap-2 bg-red-900/30 border border-red-800/50 rounded-lg px-3 py-2 text-xs text-red-400">
              <AlertCircle size={14} /> {error}
            </div>
          )}

          {mode === 'register' && (
            <div>
              <label className="text-xs text-slate-400 block mb-1">Full Name</label>
              <input
                type="text"
                value={fullName}
                onChange={e => setFullName(e.target.value)}
                placeholder="John Doe"
                className="w-full bg-[#0f1117] border border-[#2d3748] rounded-lg px-3 py-2 text-sm text-slate-200 placeholder-slate-600 outline-none focus:border-blue-500"
              />
            </div>
          )}

          <div>
            <label className="text-xs text-slate-400 block mb-1">Email</label>
            <input
              type="email"
              value={email}
              onChange={e => setEmail(e.target.value)}
              placeholder="you@company.com"
              required
              className="w-full bg-[#0f1117] border border-[#2d3748] rounded-lg px-3 py-2 text-sm text-slate-200 placeholder-slate-600 outline-none focus:border-blue-500"
            />
          </div>

          <div>
            <label className="text-xs text-slate-400 block mb-1">Password</label>
            <input
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              placeholder={mode === 'register' ? 'Min 6 characters' : '••••••••'}
              required
              minLength={mode === 'register' ? 6 : undefined}
              className="w-full bg-[#0f1117] border border-[#2d3748] rounded-lg px-3 py-2 text-sm text-slate-200 placeholder-slate-600 outline-none focus:border-blue-500"
            />
          </div>

          {mode === 'register' && (
            <div>
              <label className="text-xs text-slate-400 block mb-1">Language Preference</label>
              <select
                value={langPref}
                onChange={e => setLangPref(e.target.value)}
                className="w-full bg-[#0f1117] border border-[#2d3748] rounded-lg px-3 py-2 text-sm text-slate-200 outline-none focus:border-blue-500"
              >
                <option value="en">English</option>
                <option value="fa">فارسی (Persian)</option>
              </select>
            </div>
          )}

          <button
            type="submit"
            disabled={submitting}
            className="w-full bg-blue-600 hover:bg-blue-500 disabled:bg-blue-800 disabled:text-blue-400 text-white font-medium rounded-lg py-2.5 text-sm transition-colors"
          >
            {submitting ? 'Please wait...' : mode === 'login' ? 'Sign In' : 'Create Account'}
          </button>

          <p className="text-center text-xs text-slate-500">
            {mode === 'login' ? (
              <>Don't have an account?{' '}
                <button type="button" onClick={() => { setMode('register'); setError('') }} className="text-blue-400 hover:text-blue-300">
                  Sign up
                </button>
              </>
            ) : (
              <>Already have an account?{' '}
                <button type="button" onClick={() => { setMode('login'); setError('') }} className="text-blue-400 hover:text-blue-300">
                  Sign in
                </button>
              </>
            )}
          </p>
        </form>
      </div>
    </div>
  )
}

// ── Main App ───────────────────────────────────────────────────────────────────
function AppInner() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [topK, setTopK] = useState(5)
  const [sourceFilter, setSourceFilter] = useState('all')
  const [lastError, setLastError] = useState<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // ── Conversation state ──────────────────────────────────────────────────────
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [convLoading, setConvLoading] = useState(false)
  const [editingConvId, setEditingConvId] = useState<string | null>(null)
  const [editTitle, setEditTitle] = useState('')

  // Poll health every 30s
  useEffect(() => {
    const poll = async () => {
      try {
        const h = await fetchHealth()
        setHealth(h)
      } catch { setHealth(null) }
    }
    poll()
    const id = setInterval(poll, 30_000)
    return () => clearInterval(id)
  }, [])

  // Load conversation list on mount
  useEffect(() => {
    refreshConversations()
  }, [])

  const refreshConversations = async () => {
    try {
      const list = await listConversations()
      setConversations(list)
    } catch { /* silent */ }
  }

  // Scroll to bottom on new message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  // Auto-resize textarea
  const autoResize = useCallback(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = '44px'
    el.style.height = Math.min(el.scrollHeight, 120) + 'px'
  }, [])

  // ── Start new chat ────────────────────────────────────────────────────────
  const startNewChat = useCallback(() => {
    setCurrentSessionId(null)
    setMessages([])
    setLastError(null)
    textareaRef.current?.focus()
  }, [])

  // ── Load existing conversation ────────────────────────────────────────────
  const loadConversation = useCallback(async (sessionId: string) => {
    if (sessionId === currentSessionId) return
    setConvLoading(true)
    setLastError(null)
    try {
      const detail = await getConversation(sessionId)
      setCurrentSessionId(sessionId)
      setMessages(
        detail.messages.map(m => ({
          id: m.id,
          role: m.role as 'user' | 'assistant',
          content: m.content,
          sources: m.sources as unknown as SourceChunk[] | undefined,
          meta: m.metadata as Message['meta'],
          timestamp: new Date(m.created_at),
        }))
      )
    } catch (err) {
      setLastError('Failed to load conversation')
    }
    setConvLoading(false)
  }, [currentSessionId])

  // ── Delete conversation ───────────────────────────────────────────────────
  const handleDeleteConversation = useCallback(async (sessionId: string) => {
    try {
      await deleteConversation(sessionId)
      if (currentSessionId === sessionId) {
        setCurrentSessionId(null)
        setMessages([])
      }
      refreshConversations()
    } catch { /* silent */ }
  }, [currentSessionId])

  // ── Rename conversation ───────────────────────────────────────────────────
  const handleRenameSubmit = useCallback(async (sessionId: string) => {
    if (!editTitle.trim()) { setEditingConvId(null); return }
    try {
      await renameConversation(sessionId, editTitle.trim())
      refreshConversations()
    } catch { /* silent */ }
    setEditingConvId(null)
  }, [editTitle])

  // ── Send message (auto-creates session on first msg) ──────────────────────
  const sendMessage = useCallback(async (text: string) => {
    if (!text.trim() || loading) return
    setInput('')
    setLoading(true)
    setLastError(null)

    // Reset textarea height
    if (textareaRef.current) textareaRef.current.style.height = '44px'

    let sessionId = currentSessionId

    // Auto-create conversation on first message
    if (!sessionId) {
      try {
        const conv = await createConversation()
        sessionId = conv.session_id
        setCurrentSessionId(sessionId)
      } catch {
        // continue without session — still works, just no persistence
      }
    }

    const userMsg: Message = {
      id: uuidv4(),
      role: 'user',
      content: text.trim(),
      timestamp: new Date(),
    }
    setMessages(prev => [...prev, userMsg])

    try {
      const res = await queryKnowledgeBase(text.trim(), topK, sourceFilter, undefined, sessionId || undefined)
      const assistantMsg: Message = {
        id: uuidv4(),
        role: 'assistant',
        content: res.answer || 'No answer generated.',
        sources: res.sources,
        chatId: res.chat_id,
        meta: {
          retrieval_time_ms: res.retrieval_time_ms,
          language_detected: res.language_detected,
          model_used: res.model_used,
          total_chunks_searched: res.total_chunks_searched,
        },
        timestamp: new Date(),
      }
      setMessages(prev => [...prev, assistantMsg])

      // Refresh sidebar
      refreshConversations()
    } catch (err: unknown) {
      const errText = err instanceof Error ? err.message : 'Request failed'
      setLastError(errText)
      const errorMsg: Message = {
        id: uuidv4(),
        role: 'assistant',
        content: `❌ Error: ${errText}`,
        timestamp: new Date(),
      }
      setMessages(prev => [...prev, errorMsg])
    } finally {
      setLoading(false)
      textareaRef.current?.focus()
    }
  }, [loading, topK, sourceFilter, currentSessionId])

  // ── Retry last failed query ────────────────────────────────────────────────
  const retryLast = useCallback(() => {
    // Find last user message
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'user') {
        // Remove the error message
        setMessages(prev => prev.filter((_, idx) => idx < prev.length - 1))
        sendMessage(messages[i].content)
        break
      }
    }
  }, [messages, sendMessage])

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage(input)
    }
  }

  const suggestions = [
    { text: 'What is the glass transition temperature of borosilicate glass?', lang: 'en' },
    { text: 'What causes devitrification in glass manufacturing?', lang: 'en' },
    { text: 'دمای انتقال شیشه\u200cای بوروسیلیکات چقدر است؟', lang: 'fa' },
    { text: 'علل ایجاد حباب در تولید شیشه چیست؟', lang: 'fa' },
  ]

  return (
    <div className="flex h-screen bg-[#0f1117] text-slate-200 overflow-hidden">

      {/* ── Sidebar ─────────────────────────────────────────────────────────── */}
      <aside className="w-64 flex-shrink-0 bg-[#13161f] border-r border-[#2d3748] flex flex-col">
        {/* Header + New Chat */}
        <div className="p-4 pb-2">
          <div className="flex items-center gap-2 mb-1">
            <Microscope size={20} className="text-blue-400" />
            <h1 className="text-base font-bold text-white">Glass Expert AI</h1>
          </div>
          <button
            onClick={startNewChat}
            className="w-full mt-2 flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium rounded-lg px-3 py-2 transition-colors active:scale-[0.98]"
          >
            <MessageSquarePlus size={14} /> New Chat
          </button>
        </div>

        {/* Conversation List */}
        <div className="flex-1 overflow-y-auto px-2 py-1 scrollbar-thin scrollbar-thumb-[#2d3748]">
          {conversations.length === 0 ? (
            <p className="text-xs text-slate-600 text-center mt-4">No conversations yet</p>
          ) : (
            conversations.map(conv => {
              const isActive = conv.session_id === currentSessionId
              const convRtl = isRtlText(conv.title)
              return (
                <div
                  key={conv.session_id}
                  className={`group flex items-center gap-1 rounded-lg px-3 py-2 mb-0.5 cursor-pointer transition-colors text-xs ${
                    isActive
                      ? 'bg-[#1e3a5f] text-blue-300'
                      : 'text-slate-400 hover:bg-[#1a1f2e] hover:text-slate-200'
                  }`}
                >
                  {editingConvId === conv.session_id ? (
                    <div className="flex items-center gap-1 flex-1 min-w-0">
                      <input
                        autoFocus
                        value={editTitle}
                        onChange={e => setEditTitle(e.target.value)}
                        onKeyDown={e => {
                          if (e.key === 'Enter') handleRenameSubmit(conv.session_id)
                          if (e.key === 'Escape') setEditingConvId(null)
                        }}
                        className="flex-1 bg-[#0f1117] border border-[#2d4a6e] rounded px-1.5 py-0.5 text-xs text-slate-200 outline-none min-w-0"
                      />
                      <button onClick={() => handleRenameSubmit(conv.session_id)} className="text-green-400 hover:text-green-300">
                        <Check size={12} />
                      </button>
                      <button onClick={() => setEditingConvId(null)} className="text-slate-500 hover:text-slate-300">
                        <X size={12} />
                      </button>
                    </div>
                  ) : (
                    <>
                      <div
                        className="flex items-center gap-2 flex-1 min-w-0"
                        onClick={() => loadConversation(conv.session_id)}
                      >
                        <MessageSquare size={13} className="flex-shrink-0" />
                        <span className="truncate" dir={convRtl ? 'rtl' : undefined}>{conv.title}</span>
                      </div>
                      <div className="hidden group-hover:flex items-center gap-0.5 flex-shrink-0">
                        <button
                          onClick={e => { e.stopPropagation(); setEditingConvId(conv.session_id); setEditTitle(conv.title) }}
                          className="text-slate-500 hover:text-blue-400 p-0.5"
                          title="Rename"
                        >
                          <Pencil size={11} />
                        </button>
                        <button
                          onClick={e => { e.stopPropagation(); handleDeleteConversation(conv.session_id) }}
                          className="text-slate-500 hover:text-red-400 p-0.5"
                          title="Delete"
                        >
                          <Trash2 size={11} />
                        </button>
                      </div>
                    </>
                  )}
                </div>
              )
            })
          )}
        </div>

        {/* Bottom: Health + Settings */}
        <div className="border-t border-[#2d3748] p-4 space-y-3">
          <div>
            <HealthBadge health={health} />
            {health && (
              <div className="mt-1 space-y-0.5 text-xs text-slate-500">
                <div>DB: <span className={health.database === 'healthy' ? 'text-green-400' : 'text-red-400'}>{health.database}</span></div>
                <div>Cache: <span className={health.redis === 'healthy' ? 'text-green-400' : 'text-yellow-400'}>{health.redis}</span></div>
              </div>
            )}
          </div>

          <div>
            <label className="text-xs text-slate-500">Sources: {topK}</label>
            <input
              type="range" min={1} max={10} value={topK}
              onChange={e => setTopK(Number(e.target.value))}
              className="w-full accent-blue-500 mt-1"
            />
            <label className="text-xs text-slate-500 mt-1 block">Source type</label>
            <select
              value={sourceFilter}
              onChange={e => setSourceFilter(e.target.value)}
              className="w-full mt-1 bg-[#1a1f2e] border border-[#2d3748] rounded text-xs text-slate-300 px-2 py-1"
            >
              <option value="all">All</option>
              <option value="textbook">Textbook</option>
              <option value="paper">Paper</option>
              <option value="sop">SOP</option>
              <option value="standard">Standard</option>
              <option value="manual">Manual</option>
            </select>
          </div>

          {/* Logout */}
          <button
            onClick={() => logout()}
            className="w-full flex items-center gap-2 text-xs text-slate-500 hover:text-red-400 transition-colors px-1 py-1"
          >
            <LogOut size={13} /> Sign Out
          </button>
        </div>
      </aside>

      {/* ── Main Chat Area ───────────────────────────────────────────────────── */}
      <main className="flex-1 flex flex-col overflow-hidden">

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-4 scrollbar-thin scrollbar-thumb-[#2d3748]">
          {convLoading && (
            <div className="flex items-center justify-center h-full">
              <div className="flex items-center gap-2 text-slate-500 text-sm">
                <Microscope size={16} className="animate-pulse text-blue-400" />
                Loading conversation...
              </div>
            </div>
          )}
          {!convLoading && messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full gap-6">
              <div className="text-center">
                <Microscope size={48} className="text-blue-400 mx-auto mb-3" />
                <h2 className="text-xl font-bold text-white mb-1">Glass Expert AI</h2>
                <p className="text-slate-500 text-sm">
                  Ask any question about glass science, manufacturing, or properties.
                </p>
                <p dir="rtl" className="text-slate-600 text-xs mt-1">
                  هر سوالی در مورد علم شیشه، تولید یا خواص شیشه بپرسید
                </p>
              </div>
              <div className="grid grid-cols-2 gap-2 max-w-2xl w-full">
                {suggestions.map((s, i) => {
                  const isRtl = s.lang === 'fa'
                  return (
                    <button
                      key={i}
                      onClick={() => sendMessage(s.text)}
                      dir={isRtl ? 'rtl' : undefined}
                      className={`${isRtl ? 'text-right font-[Vazirmatn,system-ui,sans-serif]' : 'text-left'} text-xs bg-[#1a1f2e] border border-[#2d3748] rounded-lg p-3 text-slate-400 hover:border-blue-500 hover:text-blue-300 transition-all active:scale-[0.98]`}
                    >
                      {s.text}
                    </button>
                  )
                })}
              </div>
            </div>
          )}

          {messages.map((msg, idx) => {
            const prev = idx > 0 ? messages[idx - 1] : undefined
            const prevQ = msg.role === 'assistant' && prev?.role === 'user' ? prev.content : undefined
            return <MessageBubble key={msg.id} message={msg} prevQuestion={prevQ} />
          })}

          {loading && <LoadingBubble />}

          {/* Retry button for errors */}
          {lastError && !loading && (
            <div className="flex justify-center mb-4">
              <button
                onClick={retryLast}
                className="flex items-center gap-2 text-xs text-orange-400 hover:text-orange-300 bg-orange-900/20 border border-orange-800/30 rounded-lg px-3 py-1.5 transition-colors"
              >
                <RotateCcw size={12} /> Retry last question
              </button>
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="border-t border-[#2d3748] px-6 py-4 bg-[#13161f]">
          <div className="flex items-end gap-3 max-w-4xl mx-auto">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={e => { setInput(e.target.value); autoResize() }}
              onKeyDown={handleKeyDown}
              dir={isRtlText(input) ? 'rtl' : 'ltr'}
              placeholder="Ask a glass science question... / سوال خود را بپرسید..."
              rows={1}
              className="flex-1 bg-[#1a1f2e] border border-[#2d3748] rounded-xl px-4 py-3 text-sm text-slate-200 placeholder-slate-600 resize-none focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500/30 transition-all"
              style={{ minHeight: '44px', maxHeight: '120px' }}
            />
            <button
              onClick={() => sendMessage(input)}
              disabled={loading || !input.trim()}
              className="bg-blue-600 hover:bg-blue-500 disabled:bg-[#2d3748] disabled:text-slate-600 text-white rounded-xl p-3 transition-colors active:scale-95"
            >
              <Send size={16} />
            </button>
          </div>
          <p className="text-center text-xs text-slate-700 mt-2">
            Glass Expert AI · Hybrid RAG + Cross-encoder Reranking · v3.1.0
          </p>
        </div>
      </main>
    </div>
  )
}

// ── App with Auth + Error Boundary ───────────────────────────────────────────
export default function App() {
  const [authed, setAuthed] = useState<boolean>(() => !!getStoredToken())

  useEffect(() => {
    const handleLogout = () => setAuthed(false)
    window.addEventListener('auth:logout', handleLogout)
    return () => window.removeEventListener('auth:logout', handleLogout)
  }, [])

  return (
    <ErrorBoundary>
      {authed ? (
        <AppInner />
      ) : (
        <AuthScreen onAuth={() => setAuthed(true)} />
      )}
    </ErrorBoundary>
  )
}
