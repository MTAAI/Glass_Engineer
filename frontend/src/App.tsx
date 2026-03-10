import { useState, useEffect, useRef, useCallback } from 'react'
import { v4 as uuidv4 } from 'uuid'
import ReactMarkdown from 'react-markdown'
import {
  Microscope, Send, Trash2, ChevronDown, ChevronUp,
  ThumbsUp, ThumbsDown, CheckCircle, AlertCircle,
  BookOpen, FileText, FlaskConical, Layers, Star,
  Plus, MessageSquare, X
} from 'lucide-react'
import {
  fetchHealth, queryKnowledgeBase, submitFeedback, submitSourceFeedback,
  createConversation, listConversations, getConversation, deleteConversation
} from './api/client'
import type { Message, SourceChunk, HealthResponse, Conversation } from './types'

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

function timeAgo(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  return `${days}d ago`
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
    <div className="bg-[#1a2332] border border-[#2d4a6e] rounded-lg p-3 mb-2">
      <div className="flex items-center gap-2 mb-1">
        {sourceIcon(source.source_type)}
        <span className="text-blue-400 font-medium text-sm truncate flex-1">{source.title}</span>
        <span className="text-[10px] bg-[#1e3a5f] text-slate-300 px-2 py-0.5 rounded">
          {source.source_type}
        </span>
        <span className="text-[10px] text-slate-400">{langFlag(source.language)}</span>
        <span className="text-green-400 text-xs font-semibold">{pct}%</span>
      </div>

      {/* Similarity bar */}
      <div className="h-1 bg-[#0f1117] rounded-full mb-2">
        <div
          className="h-1 rounded-full bg-gradient-to-r from-blue-500 to-cyan-400"
          style={{ width: `${pct}%` }}
        />
      </div>

      {/* Content preview */}
      <p className="text-slate-500 text-xs italic leading-relaxed line-clamp-2">
        {source.content_preview}
      </p>

      {/* Per-source feedback */}
      <div className="flex items-center gap-2 mt-2">
        <span className="text-slate-600 text-xs">Relevant?</span>
        {voted === null ? (
          <>
            <button
              onClick={() => handleVote(true)}
              className="text-slate-500 hover:text-green-400 transition-colors"
              title="Yes, relevant"
            >
              <ThumbsUp size={13} />
            </button>
            <button
              onClick={() => handleVote(false)}
              className="text-slate-500 hover:text-red-400 transition-colors"
              title="Not relevant"
            >
              <ThumbsDown size={13} />
            </button>
          </>
        ) : (
          <span className="text-xs text-green-400 flex items-center gap-1">
            <CheckCircle size={11} /> Noted
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
        <span>{sources.length} source{sources.length > 1 ? 's' : ''} used</span>
        {open ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
      </button>

      {open && (
        <div className="mt-2 border-l-2 border-[#2d4a6e] pl-3">
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
}

function FeedbackBar({ question, answer }: FeedbackBarProps) {
  const [sent, setSent] = useState<'helpful' | 'not_helpful' | null>(null)

  const handleFeedback = async (helpful: boolean) => {
    setSent(helpful ? 'helpful' : 'not_helpful')
    try {
      await submitFeedback({
        question,
        answer,
        helpful,
        rating: helpful ? 4 : 2,
      })
    } catch { /* silent */ }
  }

  if (sent !== null) {
    return (
      <div className="flex items-center gap-1 mt-2 text-xs text-green-400">
        <CheckCircle size={12} />
        <span>{sent === 'helpful' ? 'Glad it helped!' : 'Feedback recorded — we\'ll improve.'}</span>
      </div>
    )
  }

  return (
    <div className="flex items-center gap-3 mt-3 pt-2 border-t border-[#2d3748]">
      <span className="text-xs text-slate-500">Was this helpful?</span>
      <button
        onClick={() => handleFeedback(true)}
        className="flex items-center gap-1 text-xs text-slate-400 hover:text-green-400 transition-colors"
      >
        <ThumbsUp size={13} /> Yes
      </button>
      <button
        onClick={() => handleFeedback(false)}
        className="flex items-center gap-1 text-xs text-slate-400 hover:text-red-400 transition-colors"
      >
        <ThumbsDown size={13} /> No
      </button>
    </div>
  )
}

// ── Message Bubble ─────────────────────────────────────────────────────────────
interface MessageBubbleProps {
  message: Message
  userQuestion?: string
}

function MessageBubble({ message, userQuestion }: MessageBubbleProps) {
  const isUser = message.role === 'user'
  const rtl = isRtlText(message.content)
  const isResponseRtl = !isUser && message.meta?.language_detected === 'fa'
  const dirProps = (rtl || isResponseRtl) ? { dir: 'rtl' as const } : {}

  if (isUser) {
    return (
      <div className={`flex ${rtl ? 'justify-start' : 'justify-end'} mb-4`}>
        <div
          {...dirProps}
          className={`max-w-[80%] bg-[#1e3a5f] rounded-2xl ${rtl ? 'rounded-tl-sm' : 'rounded-tr-sm'} px-4 py-3 text-[#e8f4fd] text-sm`}
        >
          {message.content}
        </div>
      </div>
    )
  }

  return (
    <div className={`flex ${isResponseRtl ? 'justify-end' : 'justify-start'} mb-4`}>
      <div
        {...dirProps}
        className={`max-w-[85%] bg-[#1a1f2e] border border-[#2d3748] rounded-2xl ${isResponseRtl ? 'rounded-tr-sm' : 'rounded-tl-sm'} px-4 py-3`}
      >
        <div className="flex items-center gap-2 mb-2">
          <Microscope size={14} className="text-blue-400" />
          <span className="text-xs text-slate-500">Glass Expert AI</span>
          <span className="text-xs text-slate-600">
            {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </span>
        </div>

        <div className="text-slate-200 text-sm prose prose-invert prose-sm max-w-none">
          <ReactMarkdown>{message.content}</ReactMarkdown>
        </div>

        {/* Metadata row */}
        {message.meta && (
          <div className="flex flex-wrap gap-3 mt-2 text-xs text-slate-600">
            <span>⏱ {formatMs(message.meta.retrieval_time_ms)}</span>
            <span>🌐 {message.meta.language_detected?.toUpperCase()}</span>
            <span>🤖 {message.meta.model_used}</span>
            <span>📊 {message.meta.total_chunks_searched} chunks</span>
          </div>
        )}

        {/* Citations panel */}
        {message.sources && message.sources.length > 0 && (
          <CitationsPanel sources={message.sources} question={userQuestion || message.content} />
        )}

        {/* Answer feedback */}
        {message.content && message.content !== 'No answer generated.' && (
          <FeedbackBar question={userQuestion || message.content} answer={message.content} />
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
      <span>{health.total_chunks.toLocaleString()} chunks · {health.total_documents} docs</span>
    </div>
  )
}

// ── Main App ───────────────────────────────────────────────────────────────────
export default function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [topK, setTopK] = useState(5)
  const [sourceFilter, setSourceFilter] = useState('all')
  const bottomRef = useRef<HTMLDivElement>(null)

  // Conversation state
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)

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

  // Load conversations on mount
  useEffect(() => {
    loadConversations()
  }, [])

  // Scroll to bottom on new message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const loadConversations = async () => {
    try {
      const convs = await listConversations()
      setConversations(convs)
    } catch { /* API may not be ready */ }
  }

  const startNewChat = () => {
    setMessages([])
    setActiveSessionId(null)
  }

  const loadConversation = async (sessionId: string) => {
    try {
      const detail = await getConversation(sessionId)
      setActiveSessionId(sessionId)
      const msgs: Message[] = detail.messages.map(m => ({
        id: m.id,
        role: m.role as 'user' | 'assistant',
        content: m.content,
        sources: m.sources as SourceChunk[] | undefined,
        meta: m.metadata as Message['meta'] | undefined,
        timestamp: new Date(m.created_at),
      }))
      setMessages(msgs)
    } catch {
      // If conversation can't be loaded, start fresh
      startNewChat()
    }
  }

  const handleDeleteConversation = async (e: React.MouseEvent, sessionId: string) => {
    e.stopPropagation()
    try {
      await deleteConversation(sessionId)
      setConversations(prev => prev.filter(c => c.session_id !== sessionId))
      if (activeSessionId === sessionId) {
        startNewChat()
      }
    } catch { /* silent */ }
  }

  const sendMessage = useCallback(async (text: string) => {
    if (!text.trim() || loading) return
    setInput('')
    setLoading(true)

    // Create session on first message if none exists
    let sessionId = activeSessionId
    if (!sessionId) {
      try {
        const conv = await createConversation()
        sessionId = conv.session_id
        setActiveSessionId(sessionId)
      } catch {
        // Continue without session persistence
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
        meta: {
          retrieval_time_ms: res.retrieval_time_ms,
          language_detected: res.language_detected,
          model_used: res.model_used,
          total_chunks_searched: res.total_chunks_searched,
        },
        timestamp: new Date(),
      }
      setMessages(prev => [...prev, assistantMsg])

      // Refresh conversation list (title may have been updated)
      loadConversations()
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
  }, [loading, topK, sourceFilter, activeSessionId])

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage(input)
    }
  }

  const suggestions = [
    'What is the glass transition temperature of borosilicate glass?',
    'What causes devitrification in glass manufacturing?',
    'دمای انتقال شیشه‌ای بوروسیلیکات چقدر است؟',
    'علل ایجاد حباب در تولید شیشه چیست؟',
  ]

  // Find the user question preceding each assistant message
  const getUserQuestion = (index: number): string => {
    for (let i = index - 1; i >= 0; i--) {
      if (messages[i].role === 'user') return messages[i].content
    }
    return ''
  }

  return (
    <div className="flex h-screen bg-[#0f1117] text-slate-200 overflow-hidden">

      {/* ── Sidebar ─────────────────────────────────────────────────────────── */}
      <aside className="w-64 flex-shrink-0 bg-[#13161f] border-r border-[#2d3748] flex flex-col p-4 gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Microscope size={20} className="text-blue-400" />
            <h1 className="text-base font-bold text-white">Glass Expert AI</h1>
          </div>
          <p className="text-xs text-slate-500">BAAI/bge-large-en-v1.5 + pgvector</p>
        </div>

        {/* New Chat button */}
        <button
          onClick={startNewChat}
          className="flex items-center gap-2 w-full bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium rounded-lg px-3 py-2 transition-colors"
        >
          <Plus size={14} /> New Chat
        </button>

        {/* Conversation list */}
        <div className="border-t border-[#2d3748] pt-3 flex-1 overflow-y-auto min-h-0">
          <p className="text-xs font-semibold text-slate-400 mb-2">Conversations</p>
          {conversations.length === 0 ? (
            <p className="text-xs text-slate-600 italic">No conversations yet</p>
          ) : (
            <div className="space-y-1">
              {conversations.map(conv => (
                <div
                  key={conv.session_id}
                  onClick={() => loadConversation(conv.session_id)}
                  className={`group flex items-center gap-2 rounded-lg px-2 py-2 cursor-pointer transition-colors ${
                    activeSessionId === conv.session_id
                      ? 'bg-[#1e3a5f] text-blue-300'
                      : 'hover:bg-[#1a1f2e] text-slate-400'
                  }`}
                >
                  <MessageSquare size={13} className="flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="text-xs truncate">{conv.title}</p>
                    <p className="text-[10px] text-slate-600">{timeAgo(conv.updated_at)}</p>
                  </div>
                  <button
                    onClick={(e) => handleDeleteConversation(e, conv.session_id)}
                    className="opacity-0 group-hover:opacity-100 text-slate-600 hover:text-red-400 transition-all"
                    title="Delete conversation"
                  >
                    <X size={12} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="border-t border-[#2d3748] pt-3">
          <p className="text-xs font-semibold text-slate-400 mb-2">System Status</p>
          <HealthBadge health={health} />
          {health && (
            <div className="mt-1 space-y-0.5 text-xs text-slate-500">
              <div>DB: <span className={health.database === 'healthy' ? 'text-green-400' : 'text-red-400'}>{health.database}</span></div>
              <div>Cache: <span className={health.redis === 'healthy' ? 'text-green-400' : 'text-yellow-400'}>{health.redis}</span></div>
              <div>v{health.version}</div>
            </div>
          )}
        </div>

        <div className="border-t border-[#2d3748] pt-3">
          <p className="text-xs font-semibold text-slate-400 mb-2">Query Settings</p>
          <label className="text-xs text-slate-500">Sources: {topK}</label>
          <input
            type="range" min={1} max={10} value={topK}
            onChange={e => setTopK(Number(e.target.value))}
            className="w-full accent-blue-500 mt-1"
          />
          <label className="text-xs text-slate-500 mt-2 block">Source type</label>
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
      </aside>

      {/* ── Main Chat Area ───────────────────────────────────────────────────── */}
      <main className="flex-1 flex flex-col overflow-hidden">

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full gap-6">
              <div className="text-center">
                <Microscope size={48} className="text-blue-400 mx-auto mb-3" />
                <h2 className="text-xl font-bold text-white mb-1">Glass Expert AI</h2>
                <p className="text-slate-500 text-sm">Ask any question about glass science, manufacturing, or properties.<br /><span dir="rtl" className="text-slate-600 text-xs">هر سوالی در مورد علم شیشه، تولید یا خواص شیشه بپرسید</span></p>
              </div>
              <div className="grid grid-cols-2 gap-2 max-w-2xl w-full">
                {suggestions.map((s, i) => {
                  const rtlSuggestion = isRtlText(s)
                  return (
                    <button
                      key={i}
                      onClick={() => sendMessage(s)}
                      dir={rtlSuggestion ? 'rtl' : undefined}
                      className={`${rtlSuggestion ? 'text-right' : 'text-left'} text-xs bg-[#1a1f2e] border border-[#2d3748] rounded-lg p-3 text-slate-400 hover:border-blue-500 hover:text-blue-300 transition-all`}
                    >
                      {s}
                    </button>
                  )
                })}
              </div>
            </div>
          )}

          {messages.map((msg, idx) => (
            <MessageBubble
              key={msg.id}
              message={msg}
              userQuestion={msg.role === 'assistant' ? getUserQuestion(idx) : undefined}
            />
          ))}

          {loading && (
            <div className="flex justify-start mb-4">
              <div className="bg-[#1a1f2e] border border-[#2d3748] rounded-2xl rounded-tl-sm px-4 py-3">
                <div className="flex items-center gap-2">
                  <Microscope size={14} className="text-blue-400 animate-pulse" />
                  <span className="text-xs text-slate-500">Searching knowledge base...</span>
                  <div className="flex gap-1">
                    <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                    <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                    <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                </div>
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="border-t border-[#2d3748] px-6 py-4 bg-[#13161f]">
          <div className="flex items-end gap-3 max-w-4xl mx-auto">
            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              dir={isRtlText(input) ? 'rtl' : 'ltr'}
              placeholder="Ask a glass science question... / سوال خود را بپرسید..."
              rows={1}
              className="flex-1 bg-[#1a1f2e] border border-[#2d3748] rounded-xl px-4 py-3 text-sm text-slate-200 placeholder-slate-600 resize-none focus:outline-none focus:border-blue-500 transition-colors"
              style={{ minHeight: '44px', maxHeight: '120px' }}
            />
            <button
              onClick={() => sendMessage(input)}
              disabled={loading || !input.trim()}
              className="bg-blue-600 hover:bg-blue-500 disabled:bg-[#2d3748] disabled:text-slate-600 text-white rounded-xl p-3 transition-colors"
            >
              <Send size={16} />
            </button>
          </div>
          <p className="text-center text-xs text-slate-700 mt-2">
            Glass Expert AI · BAAI/bge-large-en-v1.5 + pgvector · v3.0.0
          </p>
        </div>
      </main>
    </div>
  )
}
