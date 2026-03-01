import { StrictMode, useState, useEffect } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App'
import AuthPage, { UserProfile } from './AuthPage'

const API_BASE = import.meta.env.VITE_API_BASE || ''

function Root() {
  const [token, setToken] = useState<string | null>(null)
  const [user, setUser] = useState<UserProfile | null>(null)
  const [checking, setChecking] = useState(true)

  useEffect(() => {
    const storedToken = localStorage.getItem('glass_ai_token')
    const storedUser = localStorage.getItem('glass_ai_user')

    if (!storedToken || !storedUser) {
      setChecking(false)
      return
    }

    let parsedUser: UserProfile

    try {
      parsedUser = JSON.parse(storedUser)
    } catch {
      localStorage.removeItem('glass_ai_token')
      localStorage.removeItem('glass_ai_user')
      setChecking(false)
      return
    }

    const verifyToken = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/v1/auth/me`, {
          headers: {
            Authorization: `Bearer ${storedToken}`,
          },
        })

        if (!res.ok) throw new Error('Token invalid')

        setToken(storedToken)
        setUser(parsedUser)
      } catch {
        // If API unreachable allow cached session
        setToken(storedToken)
        setUser(parsedUser)
      } finally {
        setChecking(false)
      }
    }

    verifyToken()
  }, [])

  const handleLogin = (newToken: string, newUser: UserProfile) => {
    localStorage.setItem('glass_ai_token', newToken)
    localStorage.setItem('glass_ai_user', JSON.stringify(newUser))

    setToken(newToken)
    setUser(newUser)
  }

  const handleLogout = () => {
    localStorage.removeItem('glass_ai_token')
    localStorage.removeItem('glass_ai_user')

    setToken(null)
    setUser(null)
  }

  if (checking) {
    return (
      <div className="min-h-screen bg-[#0f1117] flex items-center justify-center">
        <div className="flex items-center gap-3 text-slate-400">
          <span className="w-5 h-5 border-2 border-blue-400/30 border-t-blue-400 rounded-full animate-spin" />
          <span className="text-sm">Loading session...</span>
        </div>
      </div>
    )
  }

  if (!token || !user) {
    return <AuthPage onLogin={handleLogin} />
  }

  return <App token={token} user={user} onLogout={handleLogout} />
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Root />
  </StrictMode>
)