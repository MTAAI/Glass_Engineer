# Glass Expert AI — React Frontend

A modern React + TypeScript chat interface for the Glass Expert AI RAG system.

## Features

- **Chat interface** with streaming-style message display
- **Citations panel** — collapsible per-answer source list with similarity bars
- **Per-source feedback** — thumbs up/down on each retrieved chunk (writes to DB)
- **Answer feedback** — thumbs up/down on the overall answer (writes to DB)
- **System status sidebar** — live health check polling every 30s
- **Query settings** — adjustable top-k, source type filter
- **Dark theme** — matches the Streamlit UI design language

## Setup

```bash
cd frontend
npm install
npm run dev        # Development server on http://localhost:3000
npm run build      # Production build to dist/
```

## Architecture

The React frontend proxies all `/api/*` requests to the FastAPI backend on `localhost:8080` via Vite's dev server proxy. In production, configure Nginx to serve the `dist/` folder and proxy `/api` to the FastAPI server.

## Tech Stack

| Package | Purpose |
|---|---|
| React 18 + TypeScript | UI framework |
| Vite | Build tool |
| TailwindCSS | Styling |
| react-markdown | Render markdown answers |
| lucide-react | Icons |
| axios | HTTP client |
