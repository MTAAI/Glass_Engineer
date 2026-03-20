"""
Glass Expert AI — Load Test
============================
Simulates concurrent users hitting /query with questions from the golden set.

Usage:
    # Install locust first:
    pip install locust

    # Run headless (no UI) — 50 users, 10 spawn rate, 2 minutes:
    locust -f tests/load_test.py --headless -u 50 -r 10 -t 2m --host http://localhost:8080

    # Run with web UI (open http://localhost:8089 in browser):
    locust -f tests/load_test.py --host http://localhost:8080

Report metrics:
    - requests/sec
    - avg latency
    - p99 latency
    - error rate
    - at what concurrency the system breaks
"""
import json
import random
from pathlib import Path
from locust import HttpUser, task, between, events
from locust.runners import MasterRunner

# ── Golden set questions ───────────────────────────────────────────────────────
GOLDEN_SET_PATH = Path(__file__).parent.parent / "data" / "evaluation" / "golden_eval_set.jsonl"
PERSIAN_SET_PATH = Path(__file__).parent.parent / "data" / "evaluation" / "persian_eval_set.jsonl"

def _load_questions(path: Path) -> list[str]:
    questions = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        q = json.loads(line)
                        questions.append(q.get("question", ""))
                    except Exception:
                        pass
    return [q for q in questions if q]

ENGLISH_QUESTIONS = _load_questions(GOLDEN_SET_PATH)
PERSIAN_QUESTIONS = _load_questions(PERSIAN_SET_PATH)

# Fallback questions if files not found
if not ENGLISH_QUESTIONS:
    ENGLISH_QUESTIONS = [
        "What is the glass transition temperature of borosilicate glass?",
        "How does viscosity change with temperature in glass melts?",
        "What causes devitrification in glass?",
        "What is the coefficient of thermal expansion of soda-lime glass?",
        "How is float glass manufactured?",
        "What are the main types of glass defects?",
        "Explain the Vogel-Fulcher-Tammann equation.",
        "What is the annealing point of glass?",
        "How does chemical strengthening work?",
        "What is the refractive index of optical glass?",
    ]

if not PERSIAN_QUESTIONS:
    PERSIAN_QUESTIONS = [
        "دمای انتقال شیشه چیست؟",
        "ویسکوزیته شیشه مذاب چگونه تغییر می‌کند؟",
        "عیوب اصلی شیشه کدامند؟",
    ]

ALL_QUESTIONS = ENGLISH_QUESTIONS + PERSIAN_QUESTIONS


# ── Auth helper ────────────────────────────────────────────────────────────────
class GlassExpertUser(HttpUser):
    """
    Simulates a real Glass Expert AI user.
    - Logs in on start
    - Sends queries with random questions from the golden set
    - 80% English, 20% Farsi
    - Wait 1-3 seconds between queries (realistic think time)
    """
    wait_time = between(1, 3)
    token: str = None

    def on_start(self):
        """Login and get JWT token."""
        resp = self.client.post(
            "/api/v1/auth/login",
            data={
                "username": "arjun@glassai.com",
                "password": "glass2024",
            },
            name="POST /auth/login",
        )
        if resp.status_code == 200:
            self.token = resp.json().get("access_token")
        else:
            self.token = None

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    @task(8)
    def query_english(self):
        """Send an English glass science question — 80% of traffic."""
        question = random.choice(ENGLISH_QUESTIONS)
        self.client.post(
            "/api/v1/query",
            json={"question": question, "top_k": 5},
            headers=self._headers(),
            name="POST /query [EN]",
        )

    @task(2)
    def query_farsi(self):
        """Send a Farsi glass science question — 20% of traffic."""
        question = random.choice(PERSIAN_QUESTIONS)
        self.client.post(
            "/api/v1/query",
            json={"question": question, "top_k": 5},
            headers=self._headers(),
            name="POST /query [FA]",
        )

    @task(1)
    def health_check(self):
        """Health check — small % of traffic."""
        self.client.get(
            "/api/v1/health",
            name="GET /health",
        )


# ── Event hooks for reporting ──────────────────────────────────────────────────
@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    """Print final report when test ends."""
    stats = environment.runner.stats
    total = stats.total

    print("\n" + "=" * 65)
    print("  Glass Expert AI — Load Test Results")
    print("=" * 65)
    print(f"  Total requests    : {total.num_requests}")
    print(f"  Failed requests   : {total.num_failures}")
    print(f"  Error rate        : {total.fail_ratio:.1%}")
    print(f"  Requests/sec      : {total.current_rps:.1f}")
    print(f"  Avg latency       : {total.avg_response_time:.0f}ms")
    print(f"  Median latency    : {total.median_response_time:.0f}ms")
    print(f"  p95 latency       : {total.get_response_time_percentile(0.95):.0f}ms")
    print(f"  p99 latency       : {total.get_response_time_percentile(0.99):.0f}ms")
    print(f"  Min latency       : {total.min_response_time:.0f}ms")
    print(f"  Max latency       : {total.max_response_time:.0f}ms")
    print("=" * 65)

    # Per-endpoint breakdown
    print("\n  Per-endpoint breakdown:")
    for name, entry in stats.entries.items():
        if entry.num_requests > 0:
            print(
                f"  {name[1]:40s} | "
                f"reqs={entry.num_requests:4d} | "
                f"avg={entry.avg_response_time:6.0f}ms | "
                f"p95={entry.get_response_time_percentile(0.95):6.0f}ms | "
                f"err={entry.fail_ratio:.0%}"
            )
    print("=" * 65)