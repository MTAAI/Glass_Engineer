"""
Glass Expert AI — Load Test
============================
Simulates concurrent users hitting /query with glass science questions.

Usage:
    pip install locust

    # Headless (50 users, 10 spawn rate, 2 minutes):
    locust -f tests/load_test.py --headless -u 50 -r 10 -t 2m --host http://localhost:8080

    # With web UI (open http://localhost:8089):
    locust -f tests/load_test.py --host http://localhost:8080
"""
import os
import json
import random
from pathlib import Path
from locust import HttpUser, task, between, events

# ── Load test questions ────────────────────────────────────────────────────────
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


ENGLISH_QUESTIONS = _load_questions(GOLDEN_SET_PATH) or [
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

PERSIAN_QUESTIONS = _load_questions(PERSIAN_SET_PATH) or [
    "دمای انتقال شیشه چیست؟",
    "ویسکوزیته شیشه مذاب چگونه تغییر می‌کند؟",
    "عیوب اصلی شیشه کدامند؟",
]


class GlassExpertUser(HttpUser):
    """
    Simulates a real Glass Expert AI user.
    80% English queries, 20% Farsi queries.
    Wait 1-3s between queries (realistic think time).
    """
    wait_time = between(1, 3)
    token: str = None

    def on_start(self):
        """Login and get JWT token."""
        email = os.getenv("EVAL_EMAIL", "eval@glass-expert.ai")
        password = os.getenv("EVAL_PASSWORD", "eval123456")
        resp = self.client.post(
            "/api/v1/auth/login",
            data={"username": email, "password": password},
            name="POST /auth/login",
        )
        if resp.status_code == 200:
            self.token = resp.json().get("access_token")

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    @task(8)
    def query_english(self):
        """English query — 80% of traffic."""
        question = random.choice(ENGLISH_QUESTIONS)
        self.client.post(
            "/api/v1/query",
            json={"question": question, "top_k": 5},
            headers=self._headers(),
            name="POST /query [EN]",
        )

    @task(2)
    def query_farsi(self):
        """Farsi query — 20% of traffic."""
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
        self.client.get("/api/v1/health", name="GET /health")


@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    """Print final report."""
    stats = environment.runner.stats
    total = stats.total

    print(f"\n{'='*65}")
    print(f"  Glass Expert AI — Load Test Results")
    print(f"{'='*65}")
    print(f"  Total requests    : {total.num_requests}")
    print(f"  Failed requests   : {total.num_failures}")
    print(f"  Error rate        : {total.fail_ratio:.1%}")
    print(f"  Requests/sec      : {total.current_rps:.1f}")
    print(f"  Avg latency       : {total.avg_response_time:.0f}ms")
    print(f"  Median latency    : {total.median_response_time:.0f}ms")
    print(f"  p95 latency       : {total.get_response_time_percentile(0.95):.0f}ms")
    print(f"  p99 latency       : {total.get_response_time_percentile(0.99):.0f}ms")
    print(f"{'='*65}")
