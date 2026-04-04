# OpsForge 🔧
### Autonomous Operations Control Environment

An **OpenEnv-compliant** reinforcement-learning environment that simulates a real-world operations ticketing system. Tickets (bugs, complaints, fraud, requests) arrive dynamically; your AI agent must triage, prioritise, and resolve them under constrained resources and strict deadlines.

---

## Project Structure

```
opsforge/
├── opsforge/               # Core environment package
│   ├── __init__.py
│   ├── environment.py      # OpsForgeEnv (reset / step / state)
│   ├── models.py           # Pydantic models (Ticket, Action, Reward, …)
│   └── reward.py           # Reward function + resolution logic
├── tasks/                  # Evaluation tasks
│   ├── __init__.py
│   └── tasks.py            # EasyTask, MediumTask, HardTask
├── inference.py            # LLM agent runner
├── openenv.yaml            # OpenEnv specification
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run with heuristic agent (no API key needed)

```bash
python inference.py --task hard --no-llm
```

### 3. Run with LLM agent

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=sk-your-key-here

python inference.py --task hard
```

Available tasks: `easy` | `medium` | `hard`

### 4. Docker

```bash
# Build
docker build -t opsforge .

# Run (heuristic, no key needed)
docker run --rm opsforge python inference.py --task medium --no-llm

# Run with LLM
docker run --rm \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  -e HF_TOKEN=sk-... \
  opsforge
```

---

## Environment API

```python
from opsforge import OpsForgeEnv, Action

env = OpsForgeEnv()

obs = env.reset()           # → Observation
print(obs.tickets)          # List[Ticket]
print(obs.resources)        # Resources(engineers, budget, time_remaining)

action = Action(
    ticket_id=obs.tickets[0].id,
    assign_engineers=2,
    spend_budget=300.0,
    priority=3,
)

obs, reward, done, info = env.step(action)   # → (Observation, Reward, bool, dict)
state = env.state()                          # → State(step_count, done, …)
```

---

## Reward Function

| Signal | Formula |
|--------|---------|
| Resolve success | `severity × customer_value × 0.1 + deadline_bonus` |
| Resolve failure (20%) | `-(severity × 5)` |
| Resource waste | `-(over_eng × 2 + over_budget × 0.01)` |
| Missed deadline | `-(customer_value × 0.05 × severity)` |

---

## Tasks

| Task | Difficulty | Goal | Pass Condition |
|------|-----------|------|----------------|
| `EasyTask` | 🟢 Easy | Pick most urgent ticket | Priority accuracy ≥ 70 % |
| `MediumTask` | 🟡 Medium | Efficient resource allocation | Total reward > 0 |
| `HardTask` | 🔴 Hard | Full episode management | Composite score > 0 AND resolve rate ≥ 50 % |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible API base URL |
| `MODEL_NAME` | `gpt-4o-mini` | Model to use for inference |
| `HF_TOKEN` | _(empty)_ | API key / HuggingFace token |
