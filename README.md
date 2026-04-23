# AI-Agent-for-Saas# 🎬 AutoStream Conversational AI Agent

An agentic sales assistant for AutoStream built with LangGraph + Groq (Llama 3.3).

---

## 1. How to Run Locally

**Clone and setup:**
```bash
git clone https://github.com/your-username/autostream-agent.git
cd autostream-agent
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Add your API key in `agent/graph.py`:**
```python
api_key=SecretStr("gsk_your-groq-key-here")
```
Get a free key at 👉 https://console.groq.com

**Run:**
```bash
python main.py
```

**Test conversation:**


---

## 2. Architecture Explanation

**Why LangGraph?**
LangGraph was chosen over AutoGen because it gives explicit, code-defined control over state transitions. For a lead capture workflow, we need strict sequential steps — collect name, then email, then platform — and fire the tool only after all three are validated. LangGraph enforces this with a deterministic graph, whereas AutoGen's LLM-driven orchestration could skip steps unpredictably.

**How State is Managed:**
The agent uses a typed `AgentState` dictionary passed through every graph node. It stores the full message history (via LangGraph's `add_messages` reducer), the current `stage`, and three lead fields (`lead_name`, `lead_email`, `lead_platform`). State persists across all conversation turns in a session — no external database needed. The `mock_lead_capture()` tool is only triggered inside the `capture_lead` node, reached only after all fields are collected and validated.

**RAG Pipeline:**
The knowledge base (`autostream_kb.json`) is loaded at startup and injected into the system prompt as structured context, grounding all product/pricing answers without hallucination.

---

## 3. WhatsApp Deployment via Webhooks

**Architecture:**

**Key Steps:**
1. Register a webhook on Meta Developer Portal pointing to `https://yourdomain.com/webhook`
2. Verify ownership by echoing back `hub.challenge` on GET requests
3. On each POST, extract the phone number and message text from Meta's payload
4. Load the user's session state from Redis (keyed by phone number)
5. Run `graph.invoke(state)` and save the updated state back to Redis with a 1-hour TTL
6. Call the Meta Send Message API to deliver the agent's reply

**Why Redis?**
Each WhatsApp message is a separate HTTP request, so Redis stores `AgentState` between requests keyed by phone number, giving the agent memory across turns.

**Deployment:** Railway, Fly.io, or AWS Lambda + API Gateway.
