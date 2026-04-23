"""
agent/graph.py
Fixed version using Groq Llama 3.
"""
from __future__ import annotations
import re
from typing import Annotated, Literal, TypedDict
from pydantic import SecretStr
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from agent.rag import KB_CONTEXT
from tools.lead_capture import mock_lead_capture


# 1. State
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    lead_name: str | None
    lead_email: str | None
    lead_platform: str | None
    stage: Literal["chat", "collect_name", "collect_email", "collect_platform", "capture_lead"]
    lead_captured: bool


# 2. LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    api_key=SecretStr("api key here")
)


# 3. Prompts
CHAT_PROMPT = f"""You are a helpful sales assistant for AutoStream, an AI video editing SaaS.
Answer questions using ONLY this knowledge base:

{KB_CONTEXT}

IMPORTANT RULES:
- Give clear, helpful answers about pricing and features
- If user wants to sign up or buy, end your response with exactly: HIGH_INTENT
- Never say HIGH_INTENT unless user clearly wants to purchase/sign up
- Keep responses short and friendly
"""

COLLECT_NAME_PROMPT = "You are a sales assistant. The user wants to sign up for AutoStream. Ask for their full name only. Be friendly and brief."

COLLECT_EMAIL_PROMPT = "You are a sales assistant. You have the user's name. Now ask for their email address only. Be friendly and brief."

COLLECT_PLATFORM_PROMPT = "You are a sales assistant. Ask which creator platform they mainly use (YouTube, Instagram, TikTok, etc.). Be friendly and brief."

CONFIRM_PROMPT = "You are a sales assistant. Thank the user enthusiastically for signing up to AutoStream and tell them the team will reach out shortly."


# 4. Intent detection
HIGH_INTENT_KEYWORDS = [
    "sign up", "sign me up", "get started", "i want to try", "i want to subscribe",
    "ready to buy", "i'm in", "subscribe now", "how do i join", "where do i pay",
    "i want the pro plan", "i want the basic plan", "want to start", "ready to start",
    "i want to sign up", "let me sign up", "i would like to sign up", "how to sign up",
    "purchase", "buy now", "i'll take"
]

def _detect_high_intent(text: str) -> bool:
    return any(kw in text.lower() for kw in HIGH_INTENT_KEYWORDS)


# 5. Nodes
def chat_node(state: AgentState) -> AgentState:
    system = SystemMessage(content=CHAT_PROMPT)
    response = llm.invoke([system] + state["messages"])
    raw_text = str(response.content)

    high_intent = "HIGH_INTENT" in raw_text
    clean_text = raw_text.replace("HIGH_INTENT", "").strip()

    # Also check last user message
    last_user = next(
        (str(m.content) for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), ""
    )
    if _detect_high_intent(last_user):
        high_intent = True

    if not clean_text:
        clean_text = "I'd be happy to help! Could you tell me more about what you're looking for?"

    return {
        **state,
        "messages": [AIMessage(content=clean_text)],
        "stage": "collect_name" if high_intent else "chat",
    }


def collect_name_node(state: AgentState) -> AgentState:
    last_user = next(
        (str(m.content) for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
    )

    if state.get("lead_name") is None:
        response = llm.invoke([SystemMessage(content=COLLECT_NAME_PROMPT)] + state["messages"])
        return {
            **state,
            "messages": [AIMessage(content=str(response.content))],
            "stage": "collect_name",
        }

    name = last_user.strip() if last_user else "Unknown"
    response = llm.invoke([SystemMessage(content=COLLECT_EMAIL_PROMPT)] + state["messages"])
    return {
        **state,
        "lead_name": name,
        "messages": [AIMessage(content=str(response.content))],
        "stage": "collect_email",
    }


def collect_email_node(state: AgentState) -> AgentState:
    last_user = next(
        (str(m.content) for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), ""
    )

    email_match = re.search(r"[\w.\-+]+@[\w\-]+\.[a-zA-Z]{2,}", last_user)
    if not email_match:
        return {
            **state,
            "messages": [AIMessage(content="Hmm, that doesn't look like a valid email. Could you check and try again? 😊")],
            "stage": "collect_email",
        }

    response = llm.invoke([SystemMessage(content=COLLECT_PLATFORM_PROMPT)] + state["messages"])
    return {
        **state,
        "lead_email": email_match.group(0),
        "messages": [AIMessage(content=str(response.content))],
        "stage": "collect_platform",
    }


def collect_platform_node(state: AgentState) -> AgentState:
    last_user = next(
        (str(m.content) for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "Not specified"
    )
    response = llm.invoke([SystemMessage(content=CONFIRM_PROMPT)] + state["messages"])
    return {
        **state,
        "lead_platform": last_user.strip(),
        "messages": [AIMessage(content=str(response.content))],
        "stage": "capture_lead",
    }


def capture_lead_node(state: AgentState) -> AgentState:
    result = mock_lead_capture(
        name=state["lead_name"] or "Unknown",
        email=state["lead_email"] or "unknown@example.com",
        platform=state["lead_platform"] or "Unknown",
    )
    confirmation = (
        f"🎉 You're all set! Your lead ID is {result['lead_id']}. "
        "Our team will reach out shortly. Is there anything else I can help you with?"
    )
    return {
        **state,
        "messages": [AIMessage(content=confirmation)],
        "stage": "chat",
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "lead_captured": True,
    }


# 6. Router
def route(state: AgentState) -> str:
    return state["stage"]


# 7. Graph
def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("chat", chat_node)
    builder.add_node("collect_name", collect_name_node)
    builder.add_node("collect_email", collect_email_node)
    builder.add_node("collect_platform", collect_platform_node)
    builder.add_node("capture_lead", capture_lead_node)
    builder.set_entry_point("chat")

    builder.add_conditional_edges("chat", route, {
        "chat": END,
        "collect_name": "collect_name",
    })
    builder.add_conditional_edges("collect_name", route, {
        "collect_name": END,
        "collect_email": "collect_email",
    })
    builder.add_conditional_edges("collect_email", route, {
        "collect_email": END,
        "collect_platform": "collect_platform",
    })
    builder.add_edge("collect_platform", "capture_lead")
    builder.add_edge("capture_lead", END)

    return builder.compile()