"""
main.py
Entry point for the AutoStream Conversational AI Agent.
Run with:  python main.py
"""

import os
import sys
from typing import Any, cast

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage
from agent.graph import build_graph, AgentState


def main():
    print("\n" + "=" * 60)
    print("  🎬  AutoStream AI Sales Assistant")
    print("  Powered by LangGraph + Gemini 1.5 Flash")
    print("=" * 60)
    print("  Type 'quit' or 'exit' to end the session.\n")

    graph = build_graph()

    state: AgentState = AgentState(
        messages=[],
        lead_name=None,
        lead_email=None,
        lead_platform=None,
        stage="chat",
        lead_captured=False,
    )

    opening = (
        "Hi there! 👋 Welcome to AutoStream – your AI-powered video editing assistant.\n"
        "I can answer questions about our plans, features, and pricing.\n"
        "How can I help you today?"
    )
    print(f"🤖 AutoStream: {opening}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Thanks for chatting with AutoStream. Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("👋 Thanks for chatting with AutoStream. Goodbye!")
            break

        # Append new human message
        state["messages"] = list(state["messages"]) + [HumanMessage(content=user_input)]

        # Invoke the graph
        result: Any = graph.invoke(cast(AgentState, state))
        state = cast(AgentState, result)

        # Print last AI message
        last_ai = next(
            (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
            None,
        )
        if last_ai:
            print(f"\n🤖 AutoStream: {last_ai.content}\n")


if __name__ == "__main__":
    main()