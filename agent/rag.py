"""
agent/rag.py
RAG pipeline that loads the local JSON knowledge base
and provides context string used by the agent.
"""

import json
from pathlib import Path


KB_PATH = Path(__file__).parent.parent / "knowledge_base" / "autostream_kb.json"


def load_knowledge_base() -> dict:
    with open(KB_PATH, "r") as f:
        return json.load(f)


def build_kb_context(kb: dict) -> str:
    lines = []

    c = kb["company"]
    lines.append(f"## Company: {c['name']}")
    lines.append(c["description"])

    lines.append("\n## Pricing Plans")
    for plan in kb["pricing"]["plans"]:
        lines.append(f"\n### {plan['name']} – ${plan['price_monthly']}/month")
        for feat in plan["features"]:
            lines.append(f"  - {feat}")

    lines.append("\n## Company Policies")
    p = kb["policies"]
    lines.append(f"- Refund Policy: {p['refund']}")
    lines.append(f"- Support: {p['support']}")
    lines.append(f"- Free Trial: {p['trial']}")
    lines.append(f"- Cancellation: {p['cancellation']}")

    lines.append("\n## FAQ")
    for item in kb["faq"]:
        lines.append(f"Q: {item['question']}")
        lines.append(f"A: {item['answer']}")

    return "\n".join(lines)


_KB = load_knowledge_base()
KB_CONTEXT = build_kb_context(_KB)
