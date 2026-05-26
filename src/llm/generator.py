import re
from typing import List, Dict
from textwrap import shorten


def _sources_numbered(hits: List[Dict]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(
            f"[{i}] {h.get('source', 'Unknown source')} "
            f"(effective {h.get('effective_from', 'n/a')})"
        )
    return "\n".join(f"- {line}" for line in lines)


def _sentences(text: str) -> List[str]:
    sents = re.split(r'(?<=[.!?])\s+', (text or "").strip())
    return [s.strip() for s in sents if len(s.strip()) >= 30]


def _paragraph_no_llm(hits: List[Dict], max_sents: int = 5) -> str:
    pool = []
    for h in hits[:5]:
        pool.extend(_sentences(h.get("text", "")))

    out, seen = [], set()
    for s in pool:
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
        if len(out) >= max_sents:
            break
    return " ".join(out) if out else "No clear policy sentences were found."


def _bullets(hits: List[Dict], max_points: int = 6) -> str:
    points = []
    for h in hits[:max_points]:
        t = (h.get("text") or "").strip()
        t = shorten(t, width=300, placeholder="…")
        points.append(f"• {t}" if t else "• (empty snippet)")
    return "\n".join(points) if points else "• _No clear snippets found._"


def _should_fallback(hits: List[Dict]) -> bool:
    if not hits:
        return True
    top_score = max((h.get("score", 0.0) for h in hits), default=0.0)
    return top_score < 0.12


def _fallback_text(query: str) -> str:
    return (
        f"**Question:** {query}\n\n"
        "_I’m sorry, I couldn’t confidently locate a matching policy in the hotel HR knowledge base. "
        "For the most accurate guidance, please contact your HR team or HR help desk._"
    )


def generate_answer(query: str, hits: List[Dict], style: str = "bullets") -> str:
    """
    style: 'bullets' | 'paragraph'
    - bullets   -> extractive bullet list from policy text
    - paragraph -> concise paragraph summary from retrieved snippets
    """
    if _should_fallback(hits):
        return _fallback_text(query)

    if style == "paragraph":
        body = _paragraph_no_llm(hits)
    else:
        body = _bullets(hits)

    return (
        f"**Question:** {query}\n\n"
        f"{body}\n\n"
        f"**Sources:**\n{_sources_numbered(hits)}"
    )
