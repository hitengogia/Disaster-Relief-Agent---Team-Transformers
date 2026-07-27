"""
ai_message.py
-------------
Generates a human-readable coordination message for a relief officer using
an LLM (Anthropic Claude). This is the core "AI/agent" component: instead of
just returning a JSON match, the agent produces a clear, actionable message
that a human can quickly read and approve/reject.

If no API key is configured (ANTHROPIC_API_KEY env var), the module falls
back to a deterministic template so the rest of the app keeps working
offline / during a live demo without network access.
"""

import os

SYSTEM_PROMPT = (
    "You are a disaster-relief coordination assistant. Given a need and a "
    "matched resource, write a short, clear message (3-4 sentences max) for "
    "a human relief coordinator to review and approve. Be factual, calm, "
    "and specific about quantity, location, and urgency. Do not invent "
    "details that were not provided. End with a clear recommended action."
)


def _template_fallback(need: dict, resource: dict, score: float) -> str:
    """Deterministic message used when no LLM API key is available."""
    return (
        f"URGENT ({need.get('urgency_level', 'N/A')}): {need['category'].title()} "
        f"needed in {need['location']} — quantity {need['quantity']}. "
        f"Suggested match: {resource['contact']} can supply {resource['quantity']} "
        f"units (match confidence {int(score * 100)}%). "
        f"Recommended action: Contact {resource['contact']} at {resource.get('phone', 'N/A')} "
        f"to confirm dispatch to {need['location']}."
    )


def generate_coordination_message(
    need: dict,
    resource: dict,
    score: float,
    rag_context: dict = None,
    groq_triage: dict = None,
) -> str:
    """
    Generate the coordination message. Tries the Anthropic API first;
    falls back to a template if the API key is missing or the call fails.
    This fallback is itself a guardrail -- the app must never crash or
    block a human decision just because the LLM call failed.

    rag_context: optional dict from src.rag.retrieve_context(need) -- adds
        live ReliefWeb reports + similar historical examples as grounding.
    groq_triage: optional dict from src.groq_triage.fast_triage(need) -- adds
        a fast independent urgency/category read as a cross-check signal.
        This is advisory context for Claude's writing, not a re-scoring of
        the need -- the authoritative urgency_score still comes from the
        trained classifier in urgency.py.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _template_fallback(need, resource, score)

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        user_content = (
            f"Need: location={need['location']}, category={need['category']}, "
            f"quantity={need['quantity']}, urgency_level={need.get('urgency_level', 'N/A')}, "
            f"description={need.get('description', 'N/A')}\n"
            f"Matched resource: contact={resource['contact']}, "
            f"quantity_available={resource['quantity']}, status={resource['status']}\n"
            f"Match confidence: {score}\n"
        )

        if groq_triage:
            user_content += (
                f"Fast triage cross-check (Groq, independent read): "
                f"urgency_hint={groq_triage.get('urgency_hint')}, "
                f"category_guess={groq_triage.get('category_guess')} "
                f"-- treat as a secondary signal, not the source of truth.\n"
            )

        if rag_context and rag_context.get("context_text"):
            user_content += (
                f"\nGrounding context (use only if genuinely relevant; do not "
                f"invent details beyond what's here):\n{rag_context['context_text']}\n"
            )

        user_content += "\nWrite the coordination message now."

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        return response.content[0].text.strip()

    except Exception as e:
        # Never let an API failure break the pipeline -- fall back gracefully.
        fallback = _template_fallback(need, resource, score)
        return f"{fallback}\n[Note: AI message generation failed ({e}); showing template instead.]"
