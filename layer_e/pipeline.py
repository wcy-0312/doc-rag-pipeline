import logging
from typing import Optional

from .models import ClaimCitation, GenerationResult
from . import context_packer, prompt_builder, attribution, guardrail
from .llm_client import get_llm_client

logger = logging.getLogger(__name__)

_UNSUPPORTED_MARKER = "unsupported"


def generate(
    query: str,
    ranked_results: list,
    llm_client=None,
    skip_unsupported_check: bool = False,
    abstention_threshold: float = 0.10,
) -> GenerationResult:
    if llm_client is None:
        llm_client = get_llm_client()

    should_abstain, reason = guardrail.check_abstention(ranked_results, threshold=abstention_threshold)
    if should_abstain:
        return GenerationResult(
            answer="",
            claims=[],
            evidence_map={},
            unsupported_claims=[],
            abstain=True,
            abstain_reason=reason,
            safety_verdict="abstained",
        )

    evidence_list, evidence_map = context_packer.pack(ranked_results)
    prompt = prompt_builder.build(evidence_list, query)

    image_paths = context_packer.collect_image_paths(evidence_map)
    if image_paths:
        messages = prompt_builder.build_multimodal_messages(
            prompt["system"], prompt["user"], image_paths
        )
        llm_output = llm_client.generate_multimodal(messages)
    else:
        llm_output = llm_client.generate(system=prompt["system"], user=prompt["user"])

    # Filter [unsupported] out of citations — it's a model signal, not a valid evidence ID.
    # Claims carrying [unsupported] are collected directly into unsupported_claims.
    prompt_unsupported: list[str] = []
    claims: list[ClaimCitation] = []
    for c in llm_output.get("claims", []):
        raw_citations = c.get("citations", [])
        if _UNSUPPORTED_MARKER in raw_citations:
            prompt_unsupported.append(c["text"])
        valid_citations = [cid for cid in raw_citations if cid != _UNSUPPORTED_MARKER]
        claims.append(ClaimCitation(text=c["text"], citations=valid_citations))

    answer = llm_output.get("answer", "")
    llm_abstain = llm_output.get("abstain", False)
    llm_abstain_reason = llm_output.get("abstain_reason", None)

    invalid_ids = attribution.validate_citations(claims, evidence_map)
    if invalid_ids:
        logger.warning("Invalid citation IDs: %s", invalid_ids)

    if skip_unsupported_check:
        unsupported_claims = prompt_unsupported
    else:
        judge_unsupported = attribution.detect_unsupported_claims(claims, evidence_list, llm_client)
        unsupported_claims = list(dict.fromkeys(prompt_unsupported + judge_unsupported))

    # Claims with no valid citations (and not already flagged via [unsupported]) are implicitly unsupported
    empty_citation_claims = [
        c.text for c in claims
        if not c.citations and c.text not in prompt_unsupported
    ]
    if empty_citation_claims:
        unsupported_claims = list(dict.fromkeys(unsupported_claims + empty_citation_claims))

    safety_verdict = guardrail.compute_safety_verdict(False, unsupported_claims, llm_abstain)

    return GenerationResult(
        answer=answer,
        claims=claims,
        evidence_map=evidence_map,
        unsupported_claims=unsupported_claims,
        abstain=llm_abstain,
        abstain_reason=llm_abstain_reason if llm_abstain else None,
        safety_verdict=safety_verdict,
    )


class GenerationPipeline:
    """Stateful wrapper around generate() for external callers (e.g. E2E eval)."""

    def __init__(
        self,
        llm_client=None,
        skip_unsupported_check: bool = False,
        abstention_threshold: float = 0.10,
    ):
        self._llm_client = llm_client or get_llm_client()
        self._skip_unsupported_check = skip_unsupported_check
        self._abstention_threshold = abstention_threshold

    def run(self, query: str, ranked_results: list) -> GenerationResult:
        return generate(
            query=query,
            ranked_results=ranked_results,
            llm_client=self._llm_client,
            skip_unsupported_check=self._skip_unsupported_check,
            abstention_threshold=self._abstention_threshold,
        )
