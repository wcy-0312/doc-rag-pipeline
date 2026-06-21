from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EvidenceItem:
    id: str
    chunk_id: str
    content: str
    retrieval_weight: float
    source_pages: List[int]
    source_tool: str
    retrieval_unit_id: Optional[str] = None
    source_doc: Optional[str] = None


@dataclass
class ClaimCitation:
    text: str
    citations: List[str]


@dataclass
class GenerationResult:
    answer: str
    claims: List[ClaimCitation]
    evidence_map: dict
    unsupported_claims: List[str]
    abstain: bool
    abstain_reason: Optional[str]
    safety_verdict: str
    steps_log: List[dict] = field(default_factory=list)  # default so existing callers are unaffected
