"""
Unified Extraction Contract and Output Dataclasses for Structured IE.
Defines canonical representation for entities, directed relations, multi-role events,
and normalized factual records across all evaluated models.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
import json


@dataclass
class EntityMention:
    mention_id: str
    text: str
    entity_type: str
    char_start: int
    char_end: int
    confidence: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    provenance: str = "model"


@dataclass
class DirectedRelation:
    relation_id: str
    relation_type: str
    subject_id: str
    subject_text: str
    subject_type: str
    object_id: str
    object_text: str
    object_type: str
    confidence: float = 1.0
    polarity: str = "positive"          # positive, negative
    modality: str = "actual"            # actual, intended, conditional, hypothetical
    attribution: str = "narrator"       # narrator, dialogue, quote
    speaker: Optional[str] = None
    role: Optional[str] = None          # e.g., title/job in membership
    evidence_span: Optional[Tuple[int, int]] = None
    evidence_text: Optional[str] = None
    provenance: str = "model"


@dataclass
class EventRecord:
    record_id: str
    event_type: str
    anchor_id: Optional[str] = None
    anchor_text: Optional[str] = None
    roles: Dict[str, Any] = field(default_factory=dict)  # role_name -> entity_mention or value
    polarity: str = "positive"          # positive, negative
    modality: str = "actual"            # actual, intended, conditional
    attribution: str = "narrator"
    speaker: Optional[str] = None
    time_text: Optional[str] = None
    evidence_span: Optional[Tuple[int, int]] = None
    evidence_text: Optional[str] = None
    confidence: float = 1.0
    provenance: str = "model"


@dataclass
class ExtractionResult:
    sample_id: str
    model_name: str
    config_id: str
    text: str
    entities: List[EntityMention] = field(default_factory=list)
    relations: List[DirectedRelation] = field(default_factory=list)
    records: List[EventRecord] = field(default_factory=list)
    raw_output: Any = None
    execution_time_sec: float = 0.0
    peak_memory_mib: float = 0.0
    status: str = "COMPLETED"           # COMPLETED, FAILED_RUNTIME, UNSUPPORTED_FEATURE, etc.
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "model_name": self.model_name,
            "config_id": self.config_id,
            "entities": [asdict(e) for e in self.entities],
            "relations": [asdict(r) for r in self.relations],
            "records": [asdict(rec) for rec in self.records],
            "execution_time_sec": self.execution_time_sec,
            "peak_memory_mib": self.peak_memory_mib,
            "status": self.status,
            "error_message": self.error_message,
        }
