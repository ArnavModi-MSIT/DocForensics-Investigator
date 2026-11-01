from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional

BBox = tuple  # (x0, y0, x1, y1) in pixels of the page image


@dataclass
class Word:
    text: str
    bbox: BBox
    conf: float


@dataclass
class Field:
    key: str
    label: str
    raw: str
    value: object          # normalised value (str / float / 'YYYY-MM-DD')
    bbox: BBox             # location of the VALUE on the page image
    conf: float            # OCR confidence 0-1
    match: str = "fuzzy"    # how the label was matched: fuzzy | semantic
    label_sim: float = 0.0 # fuzzy ratio (0-100) or embedding cosine (0-1)


@dataclass
class Document:
    doc_id: str
    path: str
    sha256: str
    image_path: str
    doc_type: str = "unknown"
    type_conf: float = 0.0
    best_guess: str = ""       # highest-scoring type even when the page is rejected
    class_info: dict = field(default_factory=dict)   # evidence behind the classification decision
    width: int = 0
    height: int = 0
    words: list = field(default_factory=list)
    fields: list = field(default_factory=list)
    status: str = "ingested"

    def get(self, key) -> Optional[Field]:
        for f in self.fields:
            if f.key == key:
                return f
        return None


@dataclass
class Finding:
    fid: str
    kind: str              # visual | intra | cross | missing
    title: str
    severity: str          # low | medium | high
    confidence: float      # 0-1
    status: str            # observed | inconsistency | potential_manipulation
    explanation: str
    refs: list             # [{doc_id, field, value, bbox}]
    metrics: dict = field(default_factory=dict)
    method: str = ""

    def to_dict(self):
        return asdict(self)
