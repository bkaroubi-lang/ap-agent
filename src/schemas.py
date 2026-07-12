"""Pydantic v2 schemas. All I/O across the agent, tools, and eval flows through
these types so mismatches surface as validation errors, not silent bugs.

Money is stored as Decimal for exactness in arithmetic, but the JSON schema
exposed to LLM tool-calls is `{"type": "number"}` — the default Decimal
schema emits a regex with a lookahead that Groq's JSON-schema validator
rejects.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field, WithJsonSchema, field_validator

Money = Annotated[Decimal, WithJsonSchema({"type": "number"})]


class LineItem(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    description: str
    quantity: float = Field(ge=0)
    unit_price: Money
    line_total: Money


class Invoice(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    invoice_id: str
    vendor_name: str
    invoice_date: date
    due_date: Optional[date] = None
    po_reference: Optional[str] = None
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: Money
    tax: Money = Decimal("0")
    total: Money
    currency: str = "USD"

    @field_validator("currency")
    @classmethod
    def upper(cls, v: str) -> str:
        return v.upper()


class VendorRecord(BaseModel):
    """A single historical invoice record for a vendor."""
    model_config = ConfigDict(extra="forbid")
    vendor_name: str
    invoice_id: str
    invoice_date: date
    total: Money
    po_reference: Optional[str] = None


class VendorHistory(BaseModel):
    """Aggregated stats for one vendor."""
    model_config = ConfigDict(extra="forbid")
    vendor_name: str
    n_invoices: int
    mean_total: Money
    std_total: Money
    mean_days_between: float
    last_invoice_date: date
    records: list[VendorRecord]


class AnomalyCategory(str, Enum):
    DUPLICATE = "duplicate"
    AMOUNT_OUTLIER = "amount_outlier"
    OFF_CADENCE = "off_cadence"
    MISSING_PO = "missing_po"
    UNKNOWN_VENDOR = "unknown_vendor"


class Anomaly(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: AnomalyCategory
    severity: float = Field(ge=0.0, le=1.0)
    explanation: str
    evidence: dict = Field(default_factory=dict)


class AnomalyReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    invoice_id: str
    anomalies: list[Anomaly]
    is_flagged: bool

    @classmethod
    def from_anomalies(cls, invoice_id: str, anomalies: list[Anomaly]) -> "AnomalyReport":
        return cls(invoice_id=invoice_id, anomalies=anomalies, is_flagged=bool(anomalies))


class JudgeVerdict(BaseModel):
    """One judge's verdict on one item."""
    model_config = ConfigDict(extra="forbid")
    item_id: str
    judge_id: str
    score: int = Field(ge=1, le=5)             # 5-point Likert
    confidence: float = Field(ge=0.0, le=1.0)  # judge's own confidence
    rationale: str


class ExtractionEvalItem(BaseModel):
    """Per-invoice ground truth for extraction eval."""
    model_config = ConfigDict(extra="forbid")
    invoice_id: str
    invoice_text: str
    gold: Invoice


class AnomalyEvalItem(BaseModel):
    """Per-invoice ground truth for anomaly eval."""
    model_config = ConfigDict(extra="forbid")
    invoice_id: str
    injected_categories: list[AnomalyCategory]
