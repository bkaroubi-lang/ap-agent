"""Pydantic AI agent that answers Q&A over the vendor history + invoice
processing tools. For deterministic extraction/anomaly evals, prefer calling
`build_extraction_agent` (src/tools.py) or `flag_anomaly_rules` directly —
this agent is for the free-form Q&A leg that the LLM-as-judge evaluates.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING

from src.schemas import AnomalyReport, Invoice, VendorHistory
from src.store import VendorStore
from src.tools import flag_anomaly_rules, retrieve_vendor_history, spend_in_window

if TYPE_CHECKING:
    from pydantic_ai import Agent, RunContext


AP_AGENT_SYSTEM_PROMPT = """\
You are an accounts-payable assistant for a small business. Answer questions
about vendors, invoices, and spend using the tools provided. Be concise;
prefer numeric answers to prose when the user asks a numeric question. If a
tool returns no data, say so instead of guessing.
"""


@dataclass
class AgentDeps:
    store: VendorStore


def build_ap_agent(model: str, temperature: float = 0.0):
    from pydantic_ai import Agent, RunContext

    agent = Agent(
        model,
        deps_type=AgentDeps,
        system_prompt=AP_AGENT_SYSTEM_PROMPT,
        model_settings={"temperature": temperature},
    )

    @agent.tool
    def get_vendor_history(ctx: RunContext[AgentDeps], vendor_name: str) -> VendorHistory | None:
        """Return aggregated history for a vendor: n_invoices, mean/std amount,
        cadence, and per-invoice records. Returns None if the vendor is unknown."""
        return retrieve_vendor_history(vendor_name, ctx.deps.store)

    @agent.tool
    def get_vendor_spend(
        ctx: RunContext[AgentDeps], vendor_name: str, start: date, end: date
    ) -> Decimal:
        """Total spend on a vendor between start and end (inclusive)."""
        return spend_in_window(vendor_name, start, end, ctx.deps.store)

    @agent.tool
    def check_invoice_anomalies(ctx: RunContext[AgentDeps], invoice: Invoice) -> AnomalyReport:
        """Run the deterministic anomaly rule pass on an invoice against the
        vendor store, returning any flagged anomalies."""
        return flag_anomaly_rules(invoice, ctx.deps.store)

    @agent.tool
    def list_vendors(ctx: RunContext[AgentDeps]) -> list[str]:
        """List all known vendor names."""
        return ctx.deps.store.all_vendors()

    return agent
