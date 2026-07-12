"""LLM-as-judge with chain-of-thought + self-reported confidence.

Two protocols:
- single: judge scores one answer against a reference on a 1-5 Likert
- pairwise: judge picks between answer A and answer B given the question

Both return JudgeVerdict objects. Confidence is elicited explicitly; downstream
calibration compares confidence to empirical accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.schemas import JudgeVerdict


JUDGE_SINGLE_PROMPT = """\
You are grading a candidate answer against a reference answer for a question.
Follow this protocol:

1. In one paragraph, reason step-by-step about what the reference is saying,
   what the candidate is saying, and where they agree or disagree.
2. Give a score from 1 to 5:
   1 = wrong or unrelated
   2 = partially wrong / misses key content
   3 = mostly right, minor issues
   4 = right, small stylistic gaps
   5 = fully correct and complete
3. State a confidence in your own score as a probability in [0, 1].

Output ONLY valid JSON with keys: rationale (string), score (int 1-5),
confidence (number in [0,1]).
"""


JUDGE_PAIRWISE_PROMPT = """\
You are choosing which of two candidate answers is better for a question. To
control for position bias, this same question will be shown to you again with
A and B swapped; be consistent based on quality, not position.

Follow this protocol:
1. In one paragraph, reason step-by-step about what each answer does well and
   what it misses.
2. Pick A or B. If they are genuinely tied, pick the one you would show to a
   user asked to choose blindly.
3. State a confidence in [0, 1] that your choice is correct.

Output ONLY valid JSON with keys: rationale (string), choice ("A" or "B"),
confidence (number in [0,1]).
"""


class _SingleJudgeReply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rationale: str
    score: int = Field(ge=1, le=5)
    confidence: float = Field(ge=0.0, le=1.0)


class _PairwiseJudgeReply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rationale: str
    choice: Literal["A", "B"]
    confidence: float = Field(ge=0.0, le=1.0)


@dataclass
class JudgeSingleItem:
    item_id: str
    question: str
    reference: str
    candidate: str


@dataclass
class JudgePairwiseItem:
    item_id: str
    question: str
    answer_a: str
    answer_b: str


def build_single_judge(model: str):
    """Return a Pydantic AI Agent whose output_type is _SingleJudgeReply."""
    from pydantic_ai import Agent

    return Agent(
        model,
        output_type=_SingleJudgeReply,
        system_prompt=JUDGE_SINGLE_PROMPT,
        model_settings={"temperature": 0.0},
    )


def build_pairwise_judge(model: str):
    from pydantic_ai import Agent

    return Agent(
        model,
        output_type=_PairwiseJudgeReply,
        system_prompt=JUDGE_PAIRWISE_PROMPT,
        model_settings={"temperature": 0.0},
    )


def judge_single(agent, item: JudgeSingleItem, judge_id: str) -> JudgeVerdict:
    prompt = (
        f"Question:\n{item.question}\n\n"
        f"Reference answer:\n{item.reference}\n\n"
        f"Candidate answer:\n{item.candidate}"
    )
    reply: _SingleJudgeReply = agent.run_sync(prompt).output
    return JudgeVerdict(
        item_id=item.item_id,
        judge_id=judge_id,
        score=reply.score,
        confidence=reply.confidence,
        rationale=reply.rationale,
    )


def judge_pairwise(agent, item: JudgePairwiseItem, judge_id: str) -> JudgeVerdict:
    prompt = (
        f"Question:\n{item.question}\n\n"
        f"Answer A:\n{item.answer_a}\n\n"
        f"Answer B:\n{item.answer_b}"
    )
    reply: _PairwiseJudgeReply = agent.run_sync(prompt).output
    # Encode A->5, B->1 so we can reuse Likert-oriented calibration downstream.
    score = 5 if reply.choice == "A" else 1
    return JudgeVerdict(
        item_id=item.item_id,
        judge_id=judge_id,
        score=score,
        confidence=reply.confidence,
        rationale=reply.rationale,
    )
