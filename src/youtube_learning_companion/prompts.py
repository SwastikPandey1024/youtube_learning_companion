"""Prompt engineering contracts for evidence-grounded GATE Linear Algebra study notes.

Enforces:
- Strict evidence-reference linkage (timestamps, segment snippets)
- Zero-hallucination mandate (never invent formulas or lecturer statements)
- LaTeX mathematical syntax validation (balanced delimiters, valid matrices)
- Separation of direct lecture content from supplementary explanations
- Epistemic uncertainty reporting for audio noise or ambiguous symbols
"""

from __future__ import annotations

import json
from typing import Any

from youtube_learning_companion.schemas import TranscriptSegment, YouTubeVideoMetadata

SYSTEM_SYNTHESIS_PROMPT = """You are a rigorous GATE Linear Algebra Professor inside VIDEX.
Transform bounded transcript segments into clear, accurate, evidence-grounded study notes.

=== STRICT EPISTEMIC RULES & GROUNDING CONSTRAINTS ===
1. ZERO HALLUCINATION MANDATE:
   - Extract and synthesize ONLY what is directly explained in the supplied transcript segments.
   - Do NOT invent matrix entries, vectors, theorem conditions, or worked examples.
   - Do NOT assume a topic appears merely because it belongs to general Linear Algebra.

2. EVIDENCE PROVENANCE & TIMESTAMPS:
   - Every substantive claim, definition, theorem, formula, and exam trap MUST include supporting
     timestamp interval [start_seconds, end_seconds] and a short text quote from the transcript.
   - Never cite timestamps outside the supplied interval.

3. MATHEMATICAL FIDELITY & LATEX:
   - All equations, matrices, and symbols must be written in standard, balanced LaTeX.
   - Inline math must use single dollar signs: $A \\mathbf{x} = \\mathbf{b}$, $\\lambda_i$.
   - Display equations must use double dollar signs: $$\\det(A - \\lambda I) = 0$$.
   - Delimiters ({}, [], (), $$) must be strictly balanced.

4. SEPARATION OF LECTURE CONTENT VS SUPPLEMENTARY NOTES:
   - Distinguish direct lecture claims from supplementary clarifications.
   - If a concept is ambiguous or corrupted by ASR speech noise, flag it explicitly in
     `uncertainties` with type "corrupted_notation" or "speech_noise" rather than guessing.

5. OUTPUT FORMAT:
   - Respond ONLY with a valid JSON object matching the requested schema. No markdown wrappers.
"""


def build_section_synthesis_prompt(
    video_meta: YouTubeVideoMetadata,
    segments: list[TranscriptSegment],
    window_idx: int,
    total_windows: int,
) -> list[dict[str, str]]:
    """Build grounded chat prompt for synthesizing one section from transcript segments."""
    start_sec = segments[0].start_seconds
    end_sec = segments[-1].end_seconds

    transcript_lines: list[str] = [
        f"[{s.start_seconds:.1f}s - {s.end_seconds:.1f}s] {s.text}" for s in segments
    ]
    transcript_block = "\n".join(transcript_lines)

    user_prompt = f"""Target Video: "{video_meta.title}" (ID: {video_meta.video_id})
Current Section Window: {window_idx + 1} of {total_windows} [{start_sec:.1f}s - {end_sec:.1f}s]

=== SUPPLIED TRANSCRIPT SEGMENTS ===
{transcript_block}
===================================

Analyze the transcript segments above and produce a structured technical section.
Respond ONLY with a JSON object strictly following this structure:
{{
  "title": "Precise conceptual section title",
  "overview": "Clear 2-3 sentence technical overview of what is discussed in this segment interval",
  "definitions": [
    {{
      "term": "Term name",
      "formal_definition": "Precise definition stated by the instructor",
      "notation": "Optional LaTeX e.g. $A \\\\in \\\\mathbb{{R}}^{{m \\\\times n}}$",
      "evidence_snippet": "Exact quote from transcript",
      "start_seconds": {start_sec},
      "end_seconds": {end_sec}
    }}
  ],
  "theorems": [
    {{
      "name": "Theorem or Property name",
      "statement": "Formal statement of the theorem/property",
      "assumptions": ["Assumptions stated in lecture"],
      "significance": "Why this matters in GATE / ML",
      "start_seconds": {start_sec},
      "end_seconds": {end_sec}
    }}
  ],
  "formulas": [
    {{
      "name": "Formula name",
      "latex_expression": "Balanced LaTeX equation e.g. A \\\\mathbf{{x}} = \\\\mathbf{{b}}",
      "meaning": "Geometric or algebraic interpretation",
      "variables": {{"A": "matrix", "x": "vector"}},
      "start_seconds": {start_sec},
      "end_seconds": {end_sec}
    }}
  ],
  "worked_examples": [
    {{
      "title": "Example title",
      "problem_statement": "Problem discussed",
      "step_by_step_solution": ["Step 1", "Step 2"],
      "final_answer": "Final derived result",
      "start_seconds": {start_sec},
      "end_seconds": {end_sec}
    }}
  ],
  "exam_traps": [
    {{
      "trap_description": "Common student misconception or exam pitfall mentioned",
      "why_tempting": "Why students make this mistake",
      "correct_concept": "The mathematically sound distinction",
      "start_seconds": {start_sec},
      "end_seconds": {end_sec}
    }}
  ],
  "uncertainties": [
    {{
      "uncertainty_type": "speech_noise | corrupted_notation | ambiguous_claim",
      "description": "Description of ambiguous or missing information",
      "start_seconds": {start_sec},
      "end_seconds": {end_sec}
    }}
  ],
  "takeaways": [
    {{
      "point": "Concise key takeaway point",
      "start_seconds": {start_sec},
      "end_seconds": {end_sec}
    }}
  ]
}}
"""
    return [
        {"role": "system", "content": SYSTEM_SYNTHESIS_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_consolidation_prompt(
    video_meta: YouTubeVideoMetadata,
    section_summaries: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build prompt to consolidate lecture objectives, executive summary, and practice questions."""
    sections_json = json.dumps(section_summaries, indent=2)
    user_prompt = f"""Target Video: "{video_meta.title}" (ID: {video_meta.video_id})

=== EXTRACTED LECTURE SECTIONS ===
{sections_json}
==================================

Synthesize the final consolidated elements for the full lecture note:
1. Learning Objectives (3-5 bullet points)
2. Executive Summary (comprehensive, rigorous summary paragraph)
3. Practice Questions: Distinguish questions directly derived from lecture
   statements (tag with [SOURCE-DERIVED]) from supplementary questions
   (tag with [SUPPLEMENTARY PRACTICE]).
4. Overall Key Takeaways (ordered for quick revision)

Respond ONLY with a JSON object strictly following this structure:
{{
  "learning_objectives": ["Objective 1", "Objective 2"],
  "executive_summary": "Thorough summary paragraph...",
  "practice_questions": [
    {{
      "question": "[SOURCE-DERIVED] or [SUPPLEMENTARY PRACTICE] Question statement",
      "hint": "Conceptual hint",
      "solution": "Solution or approach",
      "is_exam_style": true
    }}
  ],
  "key_takeaways": [
    {{
      "point": "Takeaway 1",
      "start_seconds": 0.0,
      "end_seconds": 1331.0
    }}
  ]
}}
"""
    return [
        {"role": "system", "content": SYSTEM_SYNTHESIS_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
