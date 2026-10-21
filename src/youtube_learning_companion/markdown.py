"""Markdown and Obsidian-compatible exporter for VIDEX study notes.

Generates:
- Publication-ready, structured Markdown with clickable YouTube timestamp links
- Obsidian YAML frontmatter, wiki-links, and callouts
- Resumable playlist index (00_Playlist_Index.md)
- Global topic index (00_Topic_Index.md) cross-referencing mathematical concepts across videos
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from youtube_learning_companion.schemas import (
    EvidenceReference,
    PlaylistProcessingManifest,
    StudyNote,
)


def format_seconds(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    total_sec = int(seconds)
    hrs = total_sec // 3600
    mins = (total_sec % 3600) // 60
    secs = total_sec % 60
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def sanitize_filename(name: str) -> str:
    """Convert string to safe filename."""
    s = re.sub(r'[\\/*?:"<>|\[\]{}]', "", name)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s[:120]


def format_evidence_reference(
    reference: EvidenceReference, expected_video_id: str | None = None
) -> str:
    """Render a clickable evidence timestamp and its source segment identifier."""
    detail = f"; segment `{reference.segment_id}`" if reference.segment_id else ""
    if expected_video_id is not None and reference.video_id != expected_video_id:
        return f"(unverified reference for video `{reference.video_id}`{detail})"
    url = reference.clickable_url
    if url is None:
        return f"({reference.timestamp_label}{detail})"
    if not reference.end_timestamp_is_source:
        detail = f"{detail}; inferred end time" if detail else "; inferred end time"
    return f"([{reference.timestamp_label}]({url}){detail})"


def _yaml_string(value: str) -> str:
    """Encode a string as a YAML-compatible JSON-style quoted scalar."""
    return json.dumps(value, ensure_ascii=True)


class MarkdownExporter:
    """Exports StudyNote and Playlist manifests to publication-ready Markdown."""

    def __init__(self, enable_obsidian: bool = True) -> None:
        self.enable_obsidian = enable_obsidian

    def export_note_to_markdown(self, note: StudyNote) -> str:
        """Convert a StudyNote into clean, evidence-grounded Markdown."""
        lines: list[str] = []

        # 1. Frontmatter
        if self.enable_obsidian:
            lines.append("---")
            lines.append(f"title: {_yaml_string(note.video_title)}")
            lines.append(f"video_id: {_yaml_string(note.video_id)}")
            lines.append(f"source_url: {_yaml_string(note.source_url)}")
            if note.playlist_id:
                lines.append(f"playlist_id: {_yaml_string(note.playlist_id)}")
            if note.channel_title:
                lines.append(f"lecturer: {_yaml_string(note.channel_title)}")
            lines.append(f"duration: {_yaml_string(format_seconds(note.duration_seconds))}")
            lines.append(f"coverage: {_yaml_string(f'{note.coverage_percentage}%')}")
            lines.append(
                f"temporal_coverage: {_yaml_string(f'{note.temporal_coverage_percentage}%')}"
            )
            lines.append(
                f"density_coverage: {_yaml_string(f'{note.density_coverage_percentage}%')}"
            )
            lines.append(
                f"transcript_completeness: {_yaml_string(note.transcript_completeness)}"
            )
            lines.append(
                "transcript_completeness_rationale: "
                f"{_yaml_string(note.transcript_completeness_rationale)}"
            )
            lines.append(f"generated_at: {_yaml_string(note.generated_at.isoformat())}")
            provenance = note.generation_metadata.get("acquisition_method", "unknown")
            lines.append(f"input_provenance: {_yaml_string(str(provenance))}")
            lines.append(f"status: {_yaml_string(note.status.value)}")
            synth_mode = note.generation_metadata.get("synthesizer_mode", "deterministic")
            lines.append(f"synthesis_mode: {_yaml_string(str(synth_mode))}")
            if note.generation_metadata.get("is_mock"):
                lines.append("is_mock: true")
            if "llm_provider" in note.generation_metadata:
                lines.append(
                    f"llm_provider: {_yaml_string(str(note.generation_metadata['llm_provider']))}"
                )
            model_id = (
                note.generation_metadata.get("model_id")
                or note.generation_metadata.get("llm_model")
            )
            if model_id:
                lines.append(f"model_id: {_yaml_string(str(model_id))}")
            lines.append("tags:")
            lines.append("  - gate-data-science")
            lines.append("  - linear-algebra")
            lines.append("  - youtube-learning-companion")
            lines.append("---\n")

        # 2. Header
        lines.append(f"# {note.video_title}\n")
        lines.append(
            f"**Source Video:** [{note.source_url}]({note.source_url}) | "
            f"**Duration:** {format_seconds(note.duration_seconds)} | "
            f"**Temporal Coverage:** {note.temporal_coverage_percentage}% "
            f"({note.transcript_completeness.replace('_', ' ').title()})\n"
        )

        # Status badge if incomplete or abridged
        if note.transcript_completeness != "complete":
            coverage_label = note.transcript_completeness.replace("_", " ").upper()
            lines.append(
                f"> [!warning] Transcript Coverage: {coverage_label}\n"
                f"> {note.transcript_completeness_rationale}\n"
            )
        elif note.status.value != "complete":
            lines.append(
                f"> [!warning] Processing Status: {note.status.value.upper()}\n"
                f"> {note.executive_summary}\n"
            )

        # 3. Learning Objectives
        if note.learning_objectives:
            lines.append("## 🎯 Learning Objectives\n")
            for obj in note.learning_objectives:
                lines.append(f"- {obj}")
            lines.append("")

        # 4. Executive Summary
        lines.append("## 📌 Executive Summary\n")
        lines.append(f"{note.executive_summary}\n")

        # 5. Thematic Sections with Clickable Timestamps
        if note.sections:
            lines.append("## 📖 Lecture Notes & Concepts\n")
            for idx, sec in enumerate(note.sections, start=1):
                start_label = format_seconds(sec.start_seconds)
                if sec.start_timestamp_is_source:
                    click_url = (
                        f"https://youtube.com/watch?v={note.video_id}"
                        f"&t={int(sec.start_seconds)}s"
                    )
                    start_markup = f"[{start_label}]({click_url})"
                else:
                    start_markup = f"{start_label} (inferred timing)"
                lines.append(f"### {idx}. {sec.title} {start_markup}\n")
                lines.append(f"{sec.overview}\n")

                # Definitions
                for d in sec.definitions:
                    ts_link = ""
                    if d.evidence_refs:
                        ts_link = (
                            f" {format_evidence_reference(d.evidence_refs[0], note.video_id)}"
                        )
                    lines.append(f"> [!note] Definition: **{d.term}**{ts_link}")
                    lines.append(f"> {d.definition}")
                    if d.notation:
                        lines.append(f"> \n> **Notation:** {d.notation}")
                    if d.assumptions:
                        lines.append(f"> \n> **Assumptions:** {', '.join(d.assumptions)}")
                    lines.append("")

                # Theorems
                for thm in sec.theorems:
                    ts_link = ""
                    if thm.evidence_refs:
                        ts_link = (
                            f" {format_evidence_reference(thm.evidence_refs[0], note.video_id)}"
                        )
                    lines.append(f"> [!abstract] Theorem: **{thm.name}**{ts_link}")
                    lines.append(f"> **Statement:** {thm.statement}")
                    if thm.conditions:
                        lines.append(f"> \n> **Conditions:** {', '.join(thm.conditions)}")
                    lines.append(f"> \n> **Conclusion:** {thm.conclusion}")
                    lines.append("")

                # Formulas
                for f in sec.formulas:
                    ts_link = ""
                    if f.evidence_refs:
                        ts_link = (
                            f" {format_evidence_reference(f.evidence_refs[0], note.video_id)}"
                        )
                    lines.append(f"#### 📐 Formula: {f.name}{ts_link}\n")
                    lines.append(f"$$\n{f.latex_expression.strip('$')}\n$$\n")
                    if f.variables:
                        lines.append("**Variables:**")
                        for var_k, var_desc in f.variables.items():
                            lines.append(f"- `${var_k}$`: {var_desc}")
                        lines.append("")

                # Worked Examples
                for ex in sec.worked_examples:
                    ts_link = ""
                    if ex.evidence_refs:
                        ts_link = (
                            f" {format_evidence_reference(ex.evidence_refs[0], note.video_id)}"
                        )
                    lines.append(f"> [!example] Worked Example{ts_link}")
                    lines.append(f"> **Problem:** {ex.problem_statement}")
                    if ex.step_by_step_solution:
                        lines.append("> \n> **Solution:**")
                        for step in ex.step_by_step_solution:
                            lines.append(f"> - {step}")
                    lines.append(f"> \n> **Result:** {ex.final_result}")
                    if ex.key_insight:
                        lines.append(f"> \n> **Insight:** {ex.key_insight}")
                    lines.append("")

                # Exam Traps
                for trap in sec.exam_traps:
                    ts_link = ""
                    if trap.evidence_refs:
                        ts_link = (
                            f" {format_evidence_reference(trap.evidence_refs[0], note.video_id)}"
                        )
                    origin = (
                        "Direct from lecture"
                        if trap.is_directly_stated_in_lecture
                        else "Conceptually inferred"
                    )
                    lines.append(f"> [!caution] GATE Exam Trap ({origin}){ts_link}")
                    lines.append(f"> **Mistake / Pitfall:** {trap.trap_description}")
                    lines.append(f"> \n> **Correct Concept:** {trap.correct_understanding}")
                    lines.append(f"> \n> **Why Tested in GATE:** {trap.why_it_matters_in_gate}")
                    lines.append("")

        # 6. Key Takeaways
        if note.key_takeaways:
            lines.append("## 💡 High-Yield Revision Takeaways\n")
            for t in note.key_takeaways:
                ts_link = ""
                if t.evidence_refs:
                    ts_link = f" {format_evidence_reference(t.evidence_refs[0], note.video_id)}"
                lines.append(f"- **[{t.importance.upper()}]** {t.takeaway}{ts_link}")
            lines.append("")

        # 7. Practice Questions
        if note.practice_questions:
            lines.append("## ❓ Source-Grounded Practice Questions\n")
            for idx, q in enumerate(note.practice_questions, start=1):
                ts_link = ""
                if q.evidence_refs:
                    ts_link = f" {format_evidence_reference(q.evidence_refs[0], note.video_id)}"
                lines.append(f"**Q{idx} ({q.question_type.title()}):** {q.question}{ts_link}")
                lines.append("<details><summary>Click to view Answer / Hint</summary>")
                lines.append(f"\n> {q.answer_or_hint}\n")
                lines.append("</details>\n")

        # 8. Uncertainties and Missing Inputs
        if note.uncertainties:
            lines.append("## ⚠️ Epistemic Uncertainties & Content Notes\n")
            for u in note.uncertainties:
                lines.append(f"> [!warning] {u.uncertainty_type.value.replace('_', ' ').title()}")
                lines.append(f"> **Scope:** {u.affected_concept_or_timestamp}")
                lines.append(f"> \n> **Detail:** {u.description}")
                lines.append(f"> \n> **Action Needed:** {u.suggested_resolution}")
                lines.append("")

        # 9. Validation Audit Summary Footer
        if note.validation:
            v = note.validation
            synth_mode = note.generation_metadata.get("synthesizer_mode", "deterministic")
            lines.append("---\n")
            lines.append(
                f"*(YouTube Learning Companion grounding audit: "
                f"{v.grounded_claims}/{v.total_claims} section claims grounded "
                f"({v.grounding_ratio * 100:.1f}%), LaTeX valid: {v.formulas_valid_latex}, "
                f"Synthesis Mode: {synth_mode}, Status: {note.status.value})*\n"
            )
            if v.warnings:
                lines.append("\n> [!note] Synthesizer Audit Notes:")
                for w in v.warnings:
                    lines.append(f"> - {w}")
                lines.append("")

        return "\n".join(lines)

    def write_note_file(
        self, note: StudyNote, output_dir: Path | str, *, overwrite: bool = False
    ) -> Path:
        """Write a Markdown note, refusing to replace an existing file by default."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        fname = f"{sanitize_filename(note.video_title)}.md"
        file_path = out_path / fname
        if file_path.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing note: {file_path}. "
                "Choose another output directory or pass overwrite=True."
            )
        content = self.export_note_to_markdown(note)
        file_path.write_text(content, encoding="utf-8")
        return file_path

    def generate_playlist_index(
        self, manifest: PlaylistProcessingManifest, notes: list[StudyNote]
    ) -> str:
        """Generate a complete 00_Playlist_Index.md linking to all notes and source videos."""
        lines: list[str] = []
        lines.append(f"# 📚 {manifest.playlist_title}\n")
        lines.append(
            f"**Playlist URL:** [{manifest.source_url}]({manifest.source_url}) | "
            f"**Total Discovered:** {manifest.total_videos_discovered} videos | "
            f"**Generated Notes:** {manifest.processed_count} | "
            f"**Awaiting Transcripts:** {manifest.awaiting_transcript_count}\n"
        )
        lines.append("## 📑 Course Syllabus & Video Directory\n")
        lines.append("| # | Video Title | Duration | Status | Study Note | Source Link |")
        lines.append("|---|---|---|---|---|---|")

        notes_by_id = {n.video_id: n for n in notes}

        for r in manifest.records:
            note = notes_by_id.get(r.video_id)
            dur_str = format_seconds(note.duration_seconds) if note else "-"
            status_badge = f"`{r.status.value}`"
            note_link = f"[[{sanitize_filename(r.title)}]]" if r.note_file_path else "*Awaiting*"
            source_link = f"[Watch Video]({r.source_url})"
            row = (
                f"| {r.position + 1} | {r.title} | {dur_str} | {status_badge} | "
                f"{note_link} | {source_link} |"
            )
            lines.append(row)

        lines.append("\n---\n*Generated by VIDEX YouTube Learning Companion*")
        return "\n".join(lines)

    def generate_topic_index(self, notes: list[StudyNote]) -> str:
        """Generate 00_Topic_Index.md cross-linking concepts across multiple lectures."""
        lines: list[str] = []
        lines.append("# 🔍 GATE Linear Algebra — Topic & Concept Index\n")
        lines.append(
            "Cross-video index linking definitions, theorems, and key formulas across the course.\n"
        )

        topics_map: dict[str, list[tuple[str, str, str]]] = {}

        for note in notes:
            clean_title = sanitize_filename(note.video_title)
            for sec in note.sections:
                for d in sec.definitions:
                    topics_map.setdefault(d.term, []).append(
                        (note.video_title, clean_title, "Definition")
                    )
                for t in sec.theorems:
                    topics_map.setdefault(t.name, []).append(
                        (note.video_title, clean_title, "Theorem")
                    )
                for f in sec.formulas:
                    topics_map.setdefault(f.name, []).append(
                        (note.video_title, clean_title, "Formula")
                    )

        if not topics_map:
            lines.append("*No concepts indexed yet.*")
            return "\n".join(lines)

        for concept in sorted(topics_map.keys()):
            occurrences = topics_map[concept]
            lines.append(f"### {concept}\n")
            for vid_title, fname, item_type in occurrences:
                lines.append(f"- **{item_type}** in [[{fname}|{vid_title}]]")
            lines.append("")

        return "\n".join(lines)
