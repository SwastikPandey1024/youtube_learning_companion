# Human-Reviewed Quality Evaluation

## Status

Quality scoring is blocked: this repository currently contains no human-reviewed reference annotations from a complete, legally obtained lecture transcript. Existing unit and integration tests are software checks, not ground truth. Do not report accuracy scores from the condensed fixture.

## Annotation Procedure

1. Record the source URL, transcript file hash, transcript acquisition method, declared coverage status, duration, and reviewer/date separately from generated notes.
2. Review the transcript against its authorized source. Do not infer missing words or formulas. Mark unintelligible or visually dependent content as uncertain.
3. Create a reference concept inventory from the transcript, with each concept linked to exact transcript segment IDs and source time ranges. Mark importance as critical, major, or supporting before inspecting generated notes.
4. Independently review generated notes against the transcript and reference inventory. Annotate each substantive claim with supported, partly supported, unsupported, or not assessable, plus evidence IDs and severity.
5. Keep reference annotations immutable and separate from generated output. Resolve reviewer disagreements before calculating metrics; retain adjudication notes.

## Annotation Record Format

Store UTF-8 JSON Lines in a manually managed evaluation directory, one record per reviewed lecture. The template below defines fields only; it is not a ground-truth example and must not be scored until filled by reviewers.

```json
{
  "record_type": "lecture_reference",
  "video_id": "",
  "source_url": "",
  "transcript_sha256": "",
  "transcript_source": "",
  "transcript_coverage_status": "complete|partial|unknown",
  "coverage_rationale": "",
  "duration_seconds": null,
  "reviewers": [],
  "reviewed_at": "",
  "concepts": [
    {
      "concept_id": "",
      "label": "",
      "importance": "critical|major|supporting",
      "evidence": [
        {"segment_id": "", "start_seconds": null, "end_seconds": null}
      ],
      "technical_review": "correct|incorrect|uncertain|not_applicable",
      "reviewer_note": ""
    }
  ],
  "note_claim_reviews": [
    {
      "claim_id": "",
      "claim_text": "",
      "support": "supported|partly_supported|unsupported|not_assessable",
      "severity": "critical|major|minor",
      "evidence_segment_ids": [],
      "reviewer_note": ""
    }
  ],
  "timestamp_reviews": [
    {
      "note_timestamp_seconds": null,
      "relevant_source_start_seconds": null,
      "relevant_source_end_seconds": null,
      "within_tolerance": null,
      "reviewer_note": ""
    }
  ],
  "coverage_label_correct": null,
  "reproducibility_config": {
    "provider": "",
    "model": "",
    "prompt_version": "",
    "generation_parameters": {}
  }
}
```

## Metric Rules

Report dimensions independently per lecture and across the set; never hide a weak dimension in an aggregate score.

- **Claim support:** supported claims divided by assessable substantive claims. Report partly supported and not assessable counts separately. Also report unsupported claims by severity.
- **Concept recall:** weighted reference concepts captured in the note divided by all weighted reference concepts. Weights are fixed during annotation: critical 3, major 2, supporting 1. A concept counts as captured only when its meaning is correct and it has relevant evidence.
- **Technical correctness:** reviewer counts and severity of incorrect, uncertain, and not-assessable definitions, theorems, formulas, derivations, and examples. Do not substitute automated LaTeX balance for mathematical correctness.
- **Timestamp accuracy:** percentage of reviewed note links landing within the reviewer-marked relevant source interval plus a declared tolerance. Report tolerance and unassessable links.
- **Omission severity:** count and weighted sum of reference concepts omitted, grouped by importance.
- **Hallucination severity:** count and weighted sum of unsupported claims, grouped by critical/major/minor severity. Suggested weights: 5/3/1, fixed before review.
- **Coverage honesty:** fraction of notes whose coverage label matches the independently reviewed source coverage. Density and temporal span alone cannot establish completeness.
- **Reproducibility:** rerun the same transcript and recorded configuration; compare normalized outputs and evidence references. Report exact-match rate and explicitly list expected nondeterminism. This is not a factual-accuracy metric.

## Reporting

Publish the annotation source and consent/legal basis, number of lectures and reviewers, adjudication method, metric denominators, per-dimension results, and limitations. A missing denominator is reported as not measured, not zero. Do not use the test fixture as a reviewed reference set.
