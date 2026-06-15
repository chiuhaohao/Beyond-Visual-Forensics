"""Prompt template for multimodal and image-only medical image evaluation."""

from typing import Optional


def get_detection_prompt(json_content: Optional[str] = None) -> str:
    """Build the evaluation prompt used for the paper experiments.

    When metadata is provided, ``FINAL ANSWER`` is the multimodal decision and
    ``VISUAL VERDICT`` is the image-only decision. Without metadata, both
    verdicts are based on the image alone.
    """
    has_metadata = bool(json_content and json_content.strip())
    metadata_section = (
        f"[METADATA]\n{json_content}\n[/METADATA]\n"
        if has_metadata
        else ""
    )

    if has_metadata:
        consistency_item = """6. Metadata/Text Consistency:
  - Provide a short, high-level summary of the most relevant 3–5 metadata/text claims (a concise paraphrase is sufficient).
  - Check whether the image is broadly consistent with those claims.
  - If there is a mismatch, state it once and explain why it matters."""
        guidance = """Guidance:
    - Consider both visual evidence and metadata/text (if provided).
    - If visual findings and metadata/text disagree, describe the discrepancy explicitly.
    - Provide your best overall judgment after weighing all available evidence.
"""
    else:
        consistency_item = """6. Overall Consistency: Check for any internal inconsistencies across regions (noise, contrast, edges, anatomy)."""
        guidance = """Guidance:
    - Consider all available visual evidence.
    - Provide your best overall judgment.
"""

    return f"""[TASK] Forensic medical image authentication. This is a medical image analysis for scientific research purposes.
{metadata_section}
You are a forensic expert. Analyze whether this medical image is REAL or FAKE (AI-generated/manipulated) and explain any artifacts you observe. You MUST complete ALL checklist items before giving your verdict. Do NOT skip the checklist.

[OUTPUT FORMAT - VERY IMPORTANT]
A) First, provide and answer the checklist below. Please keep the checklist concise: aim for one short sentence per item, avoid repetition, and keep it brief so the verdict lines at the end are not truncated.

Visual Forensics Checklist:
1. Texture Analysis: Check for unnatural smoothness (AI over-smoothing), repeated patterns, or unnatural fine-detail consistency across regions.
2. Noise Pattern: Verify consistent grain/noise and compression artifacts across the image; flag region-specific mismatches (AI creates inconsistent noise).
3. Edge Analysis: Look for halos, artifacts, or unnatural boundaries, including unnatural sharpness/blur transitions at edges and high-contrast boundaries.
4. Anatomical Plausibility: Check for impossible, distorted, or warped structures, broken continuity where structures should be continuous, or other implausible geometry.
5. Color Analysis: Detect unnatural gradients or color inconsistencies, including patchy intensity/color shifts or inconsistent illumination across regions.
{consistency_item}

{guidance}
B) After you finish the checklist above, output EXACTLY TWO lines at the END (no extra text on these two lines):

1. On the FIRST line, output ONLY the FINAL decision considering BOTH the image and the provided metadata/text (if any), in this exact format:
   FINAL ANSWER: REAL
   or
   FINAL ANSWER: FAKE

2. On the SECOND line, output ONLY the VISUAL-ONLY decision based on the image appearance alone, in this exact format:
   VISUAL VERDICT: REAL
   or
   VISUAL VERDICT: FAKE

Rules for the verdict lines:
- Output exactly two lines only, in this order: FINAL ANSWER line first, then VISUAL VERDICT line.
- Use only the word REAL or FAKE after each colon, with no additional words.
- FINAL ANSWER: overall judgment using image + metadata/text (if any).
- VISUAL VERDICT: image-only judgment; ignore all metadata/text.
""".rstrip() + "\n"
