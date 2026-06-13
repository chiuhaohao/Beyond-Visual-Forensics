"""Prompt template for multimodal and image-only medical image evaluation."""


def get_detection_prompt(json_content: str | None = None) -> str:
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
        consistency_item = """6. Metadata Consistency: Summarize the most relevant metadata claims and assess whether the image is consistent with them. State any meaningful mismatch."""
        final_answer_rule = """- FINAL ANSWER must use both the image and the provided metadata.
- VISUAL VERDICT must use only the image and must ignore the metadata."""
    else:
        consistency_item = """6. Overall Consistency: Check for internal inconsistencies across image regions, including noise, contrast, edges, and anatomy."""
        final_answer_rule = """- No metadata is provided, so both decisions must be based only on the image.
- FINAL ANSWER and VISUAL VERDICT should therefore be identical."""

    return f"""[TASK] Forensic medical image authentication for scientific research.
{metadata_section}
Determine whether the medical image is REAL or FAKE (AI-generated or manipulated). Complete the concise checklist before giving the verdicts.

Visual Forensics Checklist:
1. Texture Analysis: Check for unnatural smoothness, repeated patterns, or inconsistent fine detail.
2. Noise Pattern: Check whether grain, noise, and compression artifacts are spatially consistent.
3. Edge Analysis: Look for halos, unnatural boundaries, or inconsistent sharpness and blur.
4. Anatomical Plausibility: Check for distorted structures, broken continuity, or implausible geometry.
5. Color and Intensity: Look for unnatural gradients, patchy shifts, or inconsistent illumination.
{consistency_item}

After the checklist, output exactly these two lines at the end:
FINAL ANSWER: REAL or FINAL ANSWER: FAKE
VISUAL VERDICT: REAL or VISUAL VERDICT: FAKE

Verdict rules:
{final_answer_rule}
- Use only REAL or FAKE after each colon.
- Do not add text after the two verdict lines.
""".strip() + "\n"
