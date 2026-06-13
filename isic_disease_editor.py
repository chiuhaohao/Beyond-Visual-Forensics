#!/usr/bin/env python3
"""
ISIC 2019 skin lesion editing: transform one disease class into another
Supports four target-label strategies: mirror, original, balanced, and uniform

Additional options:
  --target-label  Force a target label (for example, NV, MEL, or BCC)
  --image-id      Process only the specified image ID (for example, ISIC_0034321)
"""

from google import genai
from google.genai import types
from pathlib import Path
from PIL import Image
from io import BytesIO
import json
import argparse
import time
import threading
import random
import numpy as np
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# ISIC 2019 disease mapping
DISEASE_MAPPING = {
    "MEL": "Melanoma",
    "NV": "Melanocytic nevus",
    "BCC": "Basal cell carcinoma",
    "AK": "Actinic keratosis",
    "BKL": "Benign keratosis",
    "DF": "Dermatofibroma",
    "VASC": "Vascular lesion",
    "SCC": "Squamous cell carcinoma",
    "UNK": "None of the others"
}


class ISICDiseaseEditor:
    def __init__(self, image_dir, json_dir, output_base, max_workers=20, max_rounds=5,
                 num_images=50, selection_strategy="mirror",
                 forced_target_label=None, target_image_id=None):
        """
        Initialize the ISIC skin lesion editor.

        Args:
            image_dir: ISIC image directory
            json_dir: ISIC JSON label directory
            output_base: Base output directory
            max_workers: Number of worker threads
            max_rounds: Maximum number of attempts
            num_images: Number of images to process
            selection_strategy: Label selection strategy
                - "mirror": Mirror the original distribution (recommended)
                - "original": Sample according to the original distribution
                - "balanced": Favor underrepresented target classes
                - "uniform": Sample all target classes uniformly
            forced_target_label: Force a target label such as "NV"; None uses the sampling strategy.
            target_image_id: Process only this image ID; None disables this filter.
        """
        self.client = genai.Client()
        self.image_dir = Path(image_dir)
        self.json_dir = Path(json_dir)
        self.output_base = Path(output_base)
        self.max_workers = max_workers
        self.max_rounds = max_rounds
        self.num_images = num_images
        self.selection_strategy = selection_strategy
        self.forced_target_label = forced_target_label
        self.target_image_id = target_image_id

        # Validate the forced target label
        if self.forced_target_label is not None:
            if self.forced_target_label not in DISEASE_MAPPING:
                raise ValueError(f"Invalid target label: {self.forced_target_label}. "
                                 f"Must be one of: {list(DISEASE_MAPPING.keys())}")
            print(f"  Forced target label: {self.forced_target_label} "
                  f"({DISEASE_MAPPING[self.forced_target_label]})")

        if self.target_image_id is not None:
            print(f"  Process only the specified image ID: {self.target_image_id}")

        # Analyze the dataset label distribution
        self.label_distribution = self.analyze_label_distribution()
        self.print_label_statistics()

        # Output directories
        self.output_dir = self.output_base / "ISIC_2019_edited"
        self.output_images = self.output_dir / "images"
        self.output_jsons = self.output_dir / "jsons"
        self.failed_dir = self.output_base / "ISIC_2019_failed"

        # Create output directories
        self.output_images.mkdir(parents=True, exist_ok=True)
        self.output_jsons.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)

        # Save label distribution metadata
        distribution_file = self.output_dir / "label_distribution.json"
        with open(distribution_file, 'w') as f:
            json.dump({
                "total_samples": len(list(self.json_dir.glob("*.json"))),
                "distribution": self.label_distribution,
                "selection_strategy": self.selection_strategy,
                "forced_target_label": self.forced_target_label,
                "target_image_id": self.target_image_id,
            }, f, indent=2)

        # Progress and failure records
        self.progress_file = self.output_dir / "progress.json"
        self.api_failures_file = self.output_dir / "api_failures.json"
        self.failed_summary_file = self.failed_dir / "failed_summary.json"
        self.final_prompts_file = self.output_dir / "final_prompts.json"
        self.conversations_file = self.output_dir / "all_conversations.json"

        # Load saved state
        self.progress = self.load_progress()
        self.api_failures = self.load_api_failures()
        self.failed_summary = self.load_failed_summary()
        self.final_prompts = self.load_final_prompts()
        self.all_conversations = self.load_all_conversations()

        # Thread lock
        self.lock = threading.Lock()

    def analyze_label_distribution(self):
        """Analyze the label distribution across the JSON directory."""
        label_counts = Counter()

        json_files = list(self.json_dir.glob("*.json"))
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    label = data.get("label")
                    if label:
                        label_counts[label] += 1
            except Exception as e:
                print(f"Warning: failed to read {json_file}: {e}")

        return dict(label_counts)

    def analyze_edited_distribution(self):
        """Analyze the edited label distribution."""
        label_counts = Counter()

        json_files = list(self.output_jsons.glob("*.json"))
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    label = data.get("label")
                    if label:
                        label_counts[label] += 1
            except Exception as e:
                print(f"Warning: failed to read edited file {json_file}: {e}")

        return dict(label_counts)

    def analyze_and_compare_distributions(self):
        """Compare the original and edited label distributions."""
        edited_distribution = self.analyze_edited_distribution()

        if not edited_distribution:
            print("\n  No edited data found; skipping distribution comparison")
            return

        print(f"\n{'='*80}")
        print("Label distribution comparison: original vs. edited")
        print(f"{'='*80}")

        all_labels = set(self.label_distribution.keys()) | set(edited_distribution.keys())

        original_total = sum(self.label_distribution.values())
        edited_total = sum(edited_distribution.values())

        print(f"\n{'Label':<8} {'Disease':<32} {'Original':<15} {'Edited':<15} {'Change':<10}")
        print(f"{'-'*8} {'-'*32} {'-'*15} {'-'*15} {'-'*10}")

        sorted_labels = sorted(all_labels,
                               key=lambda x: self.label_distribution.get(x, 0),
                               reverse=True)

        for label in sorted_labels:
            disease_name = DISEASE_MAPPING.get(label, "Unknown")

            orig_count = self.label_distribution.get(label, 0)
            orig_pct = (orig_count / original_total * 100) if original_total > 0 else 0

            edit_count = edited_distribution.get(label, 0)
            edit_pct = (edit_count / edited_total * 100) if edited_total > 0 else 0

            diff = edit_count - orig_count
            diff_str = f"{diff:+d}" if diff != 0 else "0"

            print(f"{label:<8} {disease_name:<32} "
                  f"{orig_count:4d} ({orig_pct:5.2f}%) "
                  f"{edit_count:4d} ({edit_pct:5.2f}%) "
                  f"{diff_str:>10}")

        print(f"{'-'*8} {'-'*32} {'-'*15} {'-'*15} {'-'*10}")
        print(f"{'Total':<8} {'':<32} {original_total:4d} (100.00%) {edited_total:4d} (100.00%)")
        print(f"{'='*80}\n")

        comparison_file = self.output_dir / "distribution_comparison.json"
        comparison_data = {
            "original_distribution": self.label_distribution,
            "edited_distribution": edited_distribution,
            "original_total": original_total,
            "edited_total": edited_total,
            "selection_strategy": self.selection_strategy,
            "forced_target_label": self.forced_target_label,
            "changes": {
                label: {
                    "original_count": self.label_distribution.get(label, 0),
                    "edited_count": edited_distribution.get(label, 0),
                    "difference": edited_distribution.get(label, 0) - self.label_distribution.get(label, 0)
                }
                for label in all_labels
            }
        }

        with open(comparison_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)

        print(f" Distribution comparison saved to: {comparison_file}\n")

    def print_label_statistics(self):
        """Print label distribution statistics."""
        print(f"\n{'='*80}")
        print("Dataset label distribution:")
        print(f"{'='*80}")

        total = sum(self.label_distribution.values())
        sorted_labels = sorted(self.label_distribution.items(),
                               key=lambda x: x[1], reverse=True)

        for label, count in sorted_labels:
            disease_name = DISEASE_MAPPING.get(label, "Unknown")
            percentage = (count / total) * 100
            print(f"{label:6s} ({disease_name:30s}): {count:5d} ({percentage:5.2f}%)")

        print(f"{'='*80}\n")

    def select_target_label(self, source_label):
        """
        Select a target label using the configured strategy.
        Return the forced target label when it differs from the source label.
        If the forced target equals the source, warn and fall back to sampling.
        """
        # Forced target
        if self.forced_target_label is not None:
            if self.forced_target_label != source_label:
                return self.forced_target_label
            else:
                print(f"  forced_target_label ({self.forced_target_label}) == source_label, "
                      f"fallback to strategy '{self.selection_strategy}'")

        # Strategy-based sampling
        available_labels = [k for k in DISEASE_MAPPING.keys() if k != source_label]

        if self.selection_strategy == "uniform":
            return random.choice(available_labels)

        elif self.selection_strategy in ("mirror", "original"):
            available_counts = {k: self.label_distribution.get(k, 1)
                                for k in available_labels}
            total = sum(available_counts.values())
            probabilities = [available_counts[k] / total for k in available_labels]
            return np.random.choice(available_labels, p=probabilities)

        elif self.selection_strategy == "balanced":
            available_counts = {k: self.label_distribution.get(k, 1)
                                for k in available_labels}
            max_count = max(available_counts.values())
            inverse_weights = {k: (max_count - v + 1) for k, v in available_counts.items()}
            total_weight = sum(inverse_weights.values())
            probabilities = [inverse_weights[k] / total_weight for k in available_labels]
            return np.random.choice(available_labels, p=probabilities)

        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")

    def load_progress(self):
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {}

    def save_progress(self):
        with self.lock:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress, f, indent=2)

    def load_api_failures(self):
        if self.api_failures_file.exists():
            with open(self.api_failures_file, 'r') as f:
                return json.load(f)
        return []

    def save_api_failures(self):
        with self.lock:
            with open(self.api_failures_file, 'w') as f:
                json.dump(self.api_failures, f, indent=2, ensure_ascii=False)

    def load_failed_summary(self):
        if self.failed_summary_file.exists():
            with open(self.failed_summary_file, 'r') as f:
                return json.load(f)
        return []

    def save_failed_summary(self):
        with self.lock:
            with open(self.failed_summary_file, 'w') as f:
                json.dump(self.failed_summary, f, indent=2, ensure_ascii=False)

    def load_final_prompts(self):
        if self.final_prompts_file.exists():
            with open(self.final_prompts_file, 'r') as f:
                return json.load(f)
        return {}

    def save_final_prompts(self):
        with self.lock:
            with open(self.final_prompts_file, 'w') as f:
                json.dump(self.final_prompts, f, indent=2, ensure_ascii=False)

    def load_all_conversations(self):
        if self.conversations_file.exists():
            with open(self.conversations_file, 'r') as f:
                return json.load(f)
        return {}

    def save_all_conversations(self):
        with self.lock:
            with open(self.conversations_file, 'w') as f:
                json.dump(self.all_conversations, f, indent=2, ensure_ascii=False)

    def api_call_with_retry(self, func, max_retries=3):
        for attempt in range(max_retries):
            try:
                return func(), None
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    return None, str(e)
        return None, "Max retries exceeded"

    def extract_thinking_and_content(self, response):
        thinking = ""
        content = ""

        for part in response.candidates[0].content.parts:
            if not part.text:
                continue
            if part.thought:
                thinking += part.text + "\n"
            else:
                content += part.text + "\n"

        return thinking.strip(), content.strip()

    def generate_initial_prompt(self, source_disease, target_disease):
        system_instruction = f"""You are a dermatology image editing expert. Generate a clear, concise prompt to edit a skin lesion image showing {source_disease} to make it appear as {target_disease}.

Requirements:
1. Easy for non-medical personnel to understand
2. Must look natural and realistic, not detectable as deepfake
3. Should follow dermatological principles and medical plausibility
4. Be specific about what to change/modify to transform the lesion appearance

CRITICAL CONSTRAINTS - Fidelity Preservation:
- Preserve original image quality, lighting, and skin texture
- Do NOT alter image borders, rulers, color calibration patches, or metadata overlays
- Keep the same imaging characteristics (focus, resolution, color balance)

CRITICAL CONSTRAINTS - Negative Rules:
- NO adding text, labels, annotations, or artificial markers
- NO sharp unnatural edges or boundaries on the lesion
- NO repetitive/duplicated structures or patterns
- NO introducing artifacts that look computer-generated
- NO obvious deepfake signs

CRITICAL CONSTRAINTS - Minimal Change Principle (Counterfactual Minimality):
- ONLY modify the lesion area to change from {source_disease} to {target_disease}
- Keep surrounding skin, background, and all other elements UNCHANGED
- Minimal intervention: change only what's necessary for the disease transformation

Return ONLY the editing prompt in English, no explanations."""

        request_size = len(system_instruction.encode('utf-8'))
        print(f"         [generate_initial_prompt] Request size: {request_size/1024:.2f} KB (text only)")

        def call():
            return self.client.models.generate_content(
                model="gemini-2.5-pro",
                contents=system_instruction,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(include_thoughts=True)
                )
            )

        response, error = self.api_call_with_retry(call)
        if error:
            return None, None, error

        thinking, prompt = self.extract_thinking_and_content(response)
        return thinking, prompt, None

    def update_prompt(self, original_image_path, source_disease, target_disease, prompt_history):
        with open(original_image_path, 'rb') as f:
            image_bytes = f.read()

        history_text = ""
        for i, history in enumerate(prompt_history, 1):
            history_text += f"""
Attempt {i}:
  Prompt: {history['prompt']}
  Verification Result:
    - Correct disease: {history['verification']['correct_disease']}
    - Structure reasonable: {history['verification']['structure_reasonable']}
    - Looks realistic: {history['verification']['looks_realistic']}
    - Reason: {history['verification']['reason']}
"""

        system_instruction = f"""You are a dermatology image editing expert. Multiple previous editing attempts to transform this skin lesion from {source_disease} to {target_disease} have failed. You need to analyze ALL previous attempts and generate a BETTER prompt.

HISTORY OF ALL PREVIOUS ATTEMPTS:
{history_text}

Looking at the ORIGINAL image (showing {source_disease}) and analyzing the patterns of failures above, generate an IMPROVED editing prompt to transform it to {target_disease}.

ANALYSIS REQUIREMENTS:
1. Identify common issues across multiple attempts
2. Learn from what didn't work in previous rounds
3. Avoid repeating the same mistakes
4. Address ALL verification issues mentioned in the history

BASIC REQUIREMENTS:
1. Easy for non-medical personnel to understand
2. Must look natural and realistic, not detectable as deepfake
3. Should follow dermatological principles and medical plausibility
4. Be specific about what to change/modify

CRITICAL CONSTRAINTS - Fidelity Preservation:
- Preserve original image quality, lighting, and skin texture
- Do NOT alter image borders, rulers, color calibration patches, or metadata overlays
- Keep the same imaging characteristics (focus, resolution, color balance)

CRITICAL CONSTRAINTS - Negative Rules:
- NO adding text, labels, annotations, or artificial markers
- NO sharp unnatural edges or boundaries on the lesion
- NO repetitive/duplicated structures or patterns
- NO introducing artifacts that look computer-generated
- NO obvious deepfake signs

CRITICAL CONSTRAINTS - Minimal Change Principle (Counterfactual Minimality):
- ONLY modify the lesion area to change from {source_disease} to {target_disease}
- Keep surrounding skin, background, and all other elements UNCHANGED
- Minimal intervention: change only what's necessary for the disease transformation

Return ONLY the improved editing prompt in English, no explanations."""

        image_size = len(image_bytes)
        text_size = len(system_instruction.encode('utf-8'))
        total_size = image_size + text_size
        print(f"         [update_prompt] Request size: {total_size/1024:.2f} KB (image: {image_size/1024:.2f} KB, text: {text_size/1024:.2f} KB)")

        def call():
            return self.client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'),
                    system_instruction
                ],
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(include_thoughts=True)
                )
            )

        response, error = self.api_call_with_retry(call)
        if error:
            return None, None, error

        thinking, prompt = self.extract_thinking_and_content(response)
        return thinking, prompt, None

    def edit_image(self, image_path, edit_prompt):
        image = Image.open(image_path)

        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=95)
        image_bytes = img_byte_arr.getvalue()
        image_size = len(image_bytes)
        text_size = len(edit_prompt.encode('utf-8'))
        total_size = image_size + text_size
        print(f"         [edit_image] Request size: {total_size/1024:.2f} KB (image: {image_size/1024:.2f} KB, text: {text_size/1024:.2f} KB)")

        def call():
            return self.client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=[edit_prompt, image]
            )

        response, error = self.api_call_with_retry(call)
        if error:
            return None, error

        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                edited_image = Image.open(BytesIO(part.inline_data.data))
                return edited_image, None

        return None, "No image generated"

    def verify_edited_image(self, edited_image, target_disease):
        img_byte_arr = BytesIO()
        edited_image.save(img_byte_arr, format='JPEG')
        img_bytes = img_byte_arr.getvalue()

        verification_instruction = f"""You are a dermatology image verification expert. Evaluate if this skin lesion image correctly shows {target_disease}.

IMPORTANT: Take your time to think carefully. Medical image editing is challenging, and minor imperfections are acceptable as long as the overall goal is achieved. Be thoughtful and balanced in your evaluation - don't reject an image for trivial issues.

Check these aspects:
1. Correct disease: Does the lesion appearance match {target_disease}? (Consider: Are the key diagnostic features present, even if not perfect?)
2. Structure reasonable: Is the lesion structure and morphology reasonable for {target_disease}? (Consider: Minor artifacts are acceptable if the overall appearance is correct)
3. Looks realistic: Does it look like a real dermatology image? (Consider: Some editing traces are inevitable; focus on whether it could pass as a real clinical photo)

Additional verification for image fidelity:
- Check if the image preserves natural skin texture and lighting (minor changes are acceptable)
- Check if there are unnatural sharp edges on the lesion (slight artifacts are tolerable if not obvious)
- Check if there are added text, artificial patterns, or deepfake artifacts (focus on major issues, not minor imperfections)
- Check if modifications are minimal (only lesion changes; surrounding skin should be largely unchanged)

Before deciding, ask yourself:
- Would this image be useful for the intended purpose despite minor flaws?
- Are the issues critical or just cosmetic?
- Does the image achieve the main goal of showing {target_disease}?

Return your evaluation in this JSON format:
{{
    "qualified": true/false,
    "correct_disease": true/false,
    "structure_reasonable": true/false,
    "looks_realistic": true/false,
    "reason": "detailed explanation (mention both strengths and weaknesses; explain your reasoning for acceptance or rejection)"
}}

Only qualified if ALL three aspects are true AND no MAJOR fidelity issues detected. Minor imperfections are acceptable."""

        image_size = len(img_bytes)
        text_size = len(verification_instruction.encode('utf-8'))
        total_size = image_size + text_size
        print(f"         [verify_edited_image] Request size: {total_size/1024:.2f} KB (image: {image_size/1024:.2f} KB, text: {text_size/1024:.2f} KB)")

        def call():
            return self.client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg'),
                    verification_instruction
                ],
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(include_thoughts=True)
                )
            )

        response, error = self.api_call_with_retry(call)
        if error:
            return None, None, error

        thinking, content = self.extract_thinking_and_content(response)

        try:
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()

            result = json.loads(content)
            return thinking, result, None
        except json.JSONDecodeError as e:
            return thinking, None, f"JSON parse error: {str(e)}"

    def create_new_json(self, original_json, target_label):
        new_json = original_json.copy()
        new_json["label"] = target_label
        new_json["labels_positive"] = [target_label]

        new_labels_vector = {k: 0.0 for k in new_json["labels_vector"].keys()}
        new_labels_vector[target_label] = 1.0
        new_json["labels_vector"] = new_labels_vector

        return new_json

    def process_single_task(self, json_path):
        with open(json_path, 'r') as f:
            original_json = json.load(f)

        image_name = original_json["image"]
        source_label = original_json["label"]
        source_disease = DISEASE_MAPPING[source_label]

        output_image_path = self.output_images / f"{image_name}.jpg"
        output_json_path = self.output_jsons / f"{image_name}.json"

        if output_image_path.exists() and output_json_path.exists():
            return {"status": "skipped", "task": image_name, "reason": "output_exists"}

        image_path = self.image_dir / f"{image_name}.jpg"
        if not image_path.exists():
            print(f"  Warning: source image not found {image_path}")
            return {"status": "skipped", "task": image_name, "reason": "image_not_found"}

        target_label = self.select_target_label(source_label)
        target_disease = DISEASE_MAPPING[target_label]

        task_key = f"{image_name}_{source_label}_to_{target_label}"

        conversation = {
            "image": image_name,
            "source_label": source_label,
            "source_disease": source_disease,
            "target_label": target_label,
            "target_disease": target_disease,
            "selection_strategy": self.selection_strategy,
            "forced_target_label": self.forced_target_label,
            "status": "failed",
            "rounds": []
        }

        current_prompt = None

        for round_num in range(1, self.max_rounds + 1):
            round_data = {"round": round_num}

            if round_num == 1:
                thinking, prompt, error = self.generate_initial_prompt(source_disease, target_disease)
                if error:
                    with self.lock:
                        self.api_failures.append({
                            "task": task_key, "step": "generate_prompt", "round": round_num,
                            "error": error, "image": image_name, "source": source_label, "target": target_label
                        })
                        self.progress[task_key] = "api_failed"
                    self.save_api_failures()
                    self.save_progress()
                    return {"status": "api_failed", "task": task_key, "error": error}

                round_data["generate_prompt"] = {"thinking_summary": thinking, "prompt": prompt}
                current_prompt = prompt
            else:
                prompt_history = []
                for prev_round in conversation["rounds"]:
                    if "generate_prompt" in prev_round and "verification" in prev_round:
                        prompt_history.append({
                            "round": prev_round["round"],
                            "prompt": prev_round["generate_prompt"]["prompt"],
                            "verification": prev_round["verification"]
                        })

                thinking, prompt, error = self.update_prompt(image_path, source_disease, target_disease, prompt_history)
                if error:
                    with self.lock:
                        self.api_failures.append({
                            "task": task_key, "step": "update_prompt", "round": round_num,
                            "error": error, "image": image_name, "source": source_label, "target": target_label
                        })
                        self.progress[task_key] = "api_failed"
                    self.save_api_failures()
                    self.save_progress()
                    return {"status": "api_failed", "task": task_key, "error": error}

                round_data["generate_prompt"] = {"thinking_summary": thinking, "prompt": prompt}
                current_prompt = prompt

            edited_image, error = self.edit_image(image_path, current_prompt)
            if error:
                round_data["edit_result"] = {"success": False, "error": error}
                conversation["rounds"].append(round_data)

                with self.lock:
                    self.api_failures.append({
                        "task": task_key, "step": "edit_image", "round": round_num,
                        "error": error, "image": image_name, "source": source_label, "target": target_label
                    })
                    self.progress[task_key] = "api_failed"
                self.save_api_failures()
                self.save_progress()
                return {"status": "api_failed", "task": task_key, "error": error}

            round_data["edit_result"] = {"success": True}

            thinking, verification, error = self.verify_edited_image(edited_image, target_disease)
            if error:
                round_data["verification"] = {"error": error}
                conversation["rounds"].append(round_data)

                with self.lock:
                    self.api_failures.append({
                        "task": task_key, "step": "verify_image", "round": round_num,
                        "error": error, "image": image_name, "source": source_label, "target": target_label
                    })
                    self.progress[task_key] = "api_failed"
                self.save_api_failures()
                self.save_progress()
                return {"status": "api_failed", "task": task_key, "error": error}

            round_data["verification"] = verification
            round_data["verification"]["thinking_summary"] = thinking
            conversation["rounds"].append(round_data)

            if verification.get("qualified", False):
                output_image_path = self.output_images / f"{image_name}.jpg"
                edited_image.save(output_image_path, 'JPEG', quality=95)

                new_json = self.create_new_json(original_json, target_label)
                output_json_path = self.output_jsons / f"{image_name}.json"
                with open(output_json_path, 'w') as f:
                    json.dump(new_json, f, indent=2)

                conversation["status"] = "success"
                conversation["final_prompt"] = current_prompt
                conversation["final_image_path"] = str(output_image_path)
                conversation["final_json_path"] = str(output_json_path)
                print(f"\nSuccess: {image_name} | {source_label} ({source_disease}) -> {target_label} ({target_disease}) | Round {round_num}")
                print(f"   Output: {output_image_path}")

                with self.lock:
                    self.all_conversations[task_key] = conversation
                    self.final_prompts[task_key] = {
                        "image": image_name, "source": source_label, "target": target_label,
                        "status": "success", "final_prompt": current_prompt, "rounds": round_num
                    }
                    self.progress[task_key] = "success"
                self.save_all_conversations()
                self.save_final_prompts()
                self.save_progress()

                return {"status": "success", "task": task_key, "rounds": round_num}
            else:
                failed_path = self.failed_dir / f"{image_name}_{source_label}_to_{target_label}_failed_{round_num}.jpg"
                edited_image.save(failed_path, 'JPEG', quality=95)

        conversation["status"] = "failed"
        conversation["final_prompt"] = current_prompt
        conversation["final_image_path"] = str(self.failed_dir / f"{image_name}_{source_label}_to_{target_label}_failed_{self.max_rounds}.jpg")
        print(f"\nFailed: {image_name} | {source_label} ({source_disease}) -> {target_label} ({target_disease}) after {self.max_rounds} rounds")

        with self.lock:
            self.all_conversations[task_key] = conversation
            self.failed_summary.append({
                "task": task_key, "image": image_name, "source": source_label, "target": target_label,
                "final_prompt": current_prompt, "rounds": self.max_rounds,
                "last_verification": conversation["rounds"][-1]["verification"],
                "final_image_path": conversation["final_image_path"],
                "full_conversation": conversation
            })
            self.final_prompts[task_key] = {
                "image": image_name, "source": source_label, "target": target_label,
                "status": "failed", "final_prompt": current_prompt, "rounds": self.max_rounds
            }
            self.progress[task_key] = "failed"

        self.save_all_conversations()
        self.save_failed_summary()
        self.save_final_prompts()
        self.save_progress()

        return {"status": "failed", "task": task_key, "rounds": self.max_rounds}

    def get_all_tasks(self):
        """
        Get pending tasks while skipping completed outputs.
        When target_image_id is set, return only that task.
        """
        if self.target_image_id is not None:
            # Single-image mode
            json_path = self.json_dir / f"{self.target_image_id}.json"
            if not json_path.exists():
                print(f" JSON file not found: {json_path}")
                return []

            # Check whether outputs already exist
            output_image = self.output_images / f"{self.target_image_id}.jpg"
            output_json  = self.output_jsons  / f"{self.target_image_id}.json"
            if output_image.exists() and output_json.exists():
                print(f"  {self.target_image_id} already has outputs; skipping. Delete them to regenerate.")
                return []

            return [json_path]

        # Batch mode
        all_json_files = sorted(list(self.json_dir.glob("*.json")))
        tasks_to_process = []

        for json_file in all_json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    image_name = data["image"]

                output_image_path = self.output_images / f"{image_name}.jpg"
                output_json_path  = self.output_jsons  / f"{image_name}.json"

                if output_image_path.exists() and output_json_path.exists():
                    continue

                tasks_to_process.append(json_file)

                if len(tasks_to_process) >= self.num_images:
                    break

            except Exception as e:
                print(f"Warning: failed to read {json_file}: {e}")
                continue

        return tasks_to_process

    def run(self):
        tasks = self.get_all_tasks()

        # Count completed outputs
        all_json_files = list(self.json_dir.glob("*.json"))
        completed_count = 0
        for json_file in all_json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    image_name = data["image"]
                output_image = self.output_images / f"{image_name}.jpg"
                output_json  = self.output_jsons  / f"{image_name}.json"
                if output_image.exists() and output_json.exists():
                    completed_count += 1
            except:
                pass

        print(f"\n{'='*80}")
        print(f"ISIC 2019 skin lesion editing")
        print(f"{'='*80}")
        print(f"Image directory: {self.image_dir}")
        print(f"JSON directory: {self.json_dir}")
        print(f"Output directories: {self.output_dir}")
        print(f"Selection strategy: {self.selection_strategy}")
        if self.forced_target_label:
            print(f"Forced target label: {self.forced_target_label} ({DISEASE_MAPPING[self.forced_target_label]})")
        if self.target_image_id:
            print(f"Selected image ID: {self.target_image_id}")
        print(f"Pending in this run: {len(tasks)}")
        print(f"Completed total: {completed_count}")
        print(f"Worker threads: {self.max_workers}")
        print(f"Maximum rounds: {self.max_rounds}")
        print(f"{'='*80}\n")

        stats = {"success": 0, "failed": 0, "api_failed": 0, "skipped": 0}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.process_single_task, json_path): json_path
                       for json_path in tasks}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                try:
                    result = future.result()
                    stats[result["status"]] += 1
                except Exception as e:
                    print(f"\nTask error: {e}")
                    stats["api_failed"] += 1

        print(f"\n{'='*80}")
        print(f"Processing complete.")
        print(f"{'='*80}")
        print(f"Success: {stats['success']}")
        print(f"Failed: {stats['failed']}")
        print(f"API failures: {stats['api_failed']}")
        print(f"Skipped: {stats['skipped']}")
        print(f"\nOutput images: {self.output_images}")
        print(f"Output JSON: {self.output_jsons}")
        print(f"Conversation log: {self.conversations_file}")
        print(f"Final prompts: {self.final_prompts_file}")
        print(f"Failure summary: {self.failed_summary_file}")
        if self.api_failures:
            print(f"API failure log: {self.api_failures_file}")
        print(f"{'='*80}\n")

        self.analyze_and_compare_distributions()


def main():
    parser = argparse.ArgumentParser(
        description='ISIC 2019 skin lesion editing: transform disease classes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Label selection strategies:
  mirror    : Mirror the original distribution (recommended)
  uniform   : Sample all target classes uniformly
  balanced  : Favor underrepresented target labels
  original  : Sample from the original distribution

Examples:
  # Process one image with NV as the target label
  python isic_disease_editor.py \\
      --image-id ISIC_0034321 \\
      --target-label NV \\
      --num-images 1 --max-workers 1

  # Batch processing with mirror sampling
  python isic_disease_editor.py --num-images 50

  # Batch processing with NV as the target label
  python isic_disease_editor.py --num-images 50 --target-label NV
        """
    )
    parser.add_argument('--image-dir', type=str, required=True,
                        help='ISIC image directory')
    parser.add_argument('--json-dir', type=str, required=True,
                        help='ISIC JSON label directory')
    parser.add_argument('--output-base', type=str, required=True,
                        help='Base output directory')
    parser.add_argument('--max-workers', type=int, default=20,
                        help='Number of worker threads (default: 20)')
    parser.add_argument('--max-rounds', type=int, default=5,
                        help='Maximum number of attempts (default: 5)')
    parser.add_argument('--num-images', type=int, default=50,
                        help='Batch mode: process the first N incomplete images (default: 50)')
    parser.add_argument('--selection-strategy', type=str,
                        default='mirror',
                        choices=['mirror', 'uniform', 'balanced', 'original'],
                        help='Label selection strategy (default: mirror)')
    # ISIC-specific options
    parser.add_argument('--target-label', type=str, default=None,
                        choices=list(DISEASE_MAPPING.keys()),
                        help='Force a target label such as NV, MEL, or BCC. '
                             'If source equals target, fall back to the sampling strategy.')
    parser.add_argument('--image-id', type=str, default=None,
                        help='Process only this image (for example, ISIC_0034321). '
                             'When set, --num-images is ignored.')

    args = parser.parse_args()

    editor = ISICDiseaseEditor(
        image_dir=args.image_dir,
        json_dir=args.json_dir,
        output_base=args.output_base,
        max_workers=args.max_workers,
        max_rounds=args.max_rounds,
        num_images=args.num_images,
        selection_strategy=args.selection_strategy,
        forced_target_label=args.target_label,
        target_image_id=args.image_id,
    )

    editor.run()


if __name__ == '__main__':
    main()
