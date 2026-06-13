# V2
#!/usr/bin/env python3
"""
NIH ChestX-ray14 editing: transform one finding label into another
Process records in CSV order while skipping completed and failed images
Images are resized to 784x784 for the API and saved at 1024x1024
"""

from google import genai
from google.genai import types
from pathlib import Path
from PIL import Image
from io import BytesIO
import json
import argparse
import time
import random
import pandas as pd
import numpy as np
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


class CXRDiseaseEditor:
    def __init__(self, csv_path, image_dir, output_base, max_workers=5, max_rounds=5,
                 num_images=50, selection_strategy="mirror", top_n_composite=5):
        self.client = genai.Client()
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.output_base = Path(output_base)
        self.max_workers = max_workers
        self.max_rounds = max_rounds
        self.num_images = num_images
        self.selection_strategy = selection_strategy
        self.top_n_composite = top_n_composite
        
        # API input and final output sizes
        self.api_size = (784, 784)
        self.output_size = (1024, 1024)
        

        print(" Reading CSV file...")
        self.df = pd.read_csv(csv_path)
        print(f" Loaded {len(self.df)} records")

        print("\n Analyzing label distribution...")
        self.label_stats = self.analyze_label_distribution()
        self.print_label_statistics()

        self.output_dir = self.output_base / "CXR_edited"
        self.output_images = self.output_dir / "images"
        self.output_jsons = self.output_dir / "jsons"
        self.failed_dir = self.output_base / "CXR_failed"

        self.output_images.mkdir(parents=True, exist_ok=True)
        self.output_jsons.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)

        # Build the set of failed IDs
        self.failed_ids = self._build_failed_id_set()
        print(f"Loaded failed IDs: {len(self.failed_ids)}")
        
        print(f"  Image processing settings: API={self.api_size}, output={self.output_size}")

        distribution_file = self.output_dir / "label_distribution.json"
        with open(distribution_file, 'w', encoding='utf-8') as f:
            json.dump({
                "total_samples": len(self.df),
                "single_labels": self.label_stats['single_labels'],
                "composite_labels": self.label_stats['composite_labels'],
                "top_composite_labels": self.label_stats['top_composite_labels'],
                "all_target_labels": self.label_stats['all_target_labels'],
                "selection_strategy": self.selection_strategy,
                "api_size": self.api_size,
                "output_size": self.output_size
            }, f, indent=2)

        self.progress_file = self.output_dir / "progress.json"
        self.api_failures_file = self.output_dir / "api_failures.json"
        self.failed_summary_file = self.failed_dir / "failed_summary.json"
        self.final_prompts_file = self.output_dir / "final_prompts.json"
        self.conversations_file = self.output_dir / "all_conversations.json"
        self.processed_csv = self.output_dir / "processed_records.csv"

        self.progress = self.load_progress()
        self.api_failures = self.load_api_failures()
        self.failed_summary = self.load_failed_summary()
        self.final_prompts = self.load_final_prompts()
        self.all_conversations = self.load_all_conversations()
        self.processed_records = self.load_processed_records()

    # -----------------------
    # Image resizing
    # -----------------------
    def resize_for_api(self, image_path):
        """Resize an image to 784x784 for API processing."""
        image = Image.open(image_path)
        if image.size != self.api_size:
            image = image.resize(self.api_size, Image.Resampling.LANCZOS)
        return image
    
    def resize_for_output(self, image):
        """Resize an image to 1024x1024 for final output."""
        if image.size != self.output_size:
            image = image.resize(self.output_size, Image.Resampling.LANCZOS)
        return image

    # -----------------------
    # ID parsing and failed-output discovery
    # -----------------------
    def _get_id_base(self, image_name_or_stem: str) -> str:
        """Extract ID '00000003' from '00000003_000.jpg' or '00000003_000'. """
        stem = Path(image_name_or_stem).stem
        return stem.split('_')[0] if '_' in stem else stem

    def _build_failed_id_set(self):
        """Collect leading IDs from JPEG files in the CXR_failed directory."""
        failed_ids = set()
        if not self.failed_dir.exists():
            return failed_ids
        for p in self.failed_dir.glob("*.jpg"):
            try:
                id_base = self._get_id_base(p.stem)
                if id_base:
                    failed_ids.add(id_base)
            except Exception:
                pass
        return failed_ids

    # -----------------------
    # Distribution analysis
    # -----------------------
    def analyze_label_distribution(self):
        single_label_counts = Counter()
        composite_label_counts = Counter()

        for finding_labels in self.df['Finding Labels']:
            labels = [l.strip() for l in finding_labels.split('|')]

            if len(labels) == 1:
                single_label_counts[labels[0]] += 1
            else:
                composite_label = '|'.join(sorted(labels))
                composite_label_counts[composite_label] += 1
                for label in labels:
                    single_label_counts[label] += 1

        top_composite = dict(composite_label_counts.most_common(self.top_n_composite))
        all_target_labels = {}
        all_target_labels.update(dict(single_label_counts))
        all_target_labels.update(top_composite)

        return {
            'single_labels': dict(single_label_counts),
            'composite_labels': dict(composite_label_counts),
            'top_composite_labels': top_composite,
            'all_target_labels': all_target_labels
        }

    def print_label_statistics(self):
        print(f"\n{'='*80}")
        print("Label distribution statistics:")
        print(f"{'='*80}")

        print("\nSingle-label distribution")
        single_labels = self.label_stats['single_labels']
        total_single = sum(single_labels.values())

        sorted_single = sorted(single_labels.items(), key=lambda x: x[1], reverse=True)
        for label, count in sorted_single:
            percentage = (count / total_single) * 100 if total_single > 0 else 0
            print(f"  {label:<30s}: {count:5d} ({percentage:5.2f}%)")

        print(f"\nComposite-label distribution ({len(self.label_stats['composite_labels'])} classes)")
        print(f"  Showing top {self.top_n_composite}:")
        top_composite = self.label_stats['top_composite_labels']
        total_composite = sum(self.label_stats['composite_labels'].values()) if self.label_stats['composite_labels'] else 0

        for label, count in top_composite.items():
            percentage = (count / total_composite) * 100 if total_composite > 0 else 0
            print(f"  {label:<50s}: {count:5d} ({percentage:5.2f}%)")

        print("\nTarget-label pool")
        print(f"  Single labels: {len(self.label_stats['single_labels'])} classes")
        print(f"  Composite labels: {len(top_composite)} classes")
        print(f"  Total: {len(self.label_stats['all_target_labels'])} classes")
        print(f"{'='*80}\n")

    def find_image_file(self, row):
        """Find the image file."""
        if 'image_path' in row and pd.notna(row['image_path']):
            p = Path(row['image_path'])
            if p.exists():
                return p

        image_name = row['Image Index']
        direct_path = self.image_dir / image_name
        if direct_path.exists():
            return direct_path

        stem = Path(image_name).stem
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
            candidate = self.image_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        return None

    def normalize_label(self, label_str):
        labels = [l.strip() for l in label_str.split('|')]
        return '|'.join(sorted(labels))

    def select_target_label(self, source_label):
        source_normalized = self.normalize_label(source_label)
        available_labels = [k for k in self.label_stats['all_target_labels'].keys()
                            if self.normalize_label(k) != source_normalized]

        if not available_labels:
            return None

        if self.selection_strategy == "uniform":
            return random.choice(available_labels)
        elif self.selection_strategy in ["mirror", "original"]:
            counts = {k: self.label_stats['all_target_labels'][k] for k in available_labels}
            total = sum(counts.values())
            probs = [counts[k] / total for k in available_labels]
            return np.random.choice(available_labels, p=probs)
        elif self.selection_strategy == "balanced":
            counts = {k: self.label_stats['all_target_labels'][k] for k in available_labels}
            mx = max(counts.values())
            inv = {k: (mx - v + 1) for k, v in counts.items()}
            total_w = sum(inv.values())
            probs = [inv[k] / total_w for k in available_labels]
            return np.random.choice(available_labels, p=probs)
        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")

    # -----------------------
    # State files
    # -----------------------
    def load_progress(self):
        if self.progress_file.exists():
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                try:
                    return json.load(f)
                except:
                    return {}
        return {}

    def save_progress(self):
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, indent=2)

    def load_api_failures(self):
        if self.api_failures_file.exists():
            with open(self.api_failures_file, 'r', encoding='utf-8') as f:
                try:
                    return json.load(f)
                except:
                    return []
        return []

    def save_api_failures(self):
        with open(self.api_failures_file, 'w', encoding='utf-8') as f:
            json.dump(self.api_failures, f, indent=2, ensure_ascii=False)

    def load_failed_summary(self):
        if self.failed_summary_file.exists():
            with open(self.failed_summary_file, 'r', encoding='utf-8') as f:
                try:
                    return json.load(f)
                except:
                    return []
        return []

    def save_failed_summary(self):
        with open(self.failed_summary_file, 'w', encoding='utf-8') as f:
            json.dump(self.failed_summary, f, indent=2, ensure_ascii=False)

    def load_final_prompts(self):
        if self.final_prompts_file.exists():
            with open(self.final_prompts_file, 'r', encoding='utf-8') as f:
                try:
                    return json.load(f)
                except:
                    return {}
        return {}

    def save_final_prompts(self):
        with open(self.final_prompts_file, 'w', encoding='utf-8') as f:
            json.dump(self.final_prompts, f, indent=2, ensure_ascii=False)

    def load_all_conversations(self):
        if self.conversations_file.exists():
            with open(self.conversations_file, 'r', encoding='utf-8') as f:
                try:
                    return json.load(f)
                except:
                    return {}
        return {}

    def save_all_conversations(self):
        with open(self.conversations_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_conversations, f, indent=2, ensure_ascii=False)

    def load_processed_records(self):
        if self.processed_csv.exists():
            try:
                return pd.read_csv(self.processed_csv, encoding='utf-8')
            except:
                return pd.DataFrame()
        return pd.DataFrame()

    def save_processed_records(self):
        self.processed_records.to_csv(self.processed_csv, index=False, encoding='utf-8')

    def add_processed_record(self, row_data):
        if self.processed_records.empty:
            self.processed_records = pd.DataFrame([row_data])
        else:
            self.processed_records = pd.concat([self.processed_records, pd.DataFrame([row_data])],
                                               ignore_index=True)
        self.save_processed_records()

    # -----------------------
    # API helpers
    # -----------------------
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
        thinking, content = "", ""
        if not response.candidates:
            return "", ""
        for part in response.candidates[0].content.parts:
            if not getattr(part, "text", None):
                continue
            if getattr(part, "thought", False):
                thinking += part.text + "\n"
            else:
                content += part.text + "\n"
        return thinking.strip(), content.strip()

    # -----------------------
    # Prompt generation and refinement
    # -----------------------
    def generate_initial_prompt(self, source_disease, target_disease):
        system_instruction = f"""You are a chest X-ray image editing expert. Generate a clear, concise prompt to edit a chest X-ray image showing {source_disease} to make it appear as {target_disease}.

Requirements:
1. Easy for non-medical personnel to understand
2. Must look natural and realistic, not detectable as deepfake
3. Should follow radiological principles and medical plausibility
4. Be specific about what to change/modify to transform the findings

CRITICAL CONSTRAINTS - Fidelity Preservation:
- Preserve original image quality, noise, and contrast
- Do NOT alter DICOM-like markers (patient info, technical annotations)
- Keep the same imaging characteristics (exposure, positioning, field of view)

CRITICAL CONSTRAINTS - Negative Rules:
- NO adding text, labels, annotations, or artificial markers
- NO sharp unnatural edges or boundaries
- NO repetitive/duplicated structures or patterns
- NO introducing artifacts that look computer-generated

CRITICAL CONSTRAINTS - Minimal Change Principle:
- ONLY modify areas directly related to changing from {source_disease} to {target_disease}
- Keep lung fields, heart, bones, soft tissues, and background UNCHANGED except where necessary
- Minimal intervention: change only what's required

Return ONLY the editing prompt in English, no explanations."""
        # Calculate request size
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

    def update_prompt(self, resized_image, source_disease, target_disease, prompt_history):
        """Receive an already resized image."""
        img_byte_arr = BytesIO()
        resized_image.save(img_byte_arr, format='JPEG', quality=95)
        image_bytes = img_byte_arr.getvalue()

        history_text = ""
        for i, history in enumerate(prompt_history, 1):
            history_text += (
                f"\nAttempt {i}:\n"
                f"Prompt: {history['prompt']}\n"
                f"Verification:\n"
                f"  - Correct disease: {history['verification'].get('correct_disease')}\n"
                f"  - Structure reasonable: {history['verification'].get('structure_reasonable')}\n"
                f"  - Looks realistic: {history['verification'].get('looks_realistic')}\n"
                f"  - Reason: {history['verification'].get('reason')}\n"
            )

        system_instruction = f"""You are a chest X-ray image editing expert. Previous attempts to transform this image from {source_disease} to {target_disease} failed. Analyze the failures and generate a BETTER prompt.

HISTORY OF ALL PREVIOUS ATTEMPTS:
{history_text}

Follow the same fidelity, negative rules, and minimal change constraints as before.
Return ONLY the improved editing prompt in English."""
        # Calculate request size
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

    # -----------------------
    # Image editing and verification
    # -----------------------
    def edit_image(self, resized_image, edit_prompt):
        """Receive an already resized image."""
        # Calculate request size
        img_byte_arr = BytesIO()
        resized_image.save(img_byte_arr, format='JPEG', quality=95)
        image_bytes = img_byte_arr.getvalue()
        image_size = len(image_bytes)
        text_size = len(edit_prompt.encode('utf-8'))
        total_size = image_size + text_size
        print(f"         [edit_image] Request size: {total_size/1024:.2f} KB (image: {image_size/1024:.2f} KB, text: {text_size/1024:.2f} KB)")
        def call():
            return self.client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=[edit_prompt, resized_image]
            )
        response, error = self.api_call_with_retry(call)
        if error:
            return None, error
        if not response.candidates:
            return None, "No candidates in response"
        for part in response.candidates[0].content.parts:
            if getattr(part, "inline_data", None) is not None:
                edited_image = Image.open(BytesIO(part.inline_data.data))
                return edited_image, None
        return None, "No image generated"

    def verify_edited_image(self, edited_image, target_disease):
        img_byte_arr = BytesIO()
        edited_image.save(img_byte_arr, format='JPEG', quality=95)
        img_bytes = img_byte_arr.getvalue()

        verification_instruction = f"""You are a chest X-ray verification expert. Evaluate if this image correctly shows {target_disease}.
Check:
1) Correct disease findings
2) Anatomical structures reasonable for the diagnosis
3) Overall realism (should pass as real CXR)
Return JSON:
{{
  "qualified": true/false,
  "correct_disease": true/false,
  "structure_reasonable": true/false,
  "looks_realistic": true/false,
  "reason": "short explanation"
}}
Only qualified if all three booleans are true and no major fidelity issues."""
        # Calculate request size
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
        except Exception as e:
            return thinking, None, f"JSON parse error: {str(e)}"

    # -----------------------
    # Metadata and task processing
    # -----------------------
    def create_metadata_json(self, original_row, target_label):
        metadata = {
            "Image Index": original_row['Image Index'],
            "Finding Labels": target_label,
            "Follow-up #": int(original_row['Follow-up #']) if pd.notna(original_row['Follow-up #']) else None,
            "Patient ID": int(original_row['Patient ID']) if pd.notna(original_row['Patient ID']) else None,
            "Patient Age": int(original_row['Patient Age']) if pd.notna(original_row['Patient Age']) else None,
            "Patient Gender": original_row['Patient Gender'] if pd.notna(original_row['Patient Gender']) else None,
            "View Position": original_row['View Position'] if pd.notna(original_row['View Position']) else None,
            "edited": True,
            "original_finding_labels": original_row['Finding Labels'],
            "transformation": f"{original_row['Finding Labels']} -> {target_label}"
        }
        return metadata

    def process_single_task(self, row_idx, image_name, image_path):
        """Process one task."""
        id_base = self._get_id_base(image_name)
        if id_base in self.failed_ids:
            return {"status": "skipped", "task": image_name, "reason": "id_in_failed_dir"}

        row = self.df.iloc[row_idx]

        image_stem = Path(image_name).stem
        output_image_path = self.output_images / f"{image_stem}.jpg"
        output_json_path = self.output_jsons / f"{image_stem}.json"

        if output_image_path.exists() and output_json_path.exists():
            return {"status": "skipped", "task": image_name, "reason": "output_exists"}

        if not image_path.exists():
            return {"status": "skipped", "task": image_name, "reason": "image_not_found"}

        source_label = row['Finding Labels']
        target_label = self.select_target_label(source_label)
        if target_label is None:
            return {"status": "skipped", "task": image_name, "reason": "no_target_label"}

        task_key = f"{image_stem}_{self.normalize_label(source_label)}_to_{self.normalize_label(target_label)}"

        # Resize the image to 784x784 before processing
        resized_image = self.resize_for_api(image_path)

        conversation = {
            "image": image_name,
            "patient_id": int(row['Patient ID']) if pd.notna(row['Patient ID']) else None,
            "source_label": source_label,
            "target_label": target_label,
            "selection_strategy": self.selection_strategy,
            "status": "failed",
            "rounds": []
        }

        current_prompt = None

        for round_num in range(1, self.max_rounds + 1):
            round_data = {"round": round_num}

            if round_num == 1:
                thinking, prompt, error = self.generate_initial_prompt(source_label, target_label)
                if error:
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
                            "prompt": prev_round["generate_prompt"].get("prompt", "N/A"),
                            "verification": prev_round["verification"]
                        })
                thinking, prompt, error = self.update_prompt(resized_image, source_label, target_label, prompt_history)
                if error:
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

            edited_image, error = self.edit_image(resized_image, current_prompt)
            if error:
                round_data["edit_result"] = {"success": False, "error": error}
                conversation["rounds"].append(round_data)
                self.api_failures.append({
                    "task": task_key, "step": "edit_image", "round": round_num,
                    "error": error, "image": image_name, "source": source_label, "target": target_label
                })
                self.progress[task_key] = "api_failed"
                self.save_api_failures()
                self.save_progress()
                return {"status": "api_failed", "task": task_key, "error": error}

            round_data["edit_result"] = {"success": True}

            thinking, verification, error = self.verify_edited_image(edited_image, target_label)
            if error:
                round_data["verification"] = {"error": error}
                conversation["rounds"].append(round_data)
                self.api_failures.append({
                    "task": task_key, "step": "verify_image", "round": round_num,
                    "error": error, "image": image_name, "source": source_label, "target": target_label
                })
                self.progress[task_key] = "api_failed"
                self.save_api_failures()
                self.save_progress()
                return {"status": "api_failed", "task": task_key, "error": error}

            if verification is None:
                round_data["verification"] = {"error": "Verification returned None", "thinking_summary": thinking}
                conversation["rounds"].append(round_data)
                continue

            round_data["verification"] = verification
            round_data["verification"]["thinking_summary"] = thinking
            conversation["rounds"].append(round_data)

            if verification.get("qualified", False):
                # Resize successful edits to 1024x1024 before saving
                output_image = self.resize_for_output(edited_image)
                output_image_path = self.output_images / f"{image_stem}.jpg"
                output_image.save(output_image_path, 'JPEG', quality=95)

                metadata = self.create_metadata_json(row, target_label)
                output_json_path = self.output_jsons / f"{image_stem}.json"
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)

                conversation["status"] = "success"
                conversation["final_prompt"] = current_prompt
                conversation["final_image_path"] = str(output_image_path)
                conversation["final_json_path"] = str(output_json_path)

                self.all_conversations[task_key] = conversation
                self.final_prompts[task_key] = {
                    "image": image_name, "source": source_label, "target": target_label,
                    "status": "success", "final_prompt": current_prompt, "rounds": round_num
                }
                self.progress[task_key] = "success"
                self.save_all_conversations()
                self.save_final_prompts()
                self.save_progress()

                processed_row = row.copy()
                processed_row['Finding Labels'] = target_label
                processed_row['edited'] = True
                processed_row['original_finding_labels'] = source_label
                processed_row['transformation'] = f"{source_label} -> {target_label}"
                self.add_processed_record(processed_row)

                return {"status": "success", "task": task_key, "rounds": round_num}
            else:
                # Save rejected edits in the failure directory
                failed_image = self.resize_for_output(edited_image)
                failed_path = self.failed_dir / f"{image_stem}_{self.normalize_label(source_label)}_to_{self.normalize_label(target_label)}_failed_{round_num}.jpg"
                failed_image.save(failed_path, 'JPEG', quality=95)

        # All rounds failed
        conversation["status"] = "failed"
        conversation["final_prompt"] = current_prompt
        final_failed_image = self.resize_for_output(edited_image)
        final_failed_path = self.failed_dir / f"{image_stem}_{self.normalize_label(source_label)}_to_{self.normalize_label(target_label)}_failed_{self.max_rounds}.jpg"
        final_failed_image.save(final_failed_path, 'JPEG', quality=95)
        conversation["final_image_path"] = str(final_failed_path)

        self.failed_ids.add(id_base)

        self.all_conversations[task_key] = conversation
        last_verification = {}
        if conversation["rounds"]:
            last_verification = conversation["rounds"][-1].get("verification", {"error": "No verification"})

        self.failed_summary.append({
            "task": task_key, "image": image_name, "source": source_label, "target": target_label,
            "final_prompt": current_prompt, "rounds": self.max_rounds,
            "last_verification": last_verification,
            "final_image_path": conversation["final_image_path"]
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

    # -----------------------
    # Distribution comparison
    # -----------------------
    def analyze_edited_distribution(self):
        if not self.processed_csv.exists():
            return {}
        try:
            edited_df = pd.read_csv(self.processed_csv, encoding='utf-8')
        except:
            return {}
        label_counts = Counter()
        for finding_labels in edited_df['Finding Labels']:
            label_counts[finding_labels] += 1
        return dict(label_counts)

    def analyze_and_compare_distributions(self):
        edited_distribution = self.analyze_edited_distribution()
        if not edited_distribution:
            print("\n  No edited data found; skipping distribution comparison")
            return

        print(f"\n{'='*80}")
        print("Label distribution comparison: original vs. edited")
        print(f"{'='*80}")

        all_labels = set(self.label_stats['all_target_labels'].keys()) | set(edited_distribution.keys())
        original_total = sum(self.label_stats['all_target_labels'].values())
        edited_total = sum(edited_distribution.values())

        print(f"\n{'Label':<50} {'Original':<20} {'Edited':<20} {'Change':<10}")
        print(f"{'-'*50} {'-'*20} {'-'*20} {'-'*10}")

        sorted_labels = sorted(all_labels,
                               key=lambda x: self.label_stats['all_target_labels'].get(x, 0),
                               reverse=True)

        for label in sorted_labels:
            orig_count = self.label_stats['all_target_labels'].get(label, 0)
            orig_pct = (orig_count / original_total * 100) if original_total > 0 else 0
            edit_count = edited_distribution.get(label, 0)
            edit_pct = (edit_count / edited_total * 100) if edited_total > 0 else 0
            diff = edit_count - orig_count
            diff_str = f"{diff:+d}" if diff != 0 else "0"
            display_label = label if len(label) <= 47 else label[:44] + "..."
            print(f"{display_label:<50} "
                  f"{orig_count:4d} ({orig_pct:5.2f}%) "
                  f"{edit_count:4d} ({edit_pct:5.2f}%) "
                  f"{diff_str:>10}")

        print(f"{'-'*50} {'-'*20} {'-'*20} {'-'*10}")
        print(f"{'Total':<50} {original_total:4d} (100.00%) {edited_total:4d} (100.00%)")
        print(f"{'='*80}\n")

        comparison_file = self.output_dir / "distribution_comparison.json"
        comparison_data = {
            "original_distribution": self.label_stats['all_target_labels'],
            "edited_distribution": edited_distribution,
            "original_total": original_total,
            "edited_total": edited_total,
            "selection_strategy": self.selection_strategy,
            "changes": {
                label: {
                    "original_count": self.label_stats['all_target_labels'].get(label, 0),
                    "edited_count": edited_distribution.get(label, 0),
                    "difference": edited_distribution.get(label, 0) - self.label_stats['all_target_labels'].get(label, 0)
                }
                for label in all_labels
            }
        }
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2)
        print(f" Distribution comparison saved to: {comparison_file}\n")

    # -----------------------
    # Main workflow: process records in CSV order
    # -----------------------
    def get_all_tasks(self):
        """Get pending tasks in CSV order while skipping completed and failed records."""
        tasks_to_process = []
        
        print("\n Scanning pending tasks...")
        for idx, row in self.df.iterrows():
            image_name = row['Image Index']
            image_stem = Path(image_name).stem
            
            # Skip IDs already recorded as failed
            id_base = self._get_id_base(image_name)
            if id_base in self.failed_ids:
                continue
            
            # Skip existing outputs
            output_image_path = self.output_images / f"{image_stem}.jpg"
            output_json_path = self.output_jsons / f"{image_stem}.json"
            if output_image_path.exists() and output_json_path.exists():
                continue
            
            # Find the image file
            image_path = self.find_image_file(row)
            if image_path is None:
                print(f"     Image not found: {image_name}")
                continue
            
            # Add the task to the pending list
            tasks_to_process.append((idx, image_name, image_path))
            
            # Stop after reaching the requested count
            if len(tasks_to_process) >= self.num_images:
                break
        
        print(f"    Found {len(tasks_to_process)} pending tasks")
        return tasks_to_process

    def run(self):
        tasks = self.get_all_tasks()

        # Count completed outputs
        completed_count = 0
        for idx, row in self.df.iterrows():
            image_name = row['Image Index']
            image_stem = Path(image_name).stem
            output_image = self.output_images / f"{image_stem}.jpg"
            output_json = self.output_jsons / f"{image_stem}.json"
            if output_image.exists() and output_json.exists():
                completed_count += 1

        print(f"\n{'='*80}")
        print(f"NIH ChestX-ray14 disease editing (CSV order)")
        print(f"{'='*80}")
        print(f"CSV path: {self.csv_path}")
        print(f"Image directory: {self.image_dir}")
        print(f"Output directories: {self.output_dir}")
        print(f"Selection strategy: {self.selection_strategy}")
        print(f"API input size: {self.api_size[0]}x{self.api_size[1]}")
        print(f"Output size: {self.output_size[0]}x{self.output_size[1]}")
        print(f"Pending in this run: {len(tasks)}")
        print(f"Completed total: {completed_count}")
        print(f"Worker threads: {self.max_workers} (lock-free)")
        print(f"Maximum rounds: {self.max_rounds}")
        print(f"{'='*80}\n")

        if not tasks:
            print(" No pending tasks.")
            self.analyze_and_compare_distributions()
            return

        stats = {"success": 0, "failed": 0, "api_failed": 0, "skipped": 0}

        # Process tasks concurrently (lock-free - no locks to avoid deadlocks)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.process_single_task, row_idx, image_name, image_path): (image_name, image_path)
                for row_idx, image_name, image_path in tasks
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="images"):
                image_name, image_path = futures[future]
                try:
                    result = future.result()
                    stats[result["status"]] += 1
                except Exception as e:
                    print(f"\n Error: {image_name} - {str(e)}")
                    stats["api_failed"] += 1

        print(f"\n{'='*80}")
        print(f"Processing complete.")
        print(f"{'='*80}")
        print(f" Success: {stats['success']}")
        print(f" Failed: {stats['failed']}")
        print(f"  API failures: {stats['api_failed']}")
        print(f"  Skipped: {stats['skipped']}")
        print(f"\n Output images: {self.output_images} (size: {self.output_size[0]}x{self.output_size[1]})")
        print(f" Output JSON: {self.output_jsons}")
        print(f" Processed records: {self.processed_csv}")
        print(f" Conversation log: {self.conversations_file}")
        print(f" Final prompts: {self.final_prompts_file}")
        print(f" Failure summary: {self.failed_summary_file}")
        if self.api_failures:
            print(f"  API failure log: {self.api_failures_file}")
        print(f"{'='*80}\n")

        self.analyze_and_compare_distributions()


def main():
    parser = argparse.ArgumentParser(
        description='NIH ChestX-ray14 disease editing (CSV order; API: 784x784, output: 1024x1024)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Label selection strategies:
  mirror    : Mirror the original distribution (recommended)
  uniform   : Uniform random sampling
  balanced  : Favor underrepresented labels
  original  : Sample from the original distribution

Workflow:
  1. Read images in CSV order
  2. Skip completed outputs and previously failed images
  3. Resize images to 784x784 for API processing
  4. Resize edited images to 1024x1024 before saving
        """
    )
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--output_base', type=str, required=True)
    parser.add_argument('--max_workers', type=int, default=5)
    parser.add_argument('--max_rounds', type=int, default=3)
    parser.add_argument('--num_images', type=int, default=10)
    parser.add_argument('--selection_strategy', type=str, default='mirror',
                        choices=['mirror', 'uniform', 'balanced', 'original'])
    parser.add_argument('--top_n_composite', type=int, default=5)

    args = parser.parse_args()

    editor = CXRDiseaseEditor(
        csv_path=args.csv_path,
        image_dir=args.image_dir,
        output_base=args.output_base,
        max_workers=args.max_workers,
        max_rounds=args.max_rounds,
        num_images=args.num_images,
        selection_strategy=args.selection_strategy,
        top_n_composite=args.top_n_composite
    )
    editor.run()


if __name__ == '__main__':
    main()
