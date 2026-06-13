"""Run medical image manipulation detection with local or API-based VLMs."""

import argparse
import base64
import copy
import json
import os
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from abc import ABC, abstractmethod

import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import prompts

IMG_EXTS = (".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

FINAL_ANSWER_RE = re.compile(
    r"(?:final\s*answer|overall\s+verdict|overall\s+judge?ment)"
    r"\s*(?:\([^)]*\))?\s*[:：\-]\s*\[?\s*(real|fake)\s*\]?\b",
    flags=re.IGNORECASE,
)
FINAL_ANSWER_LINE_RE = re.compile(
    r"(?:final\s*answer|overall\s+verdict|overall\s+judge?ment)"
    r"\s*(?:\([^)]*\))?\s*[:：\-]\s*(?P<tail>[^\n\r]{0,500})",
    flags=re.IGNORECASE,
)
VISUAL_VERDICT_RE = re.compile(
    r"(?:visual\s+verdict|visual[-\s]*only\s*(?:decision|verdict|judge?ment)?|"
    r"image[-\s]*only\s*(?:decision|verdict|judge?ment)?)"
    r"\s*(?:\([^)]*\))?\s*[:：\-]\s*\[?\s*(real|fake)\s*\]?\b",
    flags=re.IGNORECASE,
)
VISUAL_VERDICT_LINE_RE = re.compile(
    r"(?:visual\s+verdict|visual[-\s]*only\s*(?:decision|verdict|judge?ment)?|"
    r"image[-\s]*only\s*(?:decision|verdict|judge?ment)?)"
    r"\s*(?:\([^)]*\))?\s*[:：\-]\s*(?P<tail>[^\n\r]{0,500})",
    flags=re.IGNORECASE,
)
SENTENCE_LABEL_RE = re.compile(
    r"\b(?:this|the\s+image|the\s+medical\s+image)\s+(?:is|appears|seems)\s+"
    r"(?:a\s+)?(real|fake|authentic|genuine|synthetic|generated|manipulated|edited)\b",
    flags=re.IGNORECASE,
)
DIRECT_LABEL_RE = re.compile(
    r"\b(real|fake|authentic|genuine|synthetic|generated|manipulated|edited)\b",
    flags=re.IGNORECASE,
)

LLAMA3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if loop.first and messages[0]['role'] != 'system' %}"
    "{{ '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 23 July 2024\n\n<|eot_id|>' }}"
    "{% endif %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<|start_header_id|>user<|end_header_id|>\n\n' }}"
    "{% if message['content'] is string %}"
    "{{ message['content'] }}"
    "{% else %}"
    "{% for content in message['content'] %}"
    "{% if content['type'] == 'image' %}{{ '<|image|>' }}{% elif content['type'] == 'text' %}{{ content['text'] }}{% endif %}"
    "{% endfor %}"
    "{% endif %}"
    "{{ '<|eot_id|>' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
    "{% endif %}"
    "{% endfor %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
)

def load_frozen_ids(csv_path: str) -> Set[str]:
    """Load frozen evaluation image IDs from a CSV."""
    if not csv_path:
        return set()
    import csv
    frozen = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        key_col = "image_id"
        if key_col not in fieldnames:
            raise ValueError(f"Frozen CSV must contain an '{key_col}' column: {csv_path}")
        for row in reader:
            frozen.add(row[key_col].strip())
    print(f"[FrozenCSV] Loaded {len(frozen)} entries (key='{key_col}') from {csv_path}")
    return frozen

def load_image_any(img_dir: str, image_id: str) -> str:
    """Return an existing image path for the given image_id."""
    base_name = os.path.splitext(image_id)[0]
    for ext in IMG_EXTS:
        p = os.path.join(img_dir, base_name + ext)
        if os.path.exists(p):
            return p
    return ""

def discover_pairs(json_dir: str, images_dir: str, limit: int, verbose: bool = False,
                   modality: str = "isic", frozen_ids: Set[str] = None) -> List[Tuple[str, str, str]]:
    """Discover (image_id, json_path, image_path) pairs for multimodal mode.
    If frozen_ids is non-empty, only include pairs whose image_id is in frozen_ids.
    """
    pairs = []
    missing_count = 0
    frozen_filter = bool(frozen_ids)  # True if we should filter

    if modality == "pedi_cxr":
        if os.path.isfile(json_dir):
            json_file = Path(json_dir)
        elif os.path.isdir(json_dir):
            json_files = sorted(Path(json_dir).glob("*.json"))
            if not json_files:
                print(f"[Discovery] No JSON files found in {json_dir}")
                return pairs
            json_file = json_files[0]
        else:
            print(f"[ERROR] Invalid json_dir path: {json_dir}")
            return pairs
        
        print(f"[Discovery] Loading pediatric CXR data from {json_file}")
        
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                all_cases = json.load(f)
            
            if not isinstance(all_cases, list):
                print(f"[ERROR] Expected JSON array, got {type(all_cases)}")
                return pairs
            
            print(f"[Discovery] Found {len(all_cases)} cases in JSON")
            
            for idx, case in enumerate(all_cases):
                if len(pairs) >= limit > 0:
                    break

                if "disk_path" not in case:
                    if verbose:
                        print(f"[WARN] Case {idx} missing 'disk_path' field")
                    missing_count += 1
                    continue
                
                disk_path = case["disk_path"]
                
                if os.path.isabs(disk_path):
                    try:
                        relative_path = os.path.relpath(disk_path, images_dir)
                    except ValueError:
                        parts = Path(disk_path).parts
                        if len(parts) >= 2:
                            relative_path = os.path.join(parts[-2], parts[-1])
                        else:
                            relative_path = os.path.basename(disk_path)
                else:
                    relative_path = disk_path
                
                img_path = os.path.join(images_dir, relative_path)
                
                if not os.path.exists(img_path):
                    if verbose:
                        print(f"[WARN] Image not found: {img_path}")
                    missing_count += 1
                    continue
                
                image_id = os.path.basename(disk_path)

                if frozen_filter and image_id not in frozen_ids:
                    continue

                case_identifier = f"{json_file.stem}_case_{idx}"
                pairs.append((image_id, case_identifier, img_path))
            
            print(f"[Discovery] Successfully matched {len(pairs)} cases (Missing: {missing_count})")
            
        except Exception as e:
            print(f"[ERROR] Failed to load {json_file.name}: {e}")
            import traceback
            traceback.print_exc()
            return pairs

    else:
        json_files = sorted(Path(json_dir).glob("*.json"))
        if limit > 0 and not frozen_filter:
            json_files = json_files[:limit]
        
        print(f"[Discovery] Found {len(json_files)} JSON files")
        
        for jf in json_files:
            if len(pairs) >= limit > 0 and not frozen_filter:
                break

            image_id = jf.stem
            img_path = ""
            
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
                    
                    if modality == "cxr":
                        if "Image Index" in json_data:
                            image_id = json_data["Image Index"]
                        img_path = load_image_any(images_dir, image_id)
                        
                    elif modality == "isic":
                        if "image" in json_data:
                            image_id = json_data["image"]
                        img_path = load_image_any(images_dir, image_id)
                            
            except Exception as e:
                if verbose:
                    print(f"[WARN] Failed to read JSON {jf.name}: {e}")
                continue

            if frozen_filter and image_id not in frozen_ids:
                continue
            
            if not img_path or not os.path.exists(img_path):
                if verbose:
                    print(f"[WARN] Image not found for {jf.name}: {img_path}")
                missing_count += 1
                continue
                
            pairs.append((image_id, str(jf), img_path))

            if not frozen_filter and len(pairs) >= limit > 0:
                break
        
        print(f"[Discovery] Successfully matched {len(pairs)} JSON-image pairs (Missing images: {missing_count})")
    
    return pairs

def clean_cxr_json(rec: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Remove CXR-specific fields that reveal ground truth labels."""
    cleaned = copy.deepcopy(rec)
    fields_to_remove = ['edited', 'original_finding_labels', 'transformation']
    
    removed_fields = []
    for field in fields_to_remove:
        if field in cleaned:
            cleaned.pop(field)
            removed_fields.append(field)
    
    return cleaned, removed_fields

def clean_pedi_cxr_json(rec: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Remove pediatric CXR-specific fields that reveal ground truth labels or file paths."""
    cleaned = copy.deepcopy(rec)
    fields_to_remove = ['disk_path', 'edited', 'original_finding_labels', 'transformation']
    
    removed_fields = []
    for field in fields_to_remove:
        if field in cleaned:
            cleaned.pop(field)
            removed_fields.append(field)
    
    return cleaned, removed_fields

def discover_images(images_dir: str, limit: int, json_dir: str = None, 
                   verbose: bool = False, modality: str = "isic",
                   frozen_ids: Set[str] = None) -> List[Tuple[str, str]]:
    """Discover images for image-only mode.
    If frozen_ids is non-empty, only include images whose image_id is in frozen_ids.
    """
    pairs = []
    frozen_filter = bool(frozen_ids)
    
    if json_dir and modality == "pedi_cxr":
        if os.path.isfile(json_dir):
            json_file = Path(json_dir)
        elif os.path.isdir(json_dir):
            json_files = sorted(Path(json_dir).glob("*.json"))
            if not json_files:
                print(f"[Discovery] No JSON files found in {json_dir}")
                return pairs
            json_file = json_files[0]
        else:
            print(f"[ERROR] Invalid json_dir path: {json_dir}")
            return pairs
        
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                all_cases = json.load(f)
            
            if not isinstance(all_cases, list):
                return pairs
            
            for case in all_cases:
                if len(pairs) >= limit > 0:
                    break

                if "disk_path" not in case:
                    continue
                
                disk_path = case["disk_path"]
                if os.path.isabs(disk_path):
                    try:
                        relative_path = os.path.relpath(disk_path, images_dir)
                    except ValueError:
                        parts = Path(disk_path).parts
                        if len(parts) >= 2:
                            relative_path = os.path.join(parts[-2], parts[-1])
                        else:
                            relative_path = os.path.basename(disk_path)
                else:
                    relative_path = disk_path
                
                img_path = os.path.join(images_dir, relative_path)
                
                if os.path.exists(img_path):
                    image_id = os.path.basename(disk_path)
                    if frozen_filter and image_id not in frozen_ids:
                        continue
                    pairs.append((image_id, img_path))
        except Exception as e:
            print(f"[ERROR] Failed to load {json_file.name}: {e}")

    elif json_dir:
        json_files = sorted(Path(json_dir).glob("*.json"))
        
        for jf in json_files:
            if len(pairs) >= limit > 0 and not frozen_filter:
                break

            image_id = jf.stem
            img_path = ""
            
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
                    
                    if modality == "cxr" and "Image Index" in json_data:
                        image_id = json_data["Image Index"]
                        img_path = load_image_any(images_dir, image_id)
                        
                    elif modality == "isic" and "image" in json_data:
                        image_id = json_data["image"]
                        img_path = load_image_any(images_dir, image_id)
                    else:
                        img_path = load_image_any(images_dir, image_id)
            except:
                continue
            
            if frozen_filter and image_id not in frozen_ids:
                continue

            if img_path and os.path.exists(img_path):
                pairs.append((image_id, img_path))
    else:
        all_images = []
        for ext in IMG_EXTS:
            all_images.extend(Path(images_dir).glob(f"*{ext}"))
        all_images = sorted(all_images)

        for img_path in all_images:
            image_id = img_path.stem
            if frozen_filter and image_id not in frozen_ids:
                continue
            pairs.append((image_id, str(img_path)))
            if len(pairs) >= limit > 0:
                break
    
    print(f"[Discovery] Found {len(pairs)} images")
    return pairs

def normalize_label_word(label: str) -> Tuple[str, bool]:
    label = (label or "").strip().lower()
    if label in ("real", "authentic", "genuine"):
        return "REAL", True
    if label in ("fake", "synthetic", "generated", "manipulated", "edited"):
        return "FAKE", True
    return "ERROR", False


def extract_from_labeled_line(text: str, line_pattern: re.Pattern) -> Tuple[str, bool]:
    for match in reversed(list(line_pattern.finditer(text))):
        tail = (match.group("tail") or "").strip()
        if not tail:
            continue

        sentence_matches = list(SENTENCE_LABEL_RE.finditer(tail))
        if sentence_matches:
            return normalize_label_word(sentence_matches[-1].group(1))

        direct_matches = list(DIRECT_LABEL_RE.finditer(tail))
        if direct_matches:
            return normalize_label_word(direct_matches[-1].group(1))

    return "ERROR", False


def extract_labeled_answer(
    raw_output: Any,
    direct_pattern: re.Pattern,
    line_pattern: re.Pattern,
    allow_global_sentence_fallback: bool = False,
) -> Tuple[str, bool]:
    if raw_output is None:
        return "ERROR", False

    text = str(raw_output).strip()
    if not text:
        return "ERROR", False

    matches = list(direct_pattern.finditer(text))
    if matches:
        return matches[-1].group(1).upper(), True

    label, ok = extract_from_labeled_line(text, line_pattern)
    if ok:
        return label, True

    if allow_global_sentence_fallback:
        sentence_matches = list(SENTENCE_LABEL_RE.finditer(text))
        if sentence_matches:
            return normalize_label_word(sentence_matches[-1].group(1))

    return "ERROR", False


def extract_final_answer(raw_output: Any) -> Tuple[str, bool]:
    return extract_labeled_answer(
        raw_output,
        direct_pattern=FINAL_ANSWER_RE,
        line_pattern=FINAL_ANSWER_LINE_RE,
        allow_global_sentence_fallback=True,
    )


def extract_visual_verdict(raw_output: Any) -> Tuple[str, bool]:
    return extract_labeled_answer(
        raw_output,
        direct_pattern=VISUAL_VERDICT_RE,
        line_pattern=VISUAL_VERDICT_LINE_RE,
        allow_global_sentence_fallback=False,
    )

def resolve_source_value(preset: str | None, custom_text: str | None) -> str | None:
    """Convert preset or custom text to actual Source value."""
    if preset:
        mapping = {
            "nano": "Edited by Nano Banana (AI Editing Technique)",
            "hospital": "Hospital",
        }
        return mapping.get(preset)
    return custom_text

def ddp_env():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    return world_size, local_rank, rank

def ddp_init():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        return True
    return False

def ddp_barrier():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        import torch.distributed as dist
        dist.barrier()

def ddp_destroy():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()

class BaseVLM(ABC):
    def __init__(self, model_path: str, device: torch.device, dtype: torch.dtype, rank: int):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.rank = rank
        self.model = None
        self.processor = None
    
    @abstractmethod
    def load_model(self): pass
    
    @abstractmethod
    def generate(self, image_path: str, prompt_text: str, 
                 max_new_tokens: int = 256, temperature: float = 0.0) -> str: pass


def encode_image_base64(image_path: str, max_size: int = 2048) -> str:
    image = Image.open(image_path).convert("RGB")
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple(int(dimension * ratio) for dimension in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class BaseAPIModel(BaseVLM):
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name, torch.device("cpu"), torch.float32, rank=0)
        self.model_name = model_name
        self.api_key = api_key
        self.rate_limit_delay = 1.0

    def load_model(self):
        return None

    def generate_batch(
        self,
        image_paths: List[str],
        prompt_texts: List[str],
        max_new_tokens: int = 500,
        temperature: float = 0.0,
    ) -> List[str]:
        results = []
        for index, (image_path, prompt_text) in enumerate(
            zip(image_paths, prompt_texts), start=1
        ):
            results.append(
                self.generate(
                    image_path,
                    prompt_text,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            )
            if index < len(image_paths):
                time.sleep(self.rate_limit_delay)
        return results


class OpenAIVisionAPI(BaseAPIModel):
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name, api_key)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("OpenAI API support requires the 'openai' package") from exc
        self.client = OpenAI(api_key=api_key)
        print(f"[Model] Initialized OpenAI model {model_name}")

    def generate(
        self,
        image_path: str,
        prompt_text: str,
        max_new_tokens: int = 500,
        temperature: float = 0.0,
    ) -> str:
        try:
            encoded_image = encode_image_base64(image_path)
            model_name_lower = self.model_name.lower()
            uses_completion_tokens = (
                model_name_lower.startswith("o1")
                or model_name_lower.startswith("gpt-5")
                or "o1" in model_name_lower
            )
            api_params = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded_image}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
            }
            token_parameter = (
                "max_completion_tokens" if uses_completion_tokens else "max_tokens"
            )
            api_params[token_parameter] = max_new_tokens
            if not uses_completion_tokens:
                api_params["temperature"] = temperature

            response = self.client.chat.completions.create(**api_params)
            if not getattr(response, "choices", None):
                return "ERROR_NO_CHOICES"

            choice = response.choices[0]
            if choice.finish_reason == "content_filter":
                return "BLOCKED_CONTENT_FILTER"

            content = choice.message.content
            if content is None:
                return "ERROR_NULL_CONTENT"
            content = content.strip()
            if not content:
                return "ERROR_EMPTY_CONTENT"
            return content + " [TRUNCATED]" if choice.finish_reason == "length" else content
        except Exception as exc:
            error_message = str(exc).lower()
            if "rate_limit" in error_message or "429" in error_message:
                return "ERROR_RATE_LIMIT"
            if "quota" in error_message:
                return "ERROR_QUOTA_EXCEEDED"
            if "invalid_api_key" in error_message or "401" in error_message:
                return "ERROR_INVALID_API_KEY"
            if "timeout" in error_message:
                return "ERROR_TIMEOUT"
            print(f"[ERROR] OpenAI API: {exc}")
            return "ERROR_UNEXPECTED"

    def generate_batch(
        self,
        image_paths: List[str],
        prompt_texts: List[str],
        max_new_tokens: int = 500,
        temperature: float = 0.0,
    ) -> List[str]:
        results = []
        for index, (image_path, prompt_text) in enumerate(
            zip(image_paths, prompt_texts), start=1
        ):
            result = "ERROR_MAX_RETRIES"
            for retry_count in range(3):
                result = self.generate(
                    image_path,
                    prompt_text,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                if result != "ERROR_RATE_LIMIT":
                    break
                if retry_count < 2:
                    time.sleep((retry_count + 1) * 5)
            results.append(result)
            if index < len(image_paths):
                time.sleep(self.rate_limit_delay)
        return results


class GeminiAPI(BaseAPIModel):
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name, api_key)
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise RuntimeError(
                "Gemini API support requires the 'google-generativeai' package"
            ) from exc
        genai.configure(api_key=api_key)
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        self.model = genai.GenerativeModel(
            model_name, safety_settings=self.safety_settings
        )
        self.rate_limit_delay = 2.0
        print(f"[Model] Initialized Gemini model {model_name}")

    def generate(
        self,
        image_path: str,
        prompt_text: str,
        max_new_tokens: int = 500,
        temperature: float = 0.0,
    ) -> str:
        try:
            image = Image.open(image_path).convert("RGB")
            if max(image.size) > 2048:
                ratio = 2048 / max(image.size)
                new_size = tuple(int(dimension * ratio) for dimension in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            response = self.model.generate_content(
                [prompt_text, image],
                generation_config={
                    "max_output_tokens": max_new_tokens,
                    "temperature": temperature,
                },
                safety_settings=self.safety_settings,
            )
            return self._extract_response_text(response)
        except Exception as exc:
            error_message = str(exc).lower()
            if "quota" in error_message or "429" in error_message:
                return "ERROR_QUOTA_EXCEEDED"
            if "invalid argument" in error_message:
                return "ERROR_INVALID_ARGUMENT"
            if "api_key" in error_message:
                return "ERROR_API_KEY"
            print(f"[ERROR] Gemini API: {exc}")
            return "ERROR_UNEXPECTED"

    @staticmethod
    def _extract_response_text(response) -> str:
        prompt_feedback = getattr(response, "prompt_feedback", None)
        if prompt_feedback and getattr(prompt_feedback, "block_reason", None):
            return "BLOCKED_PROMPT"

        candidates = getattr(response, "candidates", None)
        if not candidates:
            return "ERROR_NO_CANDIDATES"

        candidate = candidates[0]
        text_parts = []
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                text_parts.append(part_text.strip())
        extracted_text = " ".join(text_parts).strip()

        finish_reason = getattr(candidate, "finish_reason", None)
        finish_reason_value = (
            finish_reason
            if isinstance(finish_reason, int)
            else getattr(finish_reason, "value", 0)
        )
        if finish_reason_value == 3:
            return "BLOCKED_SAFETY"
        if finish_reason_value == 4:
            return "BLOCKED_RECITATION"
        if finish_reason_value == 2:
            return (
                extracted_text + " [TRUNCATED]"
                if extracted_text
                else "ERROR_MAX_TOKENS_NO_TEXT"
            )
        return extracted_text or "ERROR_NO_TEXT"


class MedGemmaVLM(BaseVLM):
    def __init__(self, model_path: str, device: torch.device, dtype: torch.dtype, rank: int, use_multi_gpu: bool = False):
        super().__init__(model_path, device, dtype, rank)
        self.use_multi_gpu = use_multi_gpu
        self.first_device = None
    
    def load_model(self):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        if self.rank == 0: print(f"[Model] Loading MedGemma from {self.model_path}")
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_path, token=os.environ.get("HF_TOKEN", None), local_files_only=os.path.exists(os.path.join(self.model_path, "config.json")))
            if self.use_multi_gpu:
                num_gpus = torch.cuda.device_count()
                max_memory = {i: "20GiB" for i in range(num_gpus)}
                self.model = AutoModelForImageTextToText.from_pretrained(self.model_path, torch_dtype=self.dtype, device_map="auto", max_memory=max_memory, low_cpu_mem_usage=True, token=os.environ.get("HF_TOKEN", None), local_files_only=os.path.exists(os.path.join(self.model_path, "config.json")))
                self.first_device = list(self.model.hf_device_map.values())[0] if hasattr(self.model, "hf_device_map") else self.device
            else:
                self.model = AutoModelForImageTextToText.from_pretrained(self.model_path, torch_dtype=self.dtype, token=os.environ.get("HF_TOKEN", None), local_files_only=os.path.exists(os.path.join(self.model_path, "config.json"))).to(self.device)
                self.first_device = self.device
            self.model.eval()
            if self.rank == 0: print(f"[Model] MedGemma loaded successfully")
        except Exception as e: raise RuntimeError(f"MedGemma load failed: {e}")

    def generate(self, image_path: str, prompt_text: str, 
                 max_new_tokens: int = 256, temperature: float = 0.0) -> str:
        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}, {"type": "image", "image": image}]}]
        prompt_str = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        batch = self.processor(text=prompt_str, images=image, return_tensors="pt", padding=True, truncation=True)
        target_device = self.first_device if self.use_multi_gpu else self.device
        batch = {k: v.to(target_device) for k, v in batch.items()}
        
        with torch.no_grad():
            gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0}
            if temperature > 0: gen_kwargs["temperature"] = temperature
            out_ids = self.model.generate(**batch, **gen_kwargs)
        
        input_len = batch["input_ids"].shape[-1]
        text = self.processor.decode(out_ids[0][input_len:], skip_special_tokens=True).strip()
        
        if not text:
            full_text = self.processor.decode(out_ids[0], skip_special_tokens=True)
            if "model\n" in full_text: text = full_text.split("model\n")[-1].strip()
            else: text = full_text.strip()
        return text
    
    def generate_batch(self, image_paths: List[str], prompt_texts: List[str],
                       max_new_tokens: int = 256, temperature: float = 0.0) -> List[str]:
        images = [Image.open(p).convert("RGB") for p in image_paths]
        batch_messages = [[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": img}]}] for img, prompt in zip(images, prompt_texts)]
        
        inputs = self.processor.apply_chat_template(batch_messages, add_generation_prompt=True, tokenize=True, padding=True, return_dict=True, return_tensors="pt")
        target_device = self.first_device if self.use_multi_gpu else self.device
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor): inputs[k] = v.to(target_device, dtype=self.dtype if v.dtype.is_floating_point else None)

        with torch.no_grad():
            gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0}
            if temperature > 0: gen_kwargs["temperature"] = temperature
            out_ids = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[-1]
        results = []
        for i in range(len(out_ids)):
            text = self.processor.decode(out_ids[i][input_len:], skip_special_tokens=True).strip()
            if not text:
                full_text = self.processor.decode(out_ids[i], skip_special_tokens=True)
                if "model\n" in full_text: text = full_text.split("model\n")[-1].strip()
                else: text = full_text.strip()
            results.append(text)
        return results

class Lingshu7BVLM(BaseVLM):
    def __init__(self, model_path: str, device: torch.device, dtype: torch.dtype, rank: int, use_multi_gpu: bool = False):
        super().__init__(model_path, device, dtype, rank)
        self.use_multi_gpu = use_multi_gpu
        self.first_device = None
        self._max_input_length = None

    def _infer_max_input_length(self) -> int:
        env_v = os.environ.get("LINGSHU_MAX_INPUT_LENGTH", "").strip()
        if env_v.isdigit():
            return int(env_v)
        candidates = []
        try:
            cfg = getattr(self.model, "config", None)
            for k in ["max_position_embeddings", "max_seq_len", "seq_length", "model_max_length"]:
                v = getattr(cfg, k, None)
                if isinstance(v, int) and v > 0:
                    candidates.append(v)
        except: pass
        try:
            tok = getattr(self.processor, "tokenizer", None)
            v = getattr(tok, "model_max_length", None)
            if isinstance(v, int) and v > 0:
                candidates.append(v)
        except: pass
        sane = [v for v in candidates if v < 1000000]
        base = min(sane) if sane else 4096
        base = min(base, 8192)
        base = max(base, 2048)
        return base

    def _safe_trim_prompt_text(self, prompt_text: str, reserve: int = 768, keep_tail: bool = True) -> str:
        tok = getattr(self.processor, "tokenizer", None)
        if tok is None:
            return prompt_text
        max_ctx = int(self._max_input_length or 4096)
        budget = max(256, max_ctx - int(reserve))
        ids = tok(prompt_text, add_special_tokens=False).get("input_ids", [])
        if len(ids) <= budget:
            return prompt_text
        trimmed_ids = ids[-budget:] if keep_tail else ids[:budget]
        return tok.decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def _build_gen_kwargs(self, max_new_tokens: int, temperature: float):
        tok = getattr(self.processor, "tokenizer", None)
        eos_id = getattr(tok, "eos_token_id", None)
        pad_id = getattr(tok, "pad_token_id", None)
        if pad_id is None:
            pad_id = eos_id
        gen_kwargs = {"max_new_tokens": int(max_new_tokens), "min_new_tokens": 2, "do_sample": bool(temperature > 0)}
        if temperature > 0: gen_kwargs["temperature"] = float(temperature)
        if eos_id is not None: gen_kwargs["eos_token_id"] = eos_id
        if pad_id is not None: gen_kwargs["pad_token_id"] = pad_id
        return gen_kwargs

    def load_model(self):
        from transformers import AutoModelForVision2Seq, AutoProcessor
        if self.rank == 0: print(f"[Model] Loading Lingshu-7B from {self.model_path}")
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True, local_files_only=os.path.exists(os.path.join(self.model_path, "config.json")))
            if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
                tok = self.processor.tokenizer
                tok.padding_side = "left"
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                    tok.pad_token_id = tok.eos_token_id
            if self.use_multi_gpu:
                num_gpus = torch.cuda.device_count()
                max_memory = {i: "20GiB" for i in range(num_gpus)}
                self.model = AutoModelForVision2Seq.from_pretrained(self.model_path, torch_dtype=self.dtype, device_map="auto", max_memory=max_memory, trust_remote_code=True, low_cpu_mem_usage=True, local_files_only=os.path.exists(os.path.join(self.model_path, "config.json")))
                self.first_device = list(self.model.hf_device_map.values())[0] if hasattr(self.model, "hf_device_map") else self.device
            else:
                self.model = AutoModelForVision2Seq.from_pretrained(self.model_path, torch_dtype=self.dtype, trust_remote_code=True, local_files_only=os.path.exists(os.path.join(self.model_path, "config.json"))).to(self.device)
                self.first_device = self.device
            self.model.eval()
            self._max_input_length = self._infer_max_input_length()
            if self.rank == 0:
                print(f"[Model] Lingshu-7B loaded successfully")
                print(f"[Config] max_input_length = {self._max_input_length}")
        except Exception as e: raise RuntimeError(f"Lingshu-7B load failed: {e}")

    def generate(self, image_path: str, prompt_text: str, max_new_tokens: int = 256, temperature: float = 0.0) -> str:
        from qwen_vl_utils import process_vision_info
        image = Image.open(image_path).convert("RGB")
        prompt_text = self._safe_trim_prompt_text(prompt_text, reserve=768, keep_tail=True)
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(self.first_device if self.use_multi_gpu else self.device)
        gen_kwargs = self._build_gen_kwargs(max_new_tokens=max_new_tokens, temperature=temperature)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)
        trim_start = int(inputs["input_ids"].shape[1])
        gen_only_ids = generated_ids[0][trim_start:] if generated_ids.shape[1] > trim_start else generated_ids[0]
        output_text = self.processor.decode(gen_only_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
        if not output_text:
            output_text = self.processor.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
        return output_text

    def generate_batch(self, image_paths: List[str], prompt_texts: List[str], max_new_tokens: int = 256, temperature: float = 0.0) -> List[str]:
        from qwen_vl_utils import process_vision_info
        images = [Image.open(p).convert("RGB") for p in image_paths]
        cleaned_prompts = [self._safe_trim_prompt_text(str(p), reserve=768, keep_tail=True) for p in prompt_texts]
        batch_messages = [[{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}] for img, prompt in zip(images, cleaned_prompts)]
        all_texts, all_images, all_videos = [], [], []
        for messages in batch_messages:
            all_texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            image_inputs, video_inputs = process_vision_info(messages)
            if image_inputs: all_images.extend(image_inputs)
            if video_inputs: all_videos.extend(video_inputs)
        inputs = self.processor(text=all_texts, images=all_images if all_images else None, videos=all_videos if all_videos else None, padding=True, return_tensors="pt")
        inputs = inputs.to(self.first_device if self.use_multi_gpu else self.device)
        gen_kwargs = self._build_gen_kwargs(max_new_tokens=max_new_tokens, temperature=temperature)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)
        trim_start = int(inputs["input_ids"].shape[1])
        output_texts = []
        for i, out_ids in enumerate(generated_ids):
            gen_only = out_ids[trim_start:] if out_ids.shape[0] > trim_start else out_ids
            txt = self.processor.decode(gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
            if not txt:
                txt = self.processor.decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
            output_texts.append(txt)
        return output_texts

class HuatuoGPTVLM(BaseVLM):
    def __init__(self, model_path: str, device: torch.device, dtype: torch.dtype, rank: int, use_multi_gpu: bool = False):
        super().__init__(model_path, device, dtype, rank)
        self.use_multi_gpu = use_multi_gpu
        self.first_device = None

    def load_model(self):
        from transformers import AutoModelForVision2Seq, AutoProcessor
        if self.rank == 0: print(f"[Model] Loading Qwen-VL-based Model from {self.model_path}")
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True, local_files_only=os.path.exists(os.path.join(self.model_path, "config.json")))
            if hasattr(self.processor, "tokenizer"):
                self.processor.tokenizer.padding_side = "left"
                if self.processor.tokenizer.pad_token is None:
                    self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
                    self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id
            if self.use_multi_gpu:
                num_gpus = torch.cuda.device_count()
                max_memory = {i: "20GiB" for i in range(num_gpus)}
                self.model = AutoModelForVision2Seq.from_pretrained(self.model_path, torch_dtype=self.dtype, device_map="auto", max_memory=max_memory, trust_remote_code=True, low_cpu_mem_usage=True, local_files_only=os.path.exists(os.path.join(self.model_path, "config.json")))
                self.first_device = list(self.model.hf_device_map.values())[0] if hasattr(self.model, "hf_device_map") else self.device
            else:
                self.model = AutoModelForVision2Seq.from_pretrained(self.model_path, torch_dtype=self.dtype, trust_remote_code=True, local_files_only=os.path.exists(os.path.join(self.model_path, "config.json"))).to(self.device)
                self.first_device = self.device
            self.model.eval()
            if self.rank == 0:
                print(f"[Model] Model loaded successfully")
                try:
                    print(f"[Verify] Architecture Class: {type(self.model).__name__}")
                    print(f"[Verify] Real Model Name: {self.model.config._name_or_path}")
                    total_params = sum(p.numel() for p in self.model.parameters())
                    print(f"[Verify] Total Parameters: {total_params / 1e9:.2f} B")
                except: pass
        except Exception as e: raise RuntimeError(f"Model load failed: {e}")

    def generate(self, image_path: str, prompt_text: str, max_new_tokens: int = 256, temperature: float = 0.0) -> str:
        from qwen_vl_utils import process_vision_info
        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(self.first_device if self.use_multi_gpu else self.device)
        with torch.no_grad():
            gen_kwargs = {"max_new_tokens": max_new_tokens, "min_new_tokens": 2}
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["do_sample"] = True
            generated_ids = self.model.generate(**inputs, **gen_kwargs)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        return output_text.strip()

    def generate_batch(self, image_paths: List[str], prompt_texts: List[str], max_new_tokens: int = 256, temperature: float = 0.0) -> List[str]:
        from qwen_vl_utils import process_vision_info
        images = [Image.open(p).convert("RGB") for p in image_paths]
        cleaned_prompts = [str(p) for p in prompt_texts]
        batch_messages = [[{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}] for img, prompt in zip(images, cleaned_prompts)]
        all_texts, all_images = [], []
        for messages in batch_messages:
            all_texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            image_inputs, _ = process_vision_info(messages)
            if image_inputs: all_images.extend(image_inputs)
        inputs = self.processor(text=all_texts, images=all_images if all_images else None, padding=True, return_tensors="pt")
        inputs = inputs.to(self.first_device if self.use_multi_gpu else self.device)
        with torch.no_grad():
            gen_kwargs = {"max_new_tokens": max_new_tokens, "min_new_tokens": 2}
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["do_sample"] = True
            generated_ids = self.model.generate(**inputs, **gen_kwargs)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_texts = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return [text.strip() for text in output_texts]

class InternVLM(BaseVLM):
    def __init__(self, model_path: str, device: torch.device, dtype: torch.dtype, rank: int, use_multi_gpu: bool = False):
        super().__init__(model_path, device, dtype, rank)
        self.use_multi_gpu = use_multi_gpu
        self.transform = None
        self.tokenizer = None
    
    def build_transform(self, input_size):
        MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
    
    def load_model(self):
        from transformers import AutoTokenizer, AutoModel
        if self.rank == 0: print(f"[Model] Loading InternVL from {self.model_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, use_fast=False)
            if self.use_multi_gpu:
                num_gpus = torch.cuda.device_count()
                max_memory = {i: "20GiB" for i in range(num_gpus)}
                self.model = AutoModel.from_pretrained(self.model_path, torch_dtype=self.dtype, trust_remote_code=True, device_map="auto", max_memory=max_memory, low_cpu_mem_usage=True).eval()
                self.first_device = list(self.model.hf_device_map.values())[0] if hasattr(self.model, "hf_device_map") else self.device
            else:
                self.model = AutoModel.from_pretrained(self.model_path, torch_dtype=self.dtype, trust_remote_code=True, low_cpu_mem_usage=True).eval().to(self.device)
                self.first_device = self.device
            img_size = getattr(self.model.config, 'force_image_size', 448)
            self.transform = self.build_transform(input_size=img_size)
            if self.rank == 0: print(f"[Model] InternVL loaded successfully")
        except Exception as e: raise RuntimeError(f"InternVL load failed: {e}")

    def dynamic_preprocess_func(self, image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        target_aspect_ratio = min(target_ratios, key=lambda x: abs(x[0] / x[1] - aspect_ratio))
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size, ((i % (target_width // image_size)) + 1) * image_size, ((i // (target_width // image_size)) + 1) * image_size)
            processed_images.append(resized_img.crop(box))
        if use_thumbnail and len(processed_images) > 1:
            processed_images.append(image.resize((image_size, image_size)))
        return processed_images

    def load_image(self, image, max_num=6):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        images = self.dynamic_preprocess_func(image, image_size=448, use_thumbnail=True, max_num=max_num)
        pixel_values = [self.transform(img) for img in images]
        return torch.stack(pixel_values)

    def generate(self, image_path: str, prompt_text: str, max_new_tokens: int = 256, temperature: float = 0.0) -> str:
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.load_image(image, max_num=6).to(self.dtype).to(self.device)
        question = f"<image>\n{prompt_text}"
        generation_config = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0)
        if temperature > 0: generation_config["temperature"] = temperature
        with torch.no_grad():
            response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)
        return response

    def generate_batch(self, image_paths: List[str], prompt_texts: List[str], max_new_tokens: int = 256, temperature: float = 0.0) -> List[str]:
        results = []
        for img_path, prompt in zip(image_paths, prompt_texts):
            results.append(self.generate(img_path, prompt, max_new_tokens, temperature))
        return results

class Llama3VLM(BaseVLM):
    def __init__(self, model_path: str, device: torch.device, dtype: torch.dtype, rank: int, use_multi_gpu: bool = False):
        super().__init__(model_path, device, dtype, rank)
        self.use_multi_gpu = use_multi_gpu
        self.first_device = None
        self._max_input_length = None
        self.empty_cache = bool(int(os.environ.get("LLAMA_EMPTY_CACHE", "0")))

    def _infer_max_input_length(self) -> int:
        env_v = os.environ.get("LLAMA_MAX_INPUT_LENGTH", "").strip()
        if env_v.isdigit(): return int(env_v)
        candidates = []
        tok = getattr(self.processor, "tokenizer", None)
        if tok is not None:
            v = getattr(tok, "model_max_length", None)
            if isinstance(v, int) and v > 0 and v < 1_000_000: candidates.append(v)
        cfg = getattr(self.model, "config", None)
        if cfg is not None:
            for k in ["max_position_embeddings", "model_max_length"]:
                v = getattr(cfg, k, None)
                if isinstance(v, int) and v > 0: candidates.append(v)
            tc = getattr(cfg, "text_config", None)
            if tc is not None:
                v = getattr(tc, "max_position_embeddings", None)
                if isinstance(v, int) and v > 0: candidates.append(v)
        base = min([c for c in candidates if c < 1_000_000], default=8192)
        return max(2048, min(base, 16384))

    def _safe_trim_prompt_text(self, prompt_text: str, reserve: int = 2048, keep_tail: bool = True) -> str:
        tok = getattr(self.processor, "tokenizer", None)
        if tok is None: return prompt_text
        if self._max_input_length is None: self._max_input_length = self._infer_max_input_length()
        env_reserve = os.environ.get("LLAMA_PROMPT_RESERVE", "").strip()
        if env_reserve.isdigit(): reserve = int(env_reserve)
        max_ctx = int(self._max_input_length or 8192)
        budget = max(256, max_ctx - int(reserve))
        ids = tok(prompt_text, add_special_tokens=False).get("input_ids", [])
        if len(ids) <= budget: return prompt_text
        trimmed_ids = ids[-budget:] if keep_tail else ids[:budget]
        return tok.decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def _build_gen_kwargs(self, max_new_tokens: int, temperature: float):
        tok = getattr(self.processor, "tokenizer", None)
        eos_id = getattr(tok, "eos_token_id", None)
        pad_id = getattr(tok, "pad_token_id", None)
        if pad_id is None: pad_id = eos_id
        gen_kwargs = {"max_new_tokens": int(max_new_tokens), "min_new_tokens": 2, "do_sample": bool(temperature and temperature > 0)}
        if temperature and temperature > 0: gen_kwargs["temperature"] = float(temperature)
        if eos_id is not None: gen_kwargs["eos_token_id"] = eos_id
        if pad_id is not None: gen_kwargs["pad_token_id"] = pad_id
        return gen_kwargs

    def _is_prompt_leak(self, txt: str, prompt_tail: str | None = None) -> bool:
        if not txt: return True
        for m in ["Cutting Knowledge Date", "Today Date"]:
            if m in txt: return True
        if prompt_tail:
            pt = (prompt_tail or "").strip()
            if pt and pt in txt: return True
        return False

    def _decode_best(self, out_ids_1d, input_ids_1d, attn_mask_1d, prompt_tail):
        padded_len = int(input_ids_1d.shape[0])
        real_len = int(attn_mask_1d.sum().item()) if attn_mask_1d is not None else padded_len
        out_len = int(out_ids_1d.shape[0])
        candidates = []
        if out_len > padded_len: candidates.append(out_ids_1d[padded_len:])
        if out_len > real_len and real_len != padded_len: candidates.append(out_ids_1d[real_len:])
        candidates.append(out_ids_1d)
        for ids in candidates:
            txt = self.processor.decode(ids, skip_special_tokens=True)
            txt = (txt or "").strip()
            if txt and not self._is_prompt_leak(txt, prompt_tail=prompt_tail): return txt
        return (self.processor.decode(out_ids_1d, skip_special_tokens=True) or "").strip()

    def load_model(self):
        from transformers import (MllamaForConditionalGeneration, AutoImageProcessor, MllamaProcessor, PreTrainedTokenizerFast, LlamaTokenizerFast)
        if self.rank == 0: print(f"[Model] Loading Llama 3.2-Vision from {self.model_path}")
        try:
            tokenizer = None
            try: tokenizer = PreTrainedTokenizerFast.from_pretrained(self.model_path)
            except: pass
            if tokenizer is None:
                try: tokenizer = LlamaTokenizerFast.from_pretrained(self.model_path)
                except: pass
            if tokenizer is None:
                if self.rank == 0: print("  - [Fallback] Downloading tokenizer from HF Hub...")
                tokenizer = PreTrainedTokenizerFast.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")
            if tokenizer is None: raise RuntimeError("All tokenizer loading methods failed.")
            try: tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
            except: pass
            try: image_processor = AutoImageProcessor.from_pretrained(self.model_path)
            except: image_processor = AutoImageProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")
            self.processor = MllamaProcessor(image_processor=image_processor, tokenizer=tokenizer)
            if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
                try: self.processor.tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
                except: pass
            if self.use_multi_gpu:
                num_gpus = torch.cuda.device_count()
                max_memory = {i: "20GiB" for i in range(num_gpus)}
                self.model = MllamaForConditionalGeneration.from_pretrained(self.model_path, torch_dtype=self.dtype, device_map="auto", max_memory=max_memory, low_cpu_mem_usage=True)
                self.first_device = list(self.model.hf_device_map.values())[0] if hasattr(self.model, "hf_device_map") else self.device
            else:
                self.model = MllamaForConditionalGeneration.from_pretrained(self.model_path, torch_dtype=self.dtype, device_map=None, low_cpu_mem_usage=True).to(self.device)
                self.first_device = self.device
            self.model.eval()
            try:
                tok = self.processor.tokenizer
                if getattr(self.model, "generation_config", None) is not None:
                    if getattr(self.model.generation_config, "pad_token_id", None) is None: self.model.generation_config.pad_token_id = tok.pad_token_id
                    if getattr(self.model.generation_config, "eos_token_id", None) is None: self.model.generation_config.eos_token_id = tok.eos_token_id
            except: pass
            if self.rank == 0: print("[Model] Llama 3.2-Vision loaded successfully")
        except Exception as e:
            import traceback; traceback.print_exc()
            raise RuntimeError(f"Llama 3.2 load failed: {e}")

    def get_prompt_text(self, prompt_text: str) -> str:
        prompt_text = self._safe_trim_prompt_text(prompt_text, reserve=2048, keep_tail=True)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
        try: return self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        except: return self.processor.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    def generate(self, image_path: str, prompt_text: str, max_new_tokens: int = 256, temperature: float = 0.0) -> str:
        image = Image.open(image_path).convert("RGB")
        input_text = self.get_prompt_text(prompt_text)
        inputs = self.processor(text=[input_text], images=[[image]], add_special_tokens=False, padding=True, return_tensors="pt")
        target_device = self.first_device if self.use_multi_gpu else self.model.device
        inputs = inputs.to(target_device)
        if "pixel_values" in inputs and isinstance(inputs["pixel_values"], torch.Tensor):
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.dtype)
        gen_kwargs = self._build_gen_kwargs(max_new_tokens=max_new_tokens, temperature=temperature)
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        attn = inputs.get("attention_mask", None)
        txt = self._decode_best(outputs[0], inputs["input_ids"][0], attn[0] if attn is not None else None, (prompt_text or "")[-128:])
        if self.empty_cache: torch.cuda.empty_cache()
        return (txt or "").strip()

    def generate_batch(self, image_paths: List[str], prompt_texts: List[str], max_new_tokens: int = 256, temperature: float = 0.0) -> List[str]:
        images = [[Image.open(p).convert("RGB")] for p in image_paths]
        input_texts = [self.get_prompt_text(p) for p in prompt_texts]
        inputs = self.processor(text=input_texts, images=images, add_special_tokens=False, padding=True, return_tensors="pt")
        target_device = self.first_device if self.use_multi_gpu else self.model.device
        inputs = inputs.to(target_device)
        if "pixel_values" in inputs and isinstance(inputs["pixel_values"], torch.Tensor):
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.dtype)
        gen_kwargs = self._build_gen_kwargs(max_new_tokens=max_new_tokens, temperature=temperature)
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        attn = inputs.get("attention_mask", None)
        results = []
        for i in range(outputs.shape[0]):
            txt = self._decode_best(outputs[i], inputs["input_ids"][i], attn[i] if attn is not None else None, (prompt_texts[i] or "")[-128:])
            results.append((txt or "").strip())
        if self.empty_cache: torch.cuda.empty_cache()
        return results

def create_vlm(model_type: str, model_path: str, device: torch.device,
               dtype: torch.dtype, rank: int, use_multi_gpu: bool = False) -> BaseVLM:
    mtype = model_type.lower()
    if mtype == "medgemma": return MedGemmaVLM(model_path, device, dtype, rank, use_multi_gpu)
    elif mtype == "lingshu7b": return Lingshu7BVLM(model_path, device, dtype, rank, use_multi_gpu)
    elif mtype in ["huatuogpt", "qwen3"]: return HuatuoGPTVLM(model_path, device, dtype, rank, use_multi_gpu)
    elif mtype in ["internvl", "internvl2", "internvl2.5", "internvl3.5"]: return InternVLM(model_path, device, dtype, rank, use_multi_gpu)
    elif mtype == "llama3": return Llama3VLM(model_path, device, dtype, rank, use_multi_gpu)
    else: raise ValueError(f"Unknown model type: {model_type}")


def create_api_model(model_type: str, model_name: str, api_key: str) -> BaseAPIModel:
    mtype = model_type.lower()
    if mtype == "gpt4":
        return OpenAIVisionAPI(model_name, api_key)
    if mtype == "gemini":
        return GeminiAPI(model_name, api_key)
    raise ValueError(f"Unknown API model type: {model_type}")

def run_detection(args, vlm: BaseVLM, pairs: List, rank: int, world_size: int):
    multimodal = len(pairs[0]) == 3
    source_value = resolve_source_value(args.source_preset, args.source_text)
    
    pedi_cxr_cases = None
    if args.modality == "pedi_cxr" and multimodal:
        if os.path.isfile(args.json_dir): json_file = Path(args.json_dir)
        elif os.path.isdir(args.json_dir):
            json_files = sorted(Path(args.json_dir).glob("*.json"))
            if json_files: json_file = json_files[0]
            else: print(f"[ERROR] No JSON files found in {args.json_dir}"); return
        else: print(f"[ERROR] Invalid json_dir: {args.json_dir}"); return
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                pedi_cxr_cases = json.load(f)
            if rank == 0: print(f"[Load] Loaded {len(pedi_cxr_cases)} {args.modality} cases from {json_file}")
        except Exception as e:
            print(f"[ERROR] Failed to load pedi_cxr JSON: {e}"); return
    
    part_json = Path(args.out_json).with_suffix(f".rank{rank}.json")
    results = []
    processed_count = 0
    skipped_count = 0
    
    try:
        use_batching = args.batch_size > 1 and hasattr(vlm, 'generate_batch')
        
        if use_batching:
            num_batches = (len(pairs) + args.batch_size - 1) // args.batch_size
            pbar = tqdm(range(num_batches), desc=f"rank{rank}", disable=(rank != 0 and not args.verbose))
            
            for batch_idx in pbar:
                start_idx = batch_idx * args.batch_size
                end_idx = min(start_idx + args.batch_size, len(pairs))
                batch_pairs = pairs[start_idx:end_idx]
                
                batch_image_ids, batch_img_paths, batch_prompts, batch_metadata_objs = [], [], [], []
                
                for item in batch_pairs:
                    rec_for_prompt = {}
                    if multimodal:
                        image_id, case_ref, img_path = item
                        if args.modality == "pedi_cxr":
                            try:
                                case_idx = int(case_ref.split("_case_")[-1])
                                rec = pedi_cxr_cases[case_idx]
                            except: skipped_count += 1; continue
                        else:
                            try:
                                with open(case_ref, "r", encoding="utf-8") as f:
                                    rec = json.load(f)
                            except: skipped_count += 1; continue
                        
                        if args.modality == "cxr": rec, _ = clean_cxr_json(rec)
                        elif args.modality == "pedi_cxr": rec, _ = clean_pedi_cxr_json(rec)
                        
                        rec_for_prompt = copy.deepcopy(rec)
                        
                        if source_value is not None:
                            if args.modality in ["cxr", "pedi_cxr"]:
                                rec_for_prompt["Source"] = source_value
                            elif args.source_into == "metadata":
                                if not isinstance(rec_for_prompt.get("metadata"), dict):
                                    rec_for_prompt["metadata"] = {}
                                rec_for_prompt["metadata"]["Source"] = source_value
                            else:
                                rec_for_prompt["Source"] = source_value
                        
                        json_content_str = json.dumps(rec_for_prompt, ensure_ascii=False, indent=2)
                        prompt_text = prompts.get_detection_prompt(json_content_str)
                    else:
                        image_id, img_path = item
                        rec_for_prompt = None
                        prompt_text = prompts.get_detection_prompt()
                    
                    if not os.path.exists(img_path): skipped_count += 1; continue
                    
                    batch_image_ids.append(image_id)
                    batch_img_paths.append(img_path)
                    batch_prompts.append(prompt_text)
                    batch_metadata_objs.append(rec_for_prompt)
                
                if not batch_image_ids: continue
                
                try:
                    gen_texts = vlm.generate_batch(batch_img_paths, batch_prompts, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
                except Exception as e:
                    print(f"[rank{rank}] Batch Error: {e}"); skipped_count += len(batch_image_ids); continue
                
                for i, (image_id, gen_text, meta_obj) in enumerate(zip(batch_image_ids, gen_texts, batch_metadata_objs)):
                    final_answer, _ = extract_final_answer(gen_text)
                    visual_verdict, _ = extract_visual_verdict(gen_text)
                    if rank == 0:
                        print(
                            f"\n[Example] {image_id} -> "
                            f"FINAL={final_answer}, VISUAL={visual_verdict}"
                        )
                        print(f"PROMPT: {batch_prompts[i]}...")
                        if args.verbose: print(f"RAW: {gen_text}...")
                    results.append({
                        "image_id": image_id,
                        "pred": final_answer,
                        "final_answer": final_answer,
                        "visual_verdict": visual_verdict,
                        "raw_output": gen_text,
                        "metadata": meta_obj,
                    })
                    processed_count += 1

        else:
            pbar = tqdm(pairs, desc=f"rank{rank}", disable=(rank != 0 and not args.verbose))
            for item in pbar:
                rec_for_prompt = None
                if multimodal:
                    image_id, case_ref, img_path = item
                    if args.modality == "pedi_cxr":
                        try:
                            case_idx = int(case_ref.split("_case_")[-1])
                            rec = pedi_cxr_cases[case_idx]
                        except: skipped_count += 1; continue
                    else:
                        try:
                            with open(case_ref, "r", encoding="utf-8") as f:
                                rec = json.load(f)
                        except: skipped_count += 1; continue
                    
                    if args.modality == "cxr": rec, _ = clean_cxr_json(rec)
                    elif args.modality == "pedi_cxr": rec, _ = clean_pedi_cxr_json(rec)
                    
                    rec_for_prompt = copy.deepcopy(rec)
                    
                    if source_value is not None:
                        if args.modality in ["cxr", "pedi_cxr"]: rec_for_prompt["Source"] = source_value
                        elif args.source_into == "metadata":
                            if not isinstance(rec_for_prompt.get("metadata"), dict): rec_for_prompt["metadata"] = {}
                            rec_for_prompt["metadata"]["Source"] = source_value
                        else: rec_for_prompt["Source"] = source_value
                    
                    json_content_str = json.dumps(rec_for_prompt, ensure_ascii=False, indent=2)
                    prompt_text = prompts.get_detection_prompt(json_content_str)
                else:
                    image_id, img_path = item
                    prompt_text = prompts.get_detection_prompt()

                if not os.path.exists(img_path): skipped_count += 1; continue
                
                try:
                    gen_text = vlm.generate(img_path, prompt_text, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
                except Exception as e:
                    print(f"[rank{rank}] Error: {e}"); skipped_count += 1; continue
                
                final_answer, _ = extract_final_answer(gen_text)
                visual_verdict, _ = extract_visual_verdict(gen_text)
                if rank == 0 and args.verbose:
                    print(
                        f"\n[Example] {image_id} -> "
                        f"FINAL={final_answer}, VISUAL={visual_verdict}"
                    )
                    print(f"PROMPT: {prompt_text}")
                    print(f"RAW: {gen_text}")
                
                results.append({
                    "image_id": image_id,
                    "pred": final_answer,
                    "final_answer": final_answer,
                    "visual_verdict": visual_verdict,
                    "raw_output": gen_text,
                    "metadata": rec_for_prompt,
                })
                processed_count += 1
        
        with open(part_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        if rank == 0: print(f"\n[rank{rank}] Processed {processed_count}, skipped {skipped_count}")
        
        ddp_barrier()
        
        if rank == 0:
            final_data = []
            for r in range(world_size):
                pjson = Path(args.out_json).with_suffix(f".rank{r}.json")
                if pjson.exists():
                    try:
                        with open(pjson, "r", encoding="utf-8") as f:
                            final_data.extend(json.load(f))
                        pjson.unlink()
                    except Exception as e:
                        print(f"[WARN] Failed to merge {pjson}: {e}")
            
            with open(args.out_json, "w", encoding="utf-8") as f:
                json.dump(final_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n[Success] Wrote {len(final_data)} records to {args.out_json}")
            final_real = sum(1 for r in final_data if r["final_answer"] == "REAL")
            final_fake = sum(1 for r in final_data if r["final_answer"] == "FAKE")
            final_error = sum(1 for r in final_data if r["final_answer"] == "ERROR")
            visual_real = sum(1 for r in final_data if r["visual_verdict"] == "REAL")
            visual_fake = sum(1 for r in final_data if r["visual_verdict"] == "FAKE")
            visual_error = sum(1 for r in final_data if r["visual_verdict"] == "ERROR")
            print(
                "Stats: "
                f"FINAL(REAL={final_real}, FAKE={final_fake}, ERROR={final_error}); "
                f"VISUAL(REAL={visual_real}, FAKE={visual_fake}, ERROR={visual_error}); "
                f"Total={len(final_data)}"
            )

    finally:
        pass

def main():
    ap = argparse.ArgumentParser(
        description="Medical image manipulation detection with local or API VLMs"
    )
    
    ap.add_argument(
        "--model_type",
        choices=[
            "medgemma",
            "lingshu7b",
            "huatuogpt",
            "qwen3",
            "internvl",
            "llama3",
            "gpt4",
            "gemini",
        ],
        default="medgemma",
    )
    ap.add_argument("--model_path", default="google/medgemma-4b-it")
    ap.add_argument(
        "--model_name",
        default=None,
        help="API model name; required for gpt4 and gemini.",
    )
    ap.add_argument(
        "--api_key",
        default=None,
        help="API key. If omitted, the provider-specific environment variable is used.",
    )
    ap.add_argument("--use_multi_gpu", action="store_true")
    
    ap.add_argument("--modality", choices=["isic", "cxr", "pedi_cxr"], default="isic")
    ap.add_argument("--mode", choices=["image-only", "multimodal"], default="multimodal")
    
    ap.add_argument("--images_dir", default="")
    ap.add_argument("--json_dir", default="")
    ap.add_argument("--out_json", default="")

    ap.add_argument("--frozen_csv", default=None,
                    help="Path to a CSV with an 'image_id' column. "
                         "When set, only cases whose image_id appears in the CSV are processed. "
                         "--limit is ignored when this is set.")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=1)
    
    ap.add_argument("--source_preset", choices=["nano", "hospital"], default=None)
    ap.add_argument("--source_text", default=None)
    ap.add_argument("--source_into", choices=["top", "metadata"], default="metadata")
    
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    
    args = ap.parse_args()
    
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)

    api_model_types = {"gpt4", "gemini"}
    is_api_model = args.model_type in api_model_types

    if is_api_model:
        if args.use_multi_gpu:
            ap.error("--use_multi_gpu is only valid for local models")
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            ap.error("API models must be launched as a single process, without torchrun")
        if not args.model_name:
            ap.error("--model_name is required when using an API model")
        environment_variables = {
            "gpt4": "OPENAI_API_KEY",
            "gemini": "GOOGLE_API_KEY",
        }
        environment_variable = environment_variables[args.model_type]
        args.api_key = args.api_key or os.environ.get(environment_variable)
        if not args.api_key:
            ap.error(
                f"API key required: pass --api_key or set {environment_variable}"
            )
        world_size, local_rank, rank = 1, 0, 0
        device = torch.device("cpu")
        dtype = torch.float32
    else:
        _ = ddp_init()
        world_size, local_rank, rank = ddp_env()
        device = torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )
        if args.bf16:
            dtype = torch.bfloat16
        elif args.fp16:
            dtype = torch.float16
        elif args.fp32:
            dtype = torch.float32
        else:
            dtype = (
                torch.bfloat16
                if torch.cuda.is_available()
                and torch.cuda.get_device_capability(device)[0] >= 8
                else torch.float32
            )
    
    if rank == 0:
        print("=== Medical VLM Detector ===")
        if is_api_model:
            print(f"Model: {args.model_type}, Name: {args.model_name}")
        else:
            print(f"Model: {args.model_type}, Path: {args.model_path}")
        print(f"Modality: {args.modality}, Mode: {args.mode}")
        print(f"Output: {args.out_json}")
        if args.frozen_csv:
            print(f"FrozenCSV: {args.frozen_csv} (--limit ignored)")

    frozen_ids = load_frozen_ids(args.frozen_csv) if args.frozen_csv else set()
    effective_limit = 0 if frozen_ids else args.limit

    if args.mode == "multimodal":
        pairs = discover_pairs(args.json_dir, args.images_dir, effective_limit,
                               args.verbose, args.modality, frozen_ids)
    else:
        pairs = discover_images(args.images_dir, effective_limit, args.json_dir,
                                args.verbose, args.modality, frozen_ids)
    
    if not pairs:
        print(f"[ERROR] No valid pairs found. Exiting.")
        return
    
    pairs_shard = pairs[rank::world_size]
    if is_api_model:
        vlm = create_api_model(args.model_type, args.model_name, args.api_key)
    else:
        vlm = create_vlm(
            args.model_type,
            args.model_path,
            device,
            dtype,
            rank,
            args.use_multi_gpu,
        )
    vlm.load_model()
    
    try:
        run_detection(args, vlm, pairs_shard, rank, world_size)
    finally:
        if not is_api_model:
            ddp_destroy()

if __name__ == "__main__":
    main()
