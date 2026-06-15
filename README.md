# Beyond Visual Forensics: Auditing Multimodal Robustness for Synthetic Medical Image Detection

This repository contains the evaluation code and prompt used for medical image manipulation detection. The evaluator supports both image-only and multimodal settings.

- `FINAL ANSWER`: prediction based on the image and metadata in multimodal mode.
- `VISUAL VERDICT`: prediction based only on the image.

## Setup

Install PyTorch, Transformers, Pillow, torchvision, and the package required by the selected model. For OpenAI models, also install the OpenAI Python package:

```bash
pip install torch torchvision transformers pillow tqdm openai
```

Set the OpenAI API key:

```bash
export OPENAI_API_KEY="your_openai_api_key"
```

## Run Evaluation

Example using GPT-5 in the multimodal NIH ChestX-ray14 setting:

```bash
python run_evaluation.py \
  --model_type gpt4 \
  --model_name gpt-5 \
  --modality cxr \
  --mode multimodal \
  --images_dir /image_path \
  --json_dir /metadata_path \
  --frozen_csv nih_cxr_retained_image_ids.csv \
  --out_json /output_path/cxr_gpt5_multimodal_results.json \
  --max_new_tokens 4096 \
  --batch_size 1 \
  --verbose
```

For image-only evaluation, change the mode:

```bash
python run_evaluation.py \
  --model_type gpt4 \
  --model_name gpt-5 \
  --modality cxr \
  --mode image-only \
  --images_dir /image_path \
  --json_dir /metadata_path \
  --frozen_csv nih_cxr_retained_image_ids.csv \
  --out_json /output_path/cxr_gpt5_image_only_results.json \
  --max_new_tokens 4096
```

In image-only mode, `--json_dir` is used only to match the retained NIH ChestX-ray14 image IDs; metadata is not included in the model prompt.

The prompt is loaded automatically from `prompts.py`. Predictions and raw model responses are written to the JSON file specified by `--out_json`.

## Retained Evaluation Sets

The released evaluation subsets for ISIC 2019 and NIH ChestX-ray14 are listed in:

- ISIC 2019: `isic_retained_image_ids.csv`
- NIH ChestX-ray14: `nih_cxr_retained_image_ids.csv`

Each file contains an `image_id` column. Pass the corresponding file through `--frozen_csv` to reproduce the retained evaluation subset. When `--frozen_csv` is provided, `--limit` is ignored.

## Local Models

Local models use `--model_path` instead of `--model_name`. For example:

```bash
python run_evaluation.py \
  --model_type medgemma \
  --model_path /model_path \
  --modality cxr \
  --mode multimodal \
  --images_dir /image_path \
  --json_dir /metadata_path \
  --out_json /output_path/results.json \
  --bf16
```

Supported modalities are ISIC 2019 (`isic`), NIH ChestX-ray14 (`cxr`), and pediatric chest X-ray (`pedi_cxr`).

## Generate Counterfactual Images

The counterfactual image-generation pipeline was adapted from the methodology and implementation of [Med-Banana-50K](https://github.com/richardChenzhihui/med-banana-50k), with modifications for experiments on [ISIC 2019](https://challenge.isic-archive.com/landing/2019/) and [NIH ChestX-ray14](https://www.kaggle.com/datasets/nih-chest-xrays/data).

Both ISIC 2019 and NIH ChestX-ray14 counterfactual generation use `gemini-2.5-flash-image` for image editing. Set the Gemini API key before running:

```bash
pip install google-genai pandas numpy
export GEMINI_API_KEY="your_gemini_api_key"
```

ISIC 2019 example:

```bash
python generate_counterfactuals.py \
  --modality isic \
  --image-dir /image_path \
  --json-dir /metadata_path \
  --output-dir /output_path \
  --num-images 50
```

NIH ChestX-ray14 example:

```bash
python generate_counterfactuals.py \
  --modality cxr \
  --image-dir /image_path \
  --csv-path /metadata.csv \
  --output-dir /output_path \
  --num-images 50
```

## Data Release

Generated images are not redistributed to avoid ambiguity around derivative medical-image redistribution and source-dataset licensing. We release the retained image IDs, prompts, target labels, metadata variants, generation code, evaluation code, and results to support reproducibility.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. Note that the source datasets (NIH ChestX-ray14, ISIC2019) are subject to their own licensing terms.
