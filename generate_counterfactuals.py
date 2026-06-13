"""Generate ISIC or CXR counterfactual images with Gemini."""

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate disease-conditioned medical image counterfactuals"
    )
    parser.add_argument("--modality", choices=["isic", "cxr"], required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json-dir", help="ISIC metadata JSON directory")
    parser.add_argument("--csv-path", help="CXR metadata CSV path")
    parser.add_argument("--num-images", type=int, default=50)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument(
        "--selection-strategy",
        choices=["mirror", "uniform", "balanced", "original"],
        default="mirror",
    )
    parser.add_argument(
        "--target-label",
        help="Optional fixed ISIC target label, such as NV, MEL, or BCC",
    )
    parser.add_argument("--image-id", help="Optional ISIC image ID to process")
    parser.add_argument(
        "--top-n-composite",
        type=int,
        default=5,
        help="Number of common composite CXR labels included as targets",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.modality == "isic":
        if not args.json_dir:
            parser.error("--json-dir is required for ISIC")
        if args.csv_path:
            parser.error("--csv-path is only valid for CXR")

        from isic_disease_editor import ISICDiseaseEditor

        editor = ISICDiseaseEditor(
            image_dir=args.image_dir,
            json_dir=args.json_dir,
            output_base=args.output_dir,
            max_workers=args.max_workers if args.max_workers is not None else 20,
            max_rounds=args.max_rounds if args.max_rounds is not None else 5,
            num_images=args.num_images,
            selection_strategy=args.selection_strategy,
            forced_target_label=args.target_label,
            target_image_id=args.image_id,
        )
    else:
        if not args.csv_path:
            parser.error("--csv-path is required for CXR")
        if args.json_dir or args.target_label or args.image_id:
            parser.error(
                "--json-dir, --target-label, and --image-id are only valid for ISIC"
            )

        from cxr_disease_editor import CXRDiseaseEditor

        editor = CXRDiseaseEditor(
            csv_path=args.csv_path,
            image_dir=args.image_dir,
            output_base=args.output_dir,
            max_workers=args.max_workers if args.max_workers is not None else 5,
            max_rounds=args.max_rounds if args.max_rounds is not None else 3,
            num_images=args.num_images,
            selection_strategy=args.selection_strategy,
            top_n_composite=args.top_n_composite,
        )

    editor.run()


if __name__ == "__main__":
    main()
