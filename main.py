# main.py

import argparse
from end_to_end.multimodal import run as run_multimodal
from end_to_end.text_only import run as run_text_only
from end_to_end.visual_only import run as run_visual_only

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        choices=[
            "multimodal",
            "text_only",
            "visual_only",
        ],
    )

    parser.add_argument(
        "--prompt_type",
        type=str,
        required=False,
        choices=[
            "structured",
            "contrastive",
        ],
    )

    return parser.parse_args()

def main():
    args = parse_args()

    if args.scenario in {"text_only", "multimodal"} and args.prompt_type is None:
        raise ValueError("prompt_type is required for text_only and multimodal")

    if args.scenario == "visual_only":
        run_visual_only()
    elif args.scenario == "text_only":
        run_text_only(prompt_type=args.prompt_type)
    elif args.scenario == "multimodal":
        run_multimodal(prompt_type=args.prompt_type)

if __name__ == "__main__":
    main()