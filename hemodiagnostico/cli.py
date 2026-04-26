import argparse

from hemodiagnostico.config import load_project_config
from hemodiagnostico.data_generation import generate_dataset
from hemodiagnostico.model_training import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hemodiagnostico project CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_generate = subparsers.add_parser("generate", help="Generate synthetic CBC dataset")
    parser_generate.add_argument("--config", default="configs/project_config.json", help="Config path")
    parser_generate.add_argument("--output", default=None, help="Optional output CSV override")

    parser_train = subparsers.add_parser("train", help="Train and evaluate models")
    parser_train.add_argument("--config", default="configs/project_config.json", help="Config path")
    parser_train.add_argument("--data", default=None, help="Optional dataset path override")
    parser_train.add_argument("--output-dir", default=None, help="Optional output directory override")
    parser_train.add_argument("--tune", action="store_true", help="Enable tuning")
    parser_train.add_argument("--cv-folds", type=int, default=None, help="CV folds override")

    parser_full = subparsers.add_parser("full", help="Generate data then train models")
    parser_full.add_argument("--config", default="configs/project_config.json", help="Config path")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "generate":
        cfg = load_project_config(args.config)
        if args.output:
            cfg.data_generation.output_csv = args.output
        data = generate_dataset(cfg.data_generation)
        print(f"Dataset generated: {cfg.data_generation.output_csv}")
        print(f"Rows: {len(data):,} | Columns: {len(data.columns)}")
        return

    if args.command == "train":
        cfg = load_project_config(args.config)
        if args.data:
            cfg.training.data_path = args.data
        if args.output_dir:
            cfg.training.output_dir = args.output_dir
        if args.tune:
            cfg.training.tune = True
        if args.cv_folds is not None:
            cfg.training.cv_folds = args.cv_folds

        run_training(cfg.training)
        return

    if args.command == "full":
        cfg = load_project_config(args.config)
        generate_dataset(cfg.data_generation)
        run_training(cfg.training)


if __name__ == "__main__":
    main()
