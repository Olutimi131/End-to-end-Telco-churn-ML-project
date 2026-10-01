"""CLI pipeline entry point for training, evaluation, inference, and drift monitoring."""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.telco_churn.config import load_config
from src.telco_churn.data.dataset import load_raw_data, split_data
from src.telco_churn.models.train import train_and_select_best_model
from src.telco_churn.models.predict import TelcoChurnPredictor
from src.telco_churn.monitoring.drift import evaluate_dataset_drift
from src.telco_churn.utils.io import load_json
from src.telco_churn.utils.logger import setup_logger

logger = setup_logger("telco_pipeline")


def main() -> None:
    parser = argparse.ArgumentParser(description="Telco Customer Churn Pipeline CLI")
    parser.add_argument("--all", action="store_true", help="Run full pipeline: ingest, train, evaluate, persist")
    parser.add_argument("--ingest", action="store_true", help="Download raw dataset and split into train/test")
    parser.add_argument("--train", action="store_true", help="Train and select best model")
    parser.add_argument("--predict", type=str, help="Path to input CSV file for batch inference")
    parser.add_argument("--output", type=str, default="data/predictions.csv", help="Path to save predictions")
    parser.add_argument("--drift-check", type=str, help="Path to new dataset CSV to check against baseline drift")
    parser.add_argument("--config", type=str, default=None, help="Path to custom config YAML")

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.all or args.ingest:
        logger.info("=== STEP 1: DATA INGESTION & SPLITTING ===")
        raw_df = load_raw_data(cfg)
        train_df, test_df = split_data(raw_df, cfg, save=True)
        logger.info(f"Ingestion complete. Total: {len(raw_df)}, Train: {len(train_df)}, Test: {len(test_df)}")

    if args.all or args.train:
        logger.info("=== STEP 2: MODEL TRAINING & ARTIFACT PERSISTENCE ===")
        results = train_and_select_best_model(cfg)
        eval_metrics = results["evaluation"]["metrics"]
        logger.info("Training complete!")
        logger.info(f"Selected Best Model: {results['best_model']}")
        logger.info(f"Test ROC-AUC: {eval_metrics['roc_auc']:.4f}")
        logger.info(f"Test PR-AUC:  {eval_metrics['pr_auc']:.4f}")
        logger.info(f"Test F1:      {eval_metrics['f1']:.4f}")
        logger.info(f"Optimal Decision Threshold: {results['evaluation']['optimal_threshold_metrics']['optimal_threshold']}")

    if args.predict:
        logger.info(f"=== BATCH INFERENCE ON {args.predict} ===")
        predictor = TelcoChurnPredictor(config=cfg)
        input_df = pd.read_csv(args.predict)
        scored_df = predictor.predict_batch(input_df)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        scored_df.to_csv(out_path, index=False)
        logger.info(f"Saved {len(scored_df)} scored records to {out_path}")

    if args.drift_check:
        logger.info(f"=== DRIFT EVALUATION ON {args.drift_check} ===")
        if not cfg.paths.drift_baseline_file.exists():
            logger.error("Drift baseline file does not exist. Please train the model first.")
            sys.exit(1)
        baseline = load_json(cfg.paths.drift_baseline_file)
        new_df = pd.read_csv(args.drift_check)
        drift_report = evaluate_dataset_drift(new_df, baseline, cfg)
        logger.info(f"Drift Status: {drift_report['overall_status']}")
        logger.info(f"Flagged features ({drift_report['flagged_features_count']}): {drift_report['flagged_features']}")

    if not any([args.all, args.ingest, args.train, args.predict, args.drift_check]):
        parser.print_help()


if __name__ == "__main__":
    main()
