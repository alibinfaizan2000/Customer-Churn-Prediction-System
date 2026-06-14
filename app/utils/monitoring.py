"""
monitoring.py — Lightweight prediction logging and drift detection.

WHY MONITORING IN ML:
Models degrade silently in production when:
  1. Data distribution shifts (input drift)
  2. Label distribution shifts (concept drift)
  3. Business context changes

Without monitoring, you have no idea your model is degrading until
a business metric tanks. This module implements lightweight tracking.

INTERVIEW TIP: This is a junior-friendly implementation of what tools like
Evidently AI, Grafana, or Prometheus do in production.
"""

import json
import logging
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


class PredictionLogger:
    """
    Logs every prediction to a JSONL file for audit and monitoring.

    WHY JSONL (JSON Lines) format:
    - Each line is a valid JSON object → easy to stream/parse
    - Append-only → no read-modify-write locking issues
    - Human readable → easy to debug
    - Compatible with tools like Pandas, Spark, BigQuery
    """

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        customer_input: dict,
        prediction: dict,
        model_version: str = "v1"
    ) -> None:
        """
        Log a single prediction with timestamp.

        In production this would go to a database or message queue (Kafka),
        but JSONL is perfectly adequate for a portfolio project.
        """
        record = {
            "timestamp":     datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            "model_version": model_version,
            "input":         customer_input,
            "prediction": {
                "churn_probability": prediction.get("churn_probability"),
                "risk_level":        prediction.get("risk_level"),
            }
        }
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"Failed to write prediction log: {e}")


class DriftMonitor:
    """
    Lightweight drift detection based on mean/std shift.

    HOW IT WORKS:
    We maintain a rolling window of recent churn probabilities.
    If the mean shifts significantly from the baseline (training distribution),
    we flag it as potential drift.

    PRODUCTION EQUIVALENT: Evidently AI, WhyLabs, Arize AI
    """

    def __init__(
        self,
        baseline_mean: float = 0.27,
        baseline_std: float = 0.15,
        window_size: int = 100,
        drift_threshold: float = 0.1
    ):
        self.baseline_mean = baseline_mean
        self.baseline_std  = baseline_std
        self.window_size   = window_size
        self.threshold     = drift_threshold
        self.recent_probs  = deque(maxlen=window_size)

    def update(self, churn_probability: float) -> None:
        """Add a new prediction to the monitoring window."""
        self.recent_probs.append(churn_probability)

    def check_drift(self) -> dict:
        """
        Compare recent predictions to baseline distribution.

        WHY MEAN SHIFT:
        If the model was calibrated on 27% churn rate but is now seeing
        50% churn predictions, something has changed — either the customer
        base changed, or the model has drifted.

        Returns a status report.
        """
        if len(self.recent_probs) < 10:
            return {"status": "insufficient_data", "n": len(self.recent_probs)}

        recent = np.array(self.recent_probs)
        current_mean = float(np.mean(recent))
        current_std  = float(np.std(recent))
        mean_shift   = abs(current_mean - self.baseline_mean)

        status = "ok" if mean_shift < self.threshold else "drift_detected"

        report = {
            "status":        status,
            "n_predictions": len(self.recent_probs),
            "baseline_mean": self.baseline_mean,
            "current_mean":  round(current_mean, 4),
            "mean_shift":    round(mean_shift, 4),
            "threshold":     self.threshold,
            "drift_flag":    mean_shift >= self.threshold,
        }

        if status == "drift_detected":
            logger.warning(
                f"DRIFT DETECTED: mean churn prob shifted from "
                f"{self.baseline_mean:.3f} to {current_mean:.3f} "
                f"(shift={mean_shift:.3f}, threshold={self.threshold})"
            )

        return report

    def get_stats(self) -> dict:
        """Return summary statistics of recent predictions."""
        if not self.recent_probs:
            return {}
        recent = np.array(self.recent_probs)
        return {
            "count":    len(recent),
            "mean":     round(float(np.mean(recent)), 4),
            "std":      round(float(np.std(recent)), 4),
            "min":      round(float(np.min(recent)), 4),
            "max":      round(float(np.max(recent)), 4),
            "p50":      round(float(np.percentile(recent, 50)), 4),
            "p90":      round(float(np.percentile(recent, 90)), 4),
        }


def load_prediction_history(log_path: Path, last_n: int = 200) -> list[dict]:
    """Load the last N predictions from the log file."""
    if not log_path.exists():
        return []
    records = []
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
        for line in lines[-last_n:]:
            records.append(json.loads(line.strip()))
    except Exception as e:
        logger.error(f"Failed to load prediction history: {e}")
    return records
