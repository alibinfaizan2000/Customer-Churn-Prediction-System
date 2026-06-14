"""
run_all.py — One-command launcher for the full Churn Prediction system.

Starts both the FastAPI backend and Streamlit UI in separate processes.
Press Ctrl+C to stop both.

Usage:
    python run_all.py               # train if needed, then start services
    python run_all.py --skip-train  # skip training (use existing model)
    python run_all.py --retrain     # force retrain even if model exists
"""

import sys
import os
import time
import signal
import argparse
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import MODEL_PATH, API_PORT, STREAMLIT_PORT

ENV = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
processes: list[subprocess.Popen] = []


def train_model():
    """Run the training pipeline."""
    logger.info("="*55)
    logger.info("  STEP 1: Training model...")
    logger.info("="*55)
    result = subprocess.run(
        [sys.executable, "app/training/train.py"],
        cwd=PROJECT_ROOT,
        env=ENV
    )
    if result.returncode != 0:
        logger.error("Training failed. Exiting.")
        sys.exit(1)
    logger.info("✅ Training complete.\n")


def run_evaluation():
    """Run the evaluation report."""
    logger.info("  STEP 2: Running evaluation report...")
    subprocess.run(
        [sys.executable, "app/training/evaluate.py"],
        cwd=PROJECT_ROOT,
        env=ENV
    )


def start_api():
    """Start the FastAPI backend."""
    logger.info(f"  STEP 3: Starting FastAPI backend on port {API_PORT}...")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "app.api.main:app",
            "--host", "0.0.0.0",
            "--port", str(API_PORT),
            "--log-level", "info"
        ],
        cwd=PROJECT_ROOT,
        env=ENV
    )
    processes.append(proc)
    return proc


def start_ui():
    """Start the Streamlit frontend."""
    logger.info(f"  STEP 4: Starting Streamlit UI on port {STREAMLIT_PORT}...")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run",
            "app/ui/streamlit_app.py",
            "--server.port", str(STREAMLIT_PORT),
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
        ],
        cwd=PROJECT_ROOT,
        env=ENV
    )
    processes.append(proc)
    return proc


def shutdown(signum=None, frame=None):
    """Gracefully terminate all child processes."""
    logger.info("\n👋 Shutting down all services...")
    for proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    logger.info("All services stopped.")
    sys.exit(0)


def wait_for_api(timeout: int = 30) -> bool:
    """Poll the API health endpoint until it's ready."""
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"http://localhost:{API_PORT}/health", timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def main():
    parser = argparse.ArgumentParser(description="Start the Churn Prediction System")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training (use existing model)")
    parser.add_argument("--retrain", action="store_true",
                        help="Force retraining even if model exists")
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════╗
║     Customer Churn Prediction System                 ║
║     Production-Style ML Pipeline                     ║
╚══════════════════════════════════════════════════════╝
    """)

    # Register signal handlers for clean shutdown
    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ── Training ──────────────────────────────────────────────────────────────
    should_train = (
        args.retrain
        or (not args.skip_train and not MODEL_PATH.exists())
    )
    if should_train:
        train_model()
        run_evaluation()
    elif MODEL_PATH.exists():
        logger.info(f"✓ Model found at {MODEL_PATH} — skipping training")
        logger.info("  Use --retrain to force retraining\n")
    else:
        logger.error("No model found and --skip-train was set. Run without --skip-train first.")
        sys.exit(1)

    # ── Services ──────────────────────────────────────────────────────────────
    logger.info("="*55)
    logger.info("  Starting services...")
    logger.info("="*55)

    api_proc = start_api()
    time.sleep(1)

    # Wait for API to be healthy before starting UI
    logger.info("  Waiting for API to be ready...")
    if wait_for_api(timeout=30):
        logger.info(f"  ✅ API ready at http://localhost:{API_PORT}")
        logger.info(f"     Docs: http://localhost:{API_PORT}/docs")
    else:
        logger.warning("  API didn't respond in time, starting UI anyway...")

    ui_proc = start_ui()

    print(f"""
╔══════════════════════════════════════════════════════╗
║  🚀 System is running!                               ║
║                                                      ║
║  Streamlit UI  : http://localhost:{STREAMLIT_PORT}         ║
║  FastAPI Docs  : http://localhost:{API_PORT}/docs          ║
║  Health check  : http://localhost:{API_PORT}/health        ║
║                                                      ║
║  Press Ctrl+C to stop                                ║
╚══════════════════════════════════════════════════════╝
    """)

    # Keep main thread alive; subprocess output goes to terminal
    try:
        while True:
            # Check if either process died unexpectedly
            if api_proc.poll() is not None:
                logger.error("API process exited unexpectedly!")
                shutdown()
            if ui_proc.poll() is not None:
                logger.error("UI process exited unexpectedly!")
                shutdown()
            time.sleep(5)
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()
