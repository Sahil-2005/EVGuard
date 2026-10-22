"""Root entrypoint for EVGuard: Electric Vehicle Battery Health Analysis."""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

# Fallback to sys.executable if .venv path is not resolved
PYTHON_EXE = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable


def run_env_check() -> int:
    """Run environment and PySpark prerequisite verification."""
    cmd = [PYTHON_EXE, str(PROJECT_ROOT / "scripts" / "check_env.py")]
    print(f"Executing: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def run_pipeline(force_demo: bool = False) -> int:
    """Run the Big Data PySpark ETL, SOH analytics, and K-Means training pipeline."""
    cmd = [PYTHON_EXE, str(PROJECT_ROOT / "scripts" / "run_pipeline.py")]
    if force_demo:
        cmd.append("--force-demo")
    print(f"Executing Big Data Pipeline: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def run_tests() -> int:
    """Run the automated test suite with pytest."""
    cmd = [PYTHON_EXE, "-m", "pytest", "tests", "-v"]
    print(f"Running Tests: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def run_lint() -> int:
    """Run Ruff linter to verify code quality."""
    cmd = [PYTHON_EXE, "-m", "ruff", "check", "src", "tests", "scripts", "app"]
    print(f"Running Ruff Linter: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def run_dashboard(port: int = 8501) -> int:
    """Launch the interactive Streamlit dashboard."""
    app_script = str(PROJECT_ROOT / "app" / "app.py")
    cmd = [
        PYTHON_EXE,
        "-m",
        "streamlit",
        "run",
        app_script,
        "--server.port",
        str(port),
        "--server.headless",
        "false",
        "--browser.gatherUsageStats",
        "false",
    ]
    print("\n" + "=" * 65)
    print(" Launching EVGuard Streamlit Dashboard ")
    print(f" URL: http://localhost:{port}")
    print(" Press Ctrl+C in terminal to stop.")
    print("=" * 65 + "\n")
    return subprocess.run(cmd).returncode


def main():
    parser = argparse.ArgumentParser(
        description="EVGuard: Electric Vehicle Battery Health Analysis Using Big Data Techniques"
    )
    parser.add_argument("--pipeline", action="store_true", help="Execute the Big Data PySpark and ML pipeline.")
    parser.add_argument("--force-demo", action="store_true", help="Force rebuild of synthetic demo dataset.")
    parser.add_argument("--dashboard", action="store_true", help="Launch the Streamlit interactive dashboard.")
    parser.add_argument("--check-env", action="store_true", help="Verify Python, Java 17, and PySpark setup.")
    parser.add_argument("--test", action="store_true", help="Execute automated test suite via pytest.")
    parser.add_argument("--lint", action="store_true", help="Run Ruff linter on all codebase files.")
    parser.add_argument("--port", type=int, default=8501, help="Port for Streamlit dashboard (default: 8501).")

    args = parser.parse_args()

    if args.check_env:
        sys.exit(run_env_check())
    elif args.test:
        sys.exit(run_tests())
    elif args.lint:
        sys.exit(run_lint())
    elif args.pipeline:
        sys.exit(run_pipeline(force_demo=args.force_demo))
    elif args.dashboard:
        sys.exit(run_dashboard(port=args.port))
    else:
        # Default behavior: Ensure demo data exists, then launch dashboard
        parquet_file = PROJECT_ROOT / "data" / "processed" / "battery_cycles.parquet"
        if not parquet_file.exists():
            print("Notice: Cleaned Parquet data not detected. Executing initial pipeline first...")
            res = run_pipeline(force_demo=True)
            if res != 0:
                print("Pipeline execution failed. Aborting dashboard startup.")
                sys.exit(res)
        sys.exit(run_dashboard(port=args.port))


if __name__ == "__main__":
    main()
