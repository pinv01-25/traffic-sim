#!/usr/bin/env python3
"""Run an A/B experiment: run baseline (A) and optimized (B) then compare.

This script executes two simulation runs:
  - A (Baseline): Original traffic light timings
  - B (Optimized): Modified timings via --green-time parameter

Then generates comprehensive comparison visualizations and statistical analysis.

Usage:
    uv run python run_ab.py sim.zip --green-time 30.0

    # With custom labels
    uv run python run_ab.py sim.zip --green-time 25 --label-a "Default" --label-b "25s Green"

    # Include native SUMO visualization tools
    uv run python run_ab.py sim.zip --green-time 30 --use-sumo-tools
"""
import argparse
import subprocess
import sys
from pathlib import Path
import os


def _run_instance(
    python_exec: str,
    runner_path: str,
    zip_file: str,
    extract_dir: str,
    extra_args: list[str]
) -> int:
    """Run a single simulation instance."""
    cmd = [
        python_exec, str(runner_path), zip_file,
        "--extract-dir", extract_dir,
        "--keep-files"
    ] + extra_args

    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=str(Path(runner_path).parent))
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run A/B experiment comparing baseline vs optimized traffic light timing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic A/B test with 30s green time
  uv run python run_ab.py sim.zip --green-time 30.0

  # Custom simulation steps
  uv run python run_ab.py sim.zip --green-time 25 --sim-steps 600

  # With custom labels for reports
  uv run python run_ab.py sim.zip --green-time 30 --label-a "Original" --label-b "Optimized 30s"

  # Include SUMO native tools output
  uv run python run_ab.py sim.zip --green-time 30 --use-sumo-tools

  # Skip running simulations, just analyze existing outputs
  uv run python run_ab.py sim.zip --analyze-only --extract-base sim
        """
    )

    # Required arguments
    parser.add_argument('zip_file', help='Path to simulation ZIP file')

    # Simulation parameters
    parser.add_argument('--green-time', type=float, default=30.0,
                        help='Green light duration for optimized run B (seconds)')
    parser.add_argument('--cycle-time', type=float, default=60.0,
                        help='Total traffic light cycle time (seconds)')
    parser.add_argument('--sim-steps', type=int, default=300,
                        help='Number of simulation steps to run')

    # Directory options
    parser.add_argument('--extract-base', default='sim',
                        help='Base name for extract dirs; creates <base>_A and <base>_B')
    parser.add_argument('--output-dir', default=None,
                        help='Custom output directory for visualizations')

    # Labels
    parser.add_argument('--label-a', default='Baseline',
                        help='Label for baseline run A')
    parser.add_argument('--label-b', default='Optimized',
                        help='Label for optimized run B')

    # Execution options
    parser.add_argument('--gui', action='store_true',
                        help='Run simulations with SUMO GUI')
    parser.add_argument('--python', default=sys.executable,
                        help='Python executable to run simulation script')
    parser.add_argument('--keep-files', action='store_true',
                        help='Keep extracted files after runs')

    # Analysis options
    parser.add_argument('--use-sumo-tools', action='store_true',
                        help='Also generate plots using native SUMO visualization tools')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Skip running simulations, only analyze existing outputs')
    parser.add_argument('--time-bin', type=int, default=30,
                        help='Time bin size for time series aggregations (seconds)')

    # Verbosity
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Reduce output verbosity')

    args = parser.parse_args()

    repo_dir = Path(__file__).parent.resolve()
    runner = repo_dir / 'run_simulation.py'

    if not runner.exists():
        print(f"Error: run_simulation.py not found at {runner}")
        sys.exit(2)

    extract_a = f"{args.extract_base}_A"
    extract_b = f"{args.extract_base}_B"
    labels = (args.label_a, args.label_b)

    if not args.analyze_only:
        # Build command arguments for each run
        common_args = ["--sim-steps", str(args.sim_steps), "--cycle-time", str(args.cycle_time)]

        if args.gui:
            common_args.append('--gui')
        if args.keep_files:
            common_args.append('--keep-files')

        extra_a = common_args.copy()
        extra_b = common_args + ["--green-time", str(args.green_time)]

        # Run A (Baseline)
        print(f"\n{'#'*60}")
        print(f"# STARTING BASELINE RUN A")
        print(f"# Extract dir: {extract_a}")
        print(f"# Configuration: Default timing (no --green-time)")
        print(f"{'#'*60}")

        rc = _run_instance(args.python, str(runner), args.zip_file, extract_a, extra_a)
        if rc != 0:
            print(f"\nError: Baseline run A failed with exit code {rc}")
            sys.exit(rc)

        # Run B (Optimized)
        print(f"\n{'#'*60}")
        print(f"# STARTING OPTIMIZED RUN B")
        print(f"# Extract dir: {extract_b}")
        print(f"# Configuration: green-time={args.green_time}s, cycle-time={args.cycle_time}s")
        print(f"{'#'*60}")

        rc = _run_instance(args.python, str(runner), args.zip_file, extract_b, extra_b)
        if rc != 0:
            print(f"\nError: Optimized run B failed with exit code {rc}")
            sys.exit(rc)

    # Verify directories exist
    if not Path(extract_a).exists():
        print(f"Error: Run A directory not found: {extract_a}")
        sys.exit(1)
    if not Path(extract_b).exists():
        print(f"Error: Run B directory not found: {extract_b}")
        sys.exit(1)

    # Run A/B Analysis
    print(f"\n{'#'*60}")
    print(f"# RUNNING A/B ANALYSIS")
    print(f"# Comparing: {extract_a} ({labels[0]}) vs {extract_b} ({labels[1]})")
    print(f"{'#'*60}\n")

    try:
        sys.path.insert(0, str(repo_dir))
        from visualization import generate_ab_test, quick_compare, compare_multiple_metrics

        # Determine output directory
        if args.output_dir:
            out_dir = args.output_dir
        else:
            out_dir = str(Path(extract_b) / 'logs' / 'visualizations' / 'ab_test')

        # Run comprehensive comparison
        report = generate_ab_test(
            extract_a,
            extract_b,
            out_dir=out_dir,
            labels=labels,
            use_sumo_tools=args.use_sumo_tools,
        )

        # Print summary
        print(f"\n{'='*60}")
        print("A/B TEST RESULTS SUMMARY")
        print(f"{'='*60}\n")

        if 'statistical_results' in report:
            stats = report['statistical_results']

            if 'percent_improvement' in stats:
                pct = stats['percent_improvement']
                direction = "improvement" if pct > 0 else "degradation"
                print(f"Travel Time Change: {pct:+.2f}% ({direction})")

            if 'mean_difference' in stats:
                diff = stats['mean_difference']
                print(f"Mean Difference (A - B): {diff:.2f} seconds")

            if 'permutation_test_pvalue' in stats:
                p = stats['permutation_test_pvalue']
                significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"Permutation Test p-value: {p:.4f} {significance}")

            if 'cohens_d' in stats:
                d = stats['cohens_d']
                effect = "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
                print(f"Effect Size (Cohen's d): {d:.3f} ({effect})")

            if 'bootstrap_ci_95' in stats:
                ci = stats['bootstrap_ci_95']
                print(f"95% CI for Mean Diff: [{ci['lower']:.2f}, {ci['upper']:.2f}]")

        print(f"\nGenerated files: {len(report.get('generated_files', []))}")
        print(f"Output directory: {out_dir}")
        print(f"CSV report: {report.get('csv', 'N/A')}")
        print(f"JSON report: {report.get('json', 'N/A')}")

        # Quick metrics table
        print(f"\n{'='*60}")
        print("METRICS COMPARISON TABLE")
        print(f"{'='*60}")

        metrics_df = compare_multiple_metrics(extract_a, extract_b)
        if not metrics_df.empty:
            print(f"\n{'Metric':<15} {'Mean A':>10} {'Mean B':>10} {'Diff':>10} {'%Improv':>10} {'p-value':>10}")
            print("-" * 70)
            for _, row in metrics_df.iterrows():
                print(f"{row['metric']:<15} {row['mean_A']:>10.2f} {row['mean_B']:>10.2f} "
                      f"{row['diff']:>10.2f} {row['pct_improvement']:>9.1f}% {row['p_value']:>10.4f}")

        print(f"\n{'='*60}")
        print(f"Analysis complete! View results in: {out_dir}")
        print(f"{'='*60}\n")

    except ImportError as e:
        print(f"Error importing visualization module: {e}")
        print("Make sure all dependencies are installed: uv sync")
        sys.exit(1)
    except Exception as e:
        print(f"Error running A/B analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
