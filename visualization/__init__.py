"""Visualization utilities for traffic-sim.

This module provides comprehensive visualization and analysis tools for
SUMO traffic simulations, including:

- Single-run visualizations (tripinfo histograms, summary timelines)
- A/B test comparisons with statistical analysis
- Native SUMO tool wrappers
- FCD (Floating Car Data) analysis

Main entry points:
- generate_visualizations(): Generate plots for a single simulation run
- generate_ab_test(): Run comprehensive A/B comparison between two runs
- quick_compare(): Get quick statistics without generating plots
"""
from pathlib import Path
import os

from .parsers import (
    parse_tripinfo,
    parse_summary,
    parse_fcd,
    parse_fcd_aggregated,
    parse_fcd_by_edge,
    get_available_outputs,
    compute_trip_statistics,
    compute_summary_statistics,
)

from .plots import (
    # Single-run plots
    plot_summary_timeline,
    plot_tripinfo_histograms,
    plot_depart_delay_scatter,
    plot_network_traffic_lights,
    # A/B comparison plots
    plot_histogram_cdf_two,
    plot_boxplot_two,
    plot_time_series_mean,
    plot_metric_comparison_bars,
    plot_violin_comparison,
    plot_multi_metric_violin,
    plot_time_series_comparison,
    plot_summary_comparison,
    plot_efficiency_comparison,
    plot_speed_distribution_comparison,
    plot_waiting_time_analysis,
    plot_correlation_heatmap,
    plot_fcd_speed_heatmap,
    plot_fcd_comparison,
    plot_improvement_summary,
    plot_percentile_comparison,
    plot_congestion_timeline,
)

from .ab_test import (
    compare_runs,
    quick_compare,
    compare_multiple_metrics,
    permutation_test_mean,
    bootstrap_ci_diff,
    mann_whitney_test,
    cohens_d,
)

from .sumo_tools import (
    plot_xml_attributes,
    plot_summary as sumo_plot_summary,
    plot_tripinfo_distributions,
    plot_net_dump,
    plot_net_speeds,
    plot_trajectories,
    generate_all_sumo_plots,
    check_sumo_tools_available,
)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def generate_visualizations(simulation_dir: str, out_dir: str | None = None) -> str:
    """Generate a set of visualizations from SUMO outputs found under
    `simulation_dir`.

    Looks for common files in the following order:
      - <simulation_dir>/logs/sumo_output/*.xml
      - <simulation_dir>/*.xml

    Returns path to the output directory where PNGs were written.
    """
    sim_dir = Path(simulation_dir)
    candidate_dirs = [sim_dir / "logs" / "sumo_output", sim_dir]
    if out_dir is None:
        out_dir = str(sim_dir / "logs" / "visualizations")
    ensure_dir(out_dir)

    found_any = False
    for base in candidate_dirs:
        trip = base / "tripinfo.xml"
        summary = base / "summary.xml"
        fcd = base / "fcd.xml"

        if trip.exists():
            df_trip = parse_tripinfo(str(trip))
            plot_tripinfo_histograms(df_trip, out_dir)
            plot_depart_delay_scatter(df_trip, out_dir)

            # Additional single-run plots
            if not df_trip.empty:
                plot_correlation_heatmap(df_trip, out_dir, filename='correlation_heatmap.png')

            found_any = True

        if summary.exists():
            df_summary = parse_summary(str(summary))
            plot_summary_timeline(df_summary, out_dir)
            found_any = True

        if fcd.exists():
            df_fcd = parse_fcd(str(fcd), sample_rate=5)
            if not df_fcd.empty:
                plot_fcd_speed_heatmap(df_fcd, out_dir, filename='fcd_speed_heatmap.png')
            found_any = True

    if not found_any:
        # No SUMO outputs found — attempt network-based fallback (plot traffic lights)
        net_file = sim_dir / "network.net.xml"
        if net_file.exists():
            plot_network_traffic_lights(str(net_file), out_dir)
            found_any = True
        else:
            raise FileNotFoundError("No SUMO outputs (tripinfo.xml or summary.xml) found under simulation dir")

    return out_dir


def generate_ab_test(
    run_a: str,
    run_b: str,
    out_dir: str | None = None,
    labels: tuple[str, str] = ('Baseline', 'Optimized'),
    use_sumo_tools: bool = False,
) -> dict:
    """Run a comprehensive A/B comparison between two simulation runs.

    Generates 20+ visualizations comparing metrics, distributions, time series,
    and statistical tests.

    Args:
        run_a: Directory containing baseline simulation outputs
        run_b: Directory containing optimized simulation outputs
        out_dir: Output directory for plots and reports (default: run_b/logs/visualizations/ab_test)
        labels: Labels for the two runs
        use_sumo_tools: Whether to also generate plots using native SUMO tools

    Returns:
        Dictionary with paths to generated files and statistical results
    """
    if out_dir is None:
        out_dir = str(Path(run_b).resolve() / 'logs' / 'visualizations' / 'ab_test')

    return compare_runs(
        run_a,
        run_b,
        out_dir,
        labels=labels,
        generate_all_plots=True,
        use_sumo_tools=use_sumo_tools,
    )


# Convenience exports
__all__ = [
    # Main entry points
    'generate_visualizations',
    'generate_ab_test',
    'quick_compare',
    'compare_multiple_metrics',

    # Parsers
    'parse_tripinfo',
    'parse_summary',
    'parse_fcd',
    'parse_fcd_aggregated',
    'parse_fcd_by_edge',
    'get_available_outputs',
    'compute_trip_statistics',
    'compute_summary_statistics',

    # Single-run plots
    'plot_summary_timeline',
    'plot_tripinfo_histograms',
    'plot_depart_delay_scatter',
    'plot_network_traffic_lights',
    'plot_correlation_heatmap',
    'plot_fcd_speed_heatmap',

    # A/B comparison plots
    'plot_histogram_cdf_two',
    'plot_boxplot_two',
    'plot_time_series_mean',
    'plot_metric_comparison_bars',
    'plot_violin_comparison',
    'plot_multi_metric_violin',
    'plot_time_series_comparison',
    'plot_summary_comparison',
    'plot_efficiency_comparison',
    'plot_speed_distribution_comparison',
    'plot_waiting_time_analysis',
    'plot_fcd_comparison',
    'plot_improvement_summary',
    'plot_percentile_comparison',
    'plot_congestion_timeline',

    # Statistical functions
    'compare_runs',
    'permutation_test_mean',
    'bootstrap_ci_diff',
    'mann_whitney_test',
    'cohens_d',

    # SUMO native tools
    'plot_xml_attributes',
    'sumo_plot_summary',
    'plot_tripinfo_distributions',
    'plot_net_dump',
    'plot_net_speeds',
    'plot_trajectories',
    'generate_all_sumo_plots',
    'check_sumo_tools_available',
]
