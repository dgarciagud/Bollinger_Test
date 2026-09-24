import os
import logging
import pickle
import yaml
import typer
import pandas as pd
from vol_spillover.download import run_download
from vol_spillover.clean import run_clean
from vol_spillover.realized import run_realized
from vol_spillover.var_utils import select_var_lag, estimate_var_ols, compute_ma_coefficients, compute_gfevd
from vol_spillover.connectedness.dy_rolling import run_rolling_dy
from vol_spillover.connectedness.tvp_var import run_tvp_var
from vol_spillover.connectedness.quantile import run_quantile_connectedness
from vol_spillover.connectedness.frequency import run_frequency_connectedness
from vol_spillover.events import analyze_events
from vol_spillover.plots import generate_plots
from vol_spillover.summary import generate_summary_md

app = typer.Typer(help="Volatility Spillover Analysis Pipeline")
logger = logging.getLogger("vol_spillover.cli")


@app.command()
def download(
    config: str = typer.Option("config/config.yaml", help="Path to config YAML file"),
    force_bi5: bool = typer.Option(False, help="Force fallback bi5 downloader")
):
    """Download intraday M1 data from Dukascopy."""
    typer.echo("Starting download step...")
    run_download(config_path=config, force_bi5=force_bi5)
    typer.echo("Download step completed.")


@app.command()
def clean(
    config: str = typer.Option("config/config.yaml", help="Path to config YAML file")
):
    """Clean and synchronize M1 mid prices."""
    typer.echo("Starting cleaning step...")
    run_clean(config_path=config)
    typer.echo("Cleaning step completed.")


@app.command()
def realized(
    config: str = typer.Option("config/config.yaml", help="Path to config YAML file")
):
    """Calculate sub-sampled Realized Volatility and features."""
    typer.echo("Starting realized volatility calculation...")
    run_realized(config_path=config)
    typer.echo("Realized volatility calculation completed.")


@app.command("connect")
def connect_cmd(
    method: str = typer.Option("all", help="Connectedness method: dy, tvp, quantile, freq, all"),
    config: str = typer.Option("config/config.yaml", help="Path to config YAML file")
):
    """Estimate volatility connectedness and spillovers and cache results."""
    typer.echo(f"Starting connectedness estimation (method={method})...")
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    features_dir = cfg.get("paths", {}).get("features_dir", "data/features")
    log_rv_file = os.path.join(features_dir, "log_rv.parquet")
    if not os.path.exists(log_rv_file):
        typer.echo(f"Error: {log_rv_file} not found. Run realized step first.")
        raise typer.Exit(code=1)

    log_rv_df = pd.read_parquet(log_rv_file).dropna()
    if log_rv_df.empty:
        typer.echo("Error: log_rv.parquet is empty.")
        raise typer.Exit(code=1)

    window = cfg.get("connect", {}).get("dy_rolling_window", 100)
    H = cfg.get("connect", {}).get("forecast_horizon", 10)
    max_lag = cfg.get("connect", {}).get("var_max_lag", 5)

    results = {}

    if method in ["dy", "all"]:
        typer.echo("Running Rolling Diebold-Yilmaz...")
        results["dy"] = run_rolling_dy(log_rv_df, window=window, H=H, max_lag=max_lag)

    if method in ["tvp", "all"]:
        typer.echo("Running TVP-VAR...")
        results["tvp"] = run_tvp_var(log_rv_df, lag=1, H=H, training_window=min(window, len(log_rv_df)//2))

    if method in ["quantile", "all"]:
        typer.echo("Running Quantile VAR...")
        results["quantile"] = run_quantile_connectedness(log_rv_df, quantiles=[0.05, 0.50, 0.95], window=window, H=H)

    if method in ["freq", "all"]:
        typer.echo("Running Frequency Connectedness...")
        results["freq"] = run_frequency_connectedness(log_rv_df, lag=1)

    # Save to cache file
    cache_path = os.path.join(features_dir, "connectedness_cache.pkl")
    with open(cache_path, "wb") as f:
        pickle.dump(results, f)
    typer.echo(f"Connectedness metrics saved to {cache_path}.")


@app.command()
def report(
    config: str = typer.Option("config/config.yaml", help="Path to config YAML file")
):
    """Generate plots, event analysis, and summary.md report."""
    typer.echo("Generating reports and visualizations...")
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    features_dir = cfg.get("paths", {}).get("features_dir", "data/features")
    log_rv_file = os.path.join(features_dir, "log_rv.parquet")
    abnormal_file = os.path.join(features_dir, "abnormal_flags.parquet")
    cache_path = os.path.join(features_dir, "connectedness_cache.pkl")

    if not os.path.exists(log_rv_file) or not os.path.exists(abnormal_file):
        typer.echo("Error: Feature files missing. Run realized step first.")
        raise typer.Exit(code=1)

    log_rv_df = pd.read_parquet(log_rv_file).dropna()
    abnormal_df = pd.read_parquet(abnormal_file)

    if os.path.exists(cache_path):
        typer.echo(f"Loading connectedness results from {cache_path}...")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
            dy_res = cache.get("dy", {})
            tvp_res = cache.get("tvp", {})
            q_res = cache.get("quantile", {})
            freq_res = cache.get("freq", {})
    else:
        typer.echo("No connectedness cache found. Calculating connectedness...")
        window = cfg.get("connect", {}).get("dy_rolling_window", 100)
        H = cfg.get("connect", {}).get("forecast_horizon", 10)

        dy_res = run_rolling_dy(log_rv_df, window=window, H=H)
        tvp_res = run_tvp_var(log_rv_df, lag=1, H=H, training_window=min(window, len(log_rv_df)//2))
        q_res = run_quantile_connectedness(log_rv_df, quantiles=[0.05, 0.50, 0.95], window=window, H=H)
        freq_res = run_frequency_connectedness(log_rv_df, lag=1)

    window = cfg.get("connect", {}).get("dy_rolling_window", 100)
    H = cfg.get("connect", {}).get("forecast_horizon", 10)

    # Static GFEVD for heatmap
    lag = select_var_lag(log_rv_df, max_lag=5)
    A_list, sigma_u, _ = estimate_var_ols(log_rv_df, lag=lag)
    Phi = compute_ma_coefficients(A_list, H=H)
    gfevd_static = pd.DataFrame(compute_gfevd(Phi, sigma_u, H=H), index=log_rv_df.columns, columns=log_rv_df.columns)

    analyze_events(log_rv_df, abnormal_df, dy_res, cfg)
    generate_plots(dy_res, tvp_res, q_res, freq_res, gfevd_static, cfg)
    generate_summary_md(dy_res, q_res, tvp_res, freq_res, cfg)

    typer.echo("Report generation completed successfully.")


@app.command("run-all")
def run_all(
    config: str = typer.Option("config/config.yaml", help="Path to config YAML file"),
    force_bi5: bool = typer.Option(False, help="Force bi5 fallback")
):
    """Run full end-to-end pipeline: download -> clean -> realized -> connect -> report."""
    typer.echo("========== RUNNING FULL PIPELINE ==========")
    run_download(config_path=config, force_bi5=force_bi5)
    run_clean(config_path=config)
    run_realized(config_path=config)
    connect_cmd(method="all", config=config)
    report(config=config)
    typer.echo("========== PIPELINE EXECUTED SUCCESSFULLY ==========")


if __name__ == "__main__":
    app()
