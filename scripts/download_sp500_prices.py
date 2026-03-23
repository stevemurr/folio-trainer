"""Download S&P 500 OHLCV price data via yfinance."""

import logging
import datetime as dt
from pathlib import Path

import polars as pl
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# S&P 500 constituents as of early 2025 (comprehensive list)
SP500_TICKERS = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "ADI", "ADM", "ADP", "ADSK", "AEE",
    "AEP", "AES", "AFL", "AIG", "AIZ", "AJG", "AKAM", "ALB", "ALGN", "ALK",
    "ALL", "ALLE", "AMAT", "AMCR", "AMD", "AME", "AMGN", "AMP", "AMT", "AMZN",
    "ANET", "ANSS", "AON", "AOS", "APA", "APD", "APH", "APTV", "ARE", "ATO",
    "ATVI", "AVB", "AVGO", "AVY", "AWK", "AXP", "AZO", "BA", "BAC", "BAX",
    "BBWI", "BBY", "BDX", "BEN", "BF-B", "BIO", "BIIB", "BK", "BKNG", "BKR",
    "BLK", "BMY", "BR", "BRK-B", "BRO", "BSX", "BWA", "BXP", "C", "CAG",
    "CAH", "CARR", "CAT", "CB", "CBOE", "CBRE", "CCI", "CCL", "CDAY", "CDNS",
    "CDW", "CE", "CEG", "CF", "CFG", "CHD", "CHRW", "CHTR", "CI", "CINF",
    "CL", "CLX", "CMA", "CMCSA", "CME", "CMG", "CMI", "CMS", "CNC", "CNP",
    "COF", "COO", "COP", "COST", "CPB", "CPRT", "CPT", "CRL", "CRM", "CSCO",
    "CSGP", "CSX", "CTAS", "CTLT", "CTRA", "CTSH", "CTVA", "CVS", "CVX", "CZR",
    "D", "DAL", "DD", "DE", "DFS", "DG", "DGX", "DHI", "DHR", "DIS",
    "DISH", "DLTR", "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA", "DVN",
    "DXC", "DXCM", "EA", "EBAY", "ECL", "ED", "EFX", "EIX", "EL", "EMN",
    "EMR", "ENPH", "EOG", "EPAM", "EQIX", "EQR", "EQT", "ES", "ESS", "ETN",
    "ETR", "ETSY", "EVRG", "EW", "EXC", "EXPD", "EXPE", "EXR", "F", "FANG",
    "FAST", "FBHS", "FCX", "FDS", "FDX", "FE", "FFIV", "FIS", "FISV", "FITB",
    "FLT", "FMC", "FOX", "FOXA", "FRC", "FRT", "FTNT", "FTV", "GD", "GE",
    "GEHC", "GEN", "GILD", "GIS", "GL", "GLW", "GM", "GNRC", "GOOG", "GOOGL",
    "GPC", "GPN", "GRMN", "GS", "GWW", "HAL", "HAS", "HBAN", "HCA", "HOLX",
    "HON", "HPE", "HPQ", "HRL", "HSIC", "HST", "HSY", "HUM", "HWM", "IBM",
    "ICE", "IDXX", "IEX", "IFF", "ILMN", "INCY", "INTC", "INTU", "INVH", "IP",
    "IPG", "IQV", "IR", "IRM", "ISRG", "IT", "ITW", "IVZ", "J", "JBHT",
    "JCI", "JKHY", "JNJ", "JNPR", "JPM", "K", "KDP", "KEY", "KEYS", "KHC",
    "KIM", "KLAC", "KMB", "KMI", "KMX", "KO", "KR", "L", "LDOS", "LEN",
    "LH", "LHX", "LIN", "LKQ", "LLY", "LMT", "LNC", "LNT", "LOW", "LRCX",
    "LUMN", "LUV", "LVS", "LW", "LYB", "LYV", "MA", "MAA", "MAR", "MAS",
    "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT", "MET", "META", "MGM", "MHK",
    "MKC", "MKTX", "MLM", "MMC", "MMM", "MNST", "MO", "MOH", "MOS", "MPC",
    "MPWR", "MRK", "MRNA", "MRO", "MS", "MSCI", "MSFT", "MSI", "MTB", "MTCH",
    "MTD", "MU", "NCLH", "NDAQ", "NDSN", "NEE", "NEM", "NFLX", "NI", "NKE",
    "NOC", "NOW", "NRG", "NSC", "NTAP", "NTRS", "NUE", "NVDA", "NVR", "NWL",
    "NWS", "NWSA", "NXPI", "O", "ODFL", "OGN", "OKE", "OMC", "ON", "ORCL",
    "ORLY", "OTIS", "OXY", "PARA", "PAYC", "PAYX", "PCAR", "PCG", "PEAK", "PEG",
    "PEP", "PFE", "PFG", "PG", "PGR", "PH", "PHM", "PKG", "PKI", "PLD",
    "PM", "PNC", "PNR", "PNW", "POOL", "PPG", "PPL", "PRU", "PSA", "PSX",
    "PTC", "PVH", "PWR", "PXD", "PYPL", "QCOM", "QRVO", "RCL", "RE", "REG",
    "REGN", "RF", "RHI", "RJF", "RL", "RMD", "ROK", "ROL", "ROP", "ROST",
    "RSG", "RTX", "RVTY", "SBAC", "SBNY", "SBUX", "SCHW", "SEE", "SHW", "SIVB",
    "SJM", "SLB", "SNA", "SNPS", "SO", "SPG", "SPGI", "SRE", "STE", "STT",
    "STX", "STZ", "SWK", "SWKS", "SYF", "SYK", "SYY", "T", "TAP", "TDG",
    "TDY", "TECH", "TEL", "TER", "TFC", "TFX", "TGT", "TMO", "TMUS", "TPR",
    "TRGP", "TRMB", "TROW", "TRV", "TSCO", "TSLA", "TSN", "TT", "TTWO", "TXN",
    "TXT", "TYL", "UAL", "UDR", "UHS", "ULTA", "UNH", "UNP", "UPS", "URI",
    "USB", "V", "VFC", "VICI", "VLO", "VMC", "VNO", "VRSK", "VRSN", "VRTX",
    "VTR", "VTRS", "VZ", "WAB", "WAT", "WBA", "WBD", "WDC", "WEC", "WELL",
    "WFC", "WHR", "WM", "WMB", "WMT", "WRB", "WRK", "WST", "WTW", "WY",
    "WYNN", "XEL", "XOM", "XRAY", "XYL", "YUM", "ZBH", "ZBRA", "ZION", "ZTS",
]


def download_sp500_prices(output_dir: str = "data", start_date: str = "2010-01-01"):
    """Download S&P 500 OHLCV data and save as Parquet."""
    output_path = Path(output_dir) / "raw" / "prices"
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "prices_daily.parquet"

    logger.info("Downloading %d S&P 500 tickers from %s...", len(SP500_TICKERS), start_date)

    # Download in batches to avoid timeouts
    batch_size = 50
    all_frames = []

    for i in range(0, len(SP500_TICKERS), batch_size):
        batch = SP500_TICKERS[i : i + batch_size]
        batch_str = " ".join(batch)
        logger.info("Batch %d/%d: downloading %d tickers...",
                     i // batch_size + 1,
                     (len(SP500_TICKERS) + batch_size - 1) // batch_size,
                     len(batch))
        try:
            df = yf.download(
                batch_str,
                start=start_date,
                auto_adjust=False,
                group_by="ticker",
                threads=True,
            )
            if df is None or df.empty:
                logger.warning("No data returned for batch starting at index %d", i)
                continue

            # yf.download with group_by="ticker" returns MultiIndex columns: (ticker, field)
            # Reshape to long format
            for ticker in batch:
                try:
                    if len(batch) == 1:
                        ticker_df = df
                    else:
                        if ticker not in df.columns.get_level_values(0):
                            continue
                        ticker_df = df[ticker]

                    if ticker_df.empty or ticker_df["Close"].isna().all():
                        continue

                    records = {
                        "date": ticker_df.index.date.tolist(),
                        "ticker": [ticker] * len(ticker_df),
                        "open": ticker_df["Open"].values.tolist(),
                        "high": ticker_df["High"].values.tolist(),
                        "low": ticker_df["Low"].values.tolist(),
                        "close": ticker_df["Close"].values.tolist(),
                        "adj_close": ticker_df["Adj Close"].values.tolist(),
                        "volume": ticker_df["Volume"].values.tolist(),
                    }
                    pldf = pl.DataFrame(records)
                    # Cast all numeric columns to Float64 before concat
                    pldf = pldf.with_columns(
                        pl.col("open").cast(pl.Float64),
                        pl.col("high").cast(pl.Float64),
                        pl.col("low").cast(pl.Float64),
                        pl.col("close").cast(pl.Float64),
                        pl.col("adj_close").cast(pl.Float64),
                        pl.col("volume").cast(pl.Float64),
                    )
                    # Drop rows where close is null
                    pldf = pldf.filter(pl.col("close").is_not_null())
                    if len(pldf) > 0:
                        all_frames.append(pldf)
                except Exception as e:
                    logger.warning("Failed to process %s: %s", ticker, e)
        except Exception as e:
            logger.warning("Batch download failed: %s", e)

    if not all_frames:
        logger.error("No price data downloaded!")
        return

    # Combine all frames
    logger.info("Combining %d ticker frames...", len(all_frames))
    combined = pl.concat(all_frames)

    # Type coercion
    combined = combined.with_columns(
        pl.col("date").cast(pl.Date),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("adj_close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
    )

    # Add dollar_volume and source_ts
    combined = combined.with_columns(
        (pl.col("close") * pl.col("volume")).alias("dollar_volume"),
        pl.lit(dt.datetime.now(dt.timezone.utc)).alias("source_ts"),
    )

    # Deduplicate and sort
    combined = combined.unique(subset=["date", "ticker"], keep="last").sort(["ticker", "date"])

    # Save
    combined.write_parquet(output_file)

    # Write metadata
    import json
    meta = {
        "ingested_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "row_count": len(combined),
        "ticker_count": combined["ticker"].n_unique(),
        "tickers": sorted(combined["ticker"].unique().to_list()),
        "date_range": [str(combined["date"].min()), str(combined["date"].max())],
        "source": "yfinance",
    }
    (output_path / "_metadata.json").write_text(json.dumps(meta, indent=2))

    logger.info(
        "Done! %d rows, %d tickers, %s to %s -> %s",
        len(combined), meta["ticker_count"],
        meta["date_range"][0], meta["date_range"][1],
        output_file,
    )


if __name__ == "__main__":
    download_sp500_prices()
