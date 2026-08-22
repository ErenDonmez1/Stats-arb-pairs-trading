"""Market-data acquisition, validation, caching, and synthetic fixtures."""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

OBSERVED_PRICE_MASK_ATTR = "observed_price_mask"
_CACHE_FORMAT_VERSION = 1
_CACHE_SOURCE = "yahoo_finance"
_CACHE_ADJUSTMENT_POLICY = "auto_adjust_true_close"


class DataQualityError(ValueError):
    """Raised when prices are unusable for statistical research."""


class MarketDataLoader:
    """Download adjusted closing prices from Yahoo Finance with exact caching."""

    def __init__(self, cache_dir: str | Path = "data/raw") -> None:
        self.cache_dir = Path(cache_dir)

    @staticmethod
    def normalize_tickers(tickers: Iterable[str]) -> tuple[str, ...]:
        """Return sorted, uppercase, unique symbols after discarding blanks."""
        if isinstance(tickers, str):
            tickers = [tickers]

        normalized: set[str] = set()
        for ticker in tickers:
            if not isinstance(ticker, str):
                raise TypeError("Ticker symbols must be strings.")
            symbol = ticker.strip().upper()
            if symbol:
                normalized.add(symbol)

        symbols = tuple(sorted(normalized))
        if len(symbols) < 2:
            raise ValueError("At least two unique, non-empty tickers are required.")
        return symbols

    def download(
        self,
        tickers: Iterable[str],
        start: str,
        end: str | None = None,
        interval: str = "1d",
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Return an exact requested universe from cache or Yahoo Finance.

        Prices are adjusted by asking yfinance to apply corporate actions through
        ``auto_adjust=True``. Cleaning is deliberately separate so the raw provider
        response remains auditable.
        """
        symbols = self.normalize_tickers(tickers)
        start_value = self._non_empty_request_string(start, "start")
        if end is not None:
            end_value = self._non_empty_request_string(end, "end")
        else:
            end_value = None
        interval_value = self._non_empty_request_string(interval, "interval")
        if not isinstance(refresh, (bool, np.bool_)):
            raise TypeError("refresh must be Boolean.")

        resolved_end = end_value or self._cache_as_of_date()
        metadata = self._cache_metadata(
            symbols,
            start=start_value,
            requested_end=end_value,
            resolved_end=resolved_end,
            interval=interval_value,
        )
        cache_path = self._cache_path(
            symbols,
            start_value,
            end_value,
            resolved_end,
            interval_value,
        )

        if cache_path.is_file() and not refresh:
            return self._read_cache(cache_path, symbols, metadata)

        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError(
                "yfinance is required to download Yahoo Finance prices."
            ) from exc

        LOGGER.info("Downloading %d symbols from Yahoo Finance", len(symbols))
        try:
            raw = yf.download(
                tickers=list(symbols),
                start=start_value,
                end=end_value,
                interval=interval_value,
                auto_adjust=True,
                actions=False,
                progress=False,
                group_by="column",
                threads=True,
            )
        except Exception as exc:
            raise ConnectionError(f"Yahoo Finance download failed: {exc}") from exc

        prices = self._extract_close_prices(raw, symbols)
        self._write_cache(cache_path, prices, metadata)
        return prices

    @staticmethod
    def _non_empty_request_string(value: Any, field: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{field} must be a string.")
        normalised = value.strip()
        if not normalised:
            raise ValueError(f"{field} must not be empty.")
        return normalised

    @staticmethod
    def _cache_as_of_date() -> str:
        """Return the UTC calendar date defining an open request's horizon."""
        return pd.Timestamp.now(tz="UTC").date().isoformat()

    @staticmethod
    def _cache_metadata(
        symbols: tuple[str, ...],
        *,
        start: str,
        requested_end: str | None,
        resolved_end: str,
        interval: str,
    ) -> dict[str, Any]:
        return {
            "format_version": _CACHE_FORMAT_VERSION,
            "source": _CACHE_SOURCE,
            "symbols": list(symbols),
            "start": start,
            "requested_end": requested_end,
            "resolved_end": resolved_end,
            "interval": interval,
            "adjustment_policy": _CACHE_ADJUSTMENT_POLICY,
        }

    def _cache_path(
        self,
        symbols: tuple[str, ...],
        start: str,
        requested_end: str | None,
        resolved_end: str,
        interval: str,
    ) -> Path:
        """Build a stable cache name from the complete request identity."""
        request_mode = "OPEN_ENDED" if requested_end is None else "EXPLICIT_END"
        identity = "\x1f".join(
            [
                str(_CACHE_FORMAT_VERSION),
                _CACHE_SOURCE,
                _CACHE_ADJUSTMENT_POLICY,
                request_mode,
                *symbols,
                str(start),
                str(resolved_end),
                str(interval),
            ]
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        safe_interval = re.sub(r"[^A-Za-z0-9_-]+", "-", str(interval)).strip("-")
        safe_interval = safe_interval or "interval"
        return self.cache_dir / f"adjusted_close_{safe_interval}_{digest}.csv"

    @classmethod
    def _read_cache(
        cls,
        cache_path: Path,
        symbols: tuple[str, ...],
        expected_metadata: dict[str, Any],
    ) -> pd.DataFrame:
        metadata_path = cls._cache_metadata_path(cache_path)
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DataQualityError(
                f"Could not read cache metadata {metadata_path}: {exc}"
            ) from exc
        if not isinstance(metadata, dict):
            raise DataQualityError("Cached request metadata must be a JSON object.")
        retrieved_at = metadata.pop("retrieved_at_utc", None)
        try:
            retrieved_timestamp = pd.Timestamp(retrieved_at)
        except (TypeError, ValueError) as exc:
            raise DataQualityError(
                "Cached retrieval timestamp is missing or invalid."
            ) from exc
        if retrieved_timestamp.tzinfo is None:
            raise DataQualityError("Cached retrieval timestamp must be UTC-aware.")
        if metadata != expected_metadata:
            raise DataQualityError(
                "Cached request metadata does not exactly match the current request."
            )
        try:
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            raise DataQualityError(f"Could not read cache file {cache_path}: {exc}") from exc

        normalized_columns = cls._normalize_provider_columns(cached.columns)
        cached.columns = normalized_columns
        if set(cached.columns) != set(symbols) or len(cached.columns) != len(symbols):
            raise DataQualityError(
                "Cached ticker universe does not exactly match the request: "
                f"expected {list(symbols)}, found {list(cached.columns)}."
            )
        if cached.empty:
            raise DataQualityError(f"Cache file contains no observations: {cache_path}")
        if not isinstance(cached.index, pd.DatetimeIndex) or cached.index.hasnans:
            raise DataQualityError(f"Cache file has an invalid date index: {cache_path}")
        cached = cached.loc[:, list(symbols)].sort_index()
        if cached.isna().all(axis=0).any():
            unusable = cached.columns[cached.isna().all(axis=0)].tolist()
            raise DataQualityError(f"Cached tickers contain no prices: {unusable}")
        return cached

    @staticmethod
    def _cache_metadata_path(cache_path: Path) -> Path:
        return cache_path.with_suffix(cache_path.suffix + ".json")

    def _write_cache(
        self,
        cache_path: Path,
        prices: pd.DataFrame,
        metadata: dict[str, Any],
    ) -> None:
        """Replace one cache and its exact request metadata deterministically."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = self._cache_metadata_path(cache_path)
        csv_temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        metadata_temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        try:
            prices.to_csv(csv_temporary, index_label="Date")
            stored_metadata = dict(metadata)
            stored_metadata["retrieved_at_utc"] = pd.Timestamp.now(
                tz="UTC"
            ).isoformat()
            metadata_temporary.write_text(
                json.dumps(stored_metadata, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            csv_temporary.replace(cache_path)
            metadata_temporary.replace(metadata_path)
        except OSError as exc:
            raise DataQualityError(f"Could not write cache {cache_path}: {exc}") from exc
        finally:
            for temporary in (csv_temporary, metadata_temporary):
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    LOGGER.warning("Could not remove cache temporary file %s", temporary)

    @classmethod
    def _extract_close_prices(
        cls,
        raw: Any,
        symbols: tuple[str, ...],
    ) -> pd.DataFrame:
        """Extract Close values from supported yfinance column layouts."""
        if not isinstance(raw, pd.DataFrame):
            raise DataQualityError("Yahoo Finance returned a non-tabular response.")
        if raw.empty:
            raise DataQualityError("Yahoo Finance returned no observations.")

        if isinstance(raw.columns, pd.MultiIndex):
            prices = cls._extract_multiindex_close(raw)
        else:
            columns = cls._normalize_provider_columns(raw.columns)
            raw = raw.copy()
            raw.columns = columns
            if set(symbols).issubset(raw.columns):
                # Some mocked/proxy providers return an already flattened
                # ticker-by-column adjusted-close frame.
                prices = raw.loc[:, list(symbols)]
            elif len(symbols) == 1 and "CLOSE" in raw.columns:
                prices = raw.loc[:, ["CLOSE"]].rename(columns={"CLOSE": symbols[0]})
            elif "CLOSE" in raw.columns:
                raise DataQualityError(
                    "Yahoo Finance returned a single-ticker OHLC layout for a "
                    "multi-ticker request."
                )
            else:
                raise DataQualityError(
                    "Yahoo Finance response has no adjusted Close columns."
                )

        prices = prices.copy()
        prices.columns = cls._normalize_provider_columns(prices.columns)
        if prices.columns.duplicated().any():
            raise DataQualityError("Yahoo Finance returned duplicate ticker columns.")

        missing = sorted(set(symbols) - set(prices.columns))
        if missing:
            raise DataQualityError(
                f"Yahoo Finance did not return requested tickers: {missing}"
            )
        prices = prices.loc[:, list(symbols)]
        if prices.isna().all(axis=0).any():
            unusable = prices.columns[prices.isna().all(axis=0)].tolist()
            raise DataQualityError(
                f"Yahoo Finance returned no adjusted prices for: {unusable}"
            )
        prices.index.name = "Date"
        return prices

    @staticmethod
    def _extract_multiindex_close(raw: pd.DataFrame) -> pd.DataFrame:
        """Handle both (field, ticker) and (ticker, field) MultiIndexes."""
        for level in range(raw.columns.nlevels):
            labels = pd.Index(raw.columns.get_level_values(level)).map(
                lambda value: str(value).strip().upper()
            )
            close_positions = np.flatnonzero(labels == "CLOSE")
            if close_positions.size:
                selected = raw.iloc[:, close_positions].copy()
                remaining = selected.columns.droplevel(level)
                if isinstance(remaining, pd.MultiIndex):
                    if remaining.nlevels != 1:
                        raise DataQualityError(
                            "Yahoo Finance returned an unsupported column hierarchy."
                        )
                    remaining = remaining.get_level_values(0)
                selected.columns = remaining
                return selected
        raise DataQualityError("Yahoo Finance response has no adjusted Close field.")

    @staticmethod
    def _normalize_provider_columns(columns: Iterable[Any]) -> pd.Index:
        return pd.Index([str(column).strip().upper() for column in columns])

    @staticmethod
    def clean(
        prices: pd.DataFrame,
        min_coverage: float = 0.95,
        max_forward_fill: int = 3,
        min_observations: int = 100,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return clean prices and a pre-fill, per-symbol quality report.

        Coverage and complete-row filtering apply to the entire supplied frame.
        Historical simulations must therefore perform universe eligibility on
        formation-only data rather than using a future-inclusive cleaned frame.
        """
        if not isinstance(prices, pd.DataFrame):
            raise TypeError("prices must be a pandas DataFrame.")
        if prices.empty or prices.shape[1] == 0:
            raise DataQualityError("Price frame is empty.")
        if isinstance(min_coverage, bool) or not isinstance(
            min_coverage, (int, float)
        ):
            raise ValueError("min_coverage must be numeric.")
        if not np.isfinite(min_coverage) or not 0 < min_coverage <= 1:
            raise ValueError("min_coverage must be in (0, 1].")
        if type(max_forward_fill) is not int or max_forward_fill < 0:
            raise ValueError("max_forward_fill must be a non-negative integer.")
        if type(min_observations) is not int or min_observations < 1:
            raise ValueError("min_observations must be a positive integer.")
        if prices.columns.duplicated().any():
            raise DataQualityError("Price frame contains duplicate symbol columns.")

        frame = prices.copy(deep=True)
        try:
            converted_index = pd.to_datetime(frame.index, utc=True, errors="coerce")
        except (TypeError, ValueError, OverflowError) as exc:
            raise DataQualityError(f"Price index cannot be converted to dates: {exc}") from exc
        if converted_index.isna().any():
            raise DataQualityError("Price index contains invalid dates.")
        frame.index = pd.DatetimeIndex(converted_index).tz_convert(None)
        frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()

        numeric = frame.apply(pd.to_numeric, errors="coerce")
        finite = pd.DataFrame(
            np.isfinite(numeric.to_numpy(dtype=float)),
            index=numeric.index,
            columns=numeric.columns,
        )
        non_positive = numeric.le(0) & numeric.notna() & finite
        valid = numeric.where(finite).mask(non_positive)
        coverage = valid.notna().mean()
        retained = coverage.ge(min_coverage)

        if max_forward_fill:
            filled = valid.loc[:, retained].ffill(limit=max_forward_fill)
        else:
            filled = valid.loc[:, retained].copy()
        forward_filled = filled.notna().sum() - valid.loc[:, retained].notna().sum()

        report = pd.DataFrame(
            {
                "total_observations": len(valid),
                "valid_observations": valid.notna().sum(),
                "missing_or_invalid": valid.isna().sum(),
                "non_positive": non_positive.sum(),
                "coverage": coverage,
                "stale_fraction": valid.pct_change(fill_method=None).eq(0).mean(),
                "first_valid": valid.apply(lambda series: series.first_valid_index()),
                "last_valid": valid.apply(lambda series: series.last_valid_index()),
                "forward_filled": forward_filled.reindex(valid.columns, fill_value=0),
                "retained": retained,
            }
        )
        report.index.name = "symbol"

        dropped = report.index[~report["retained"]].tolist()
        if dropped:
            LOGGER.warning("Dropping low-coverage symbols: %s", dropped)
        if retained.sum() < 2:
            raise DataQualityError(
                "Insufficient clean data: fewer than two symbols meet coverage."
            )

        clean = filled.dropna(how="any")
        if len(clean) < min_observations:
            raise DataQualityError(
                "Insufficient clean data: "
                f"{len(clean)} complete rows; require {min_observations}."
            )
        clean = clean.astype(float)
        # Keep execution provenance separate from the numeric valuation marks.
        # A False cell is a limited forward fill: it may mark an existing
        # holding, but must never authorize a new execution.  DataFrame attrs
        # propagate to a selected Series in supported pandas versions, while
        # callers can also pass this mask explicitly to the backtester.
        observed_mask = valid.loc[clean.index, clean.columns].notna().astype(bool)
        observed_mask.index = clean.index
        observed_mask.columns = clean.columns
        clean.attrs[OBSERVED_PRICE_MASK_ATTR] = observed_mask
        clean.attrs["valuation_policy"] = (
            "limited_forward_fill_for_valuation_only; execution_requires_observed"
        )
        return clean, report.sort_index()


def make_synthetic_universe(
    n_days: int = 1400,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Create two cointegrated pairs plus unrelated assets for offline tests."""
    if type(n_days) is not int or n_days < 300:
        raise ValueError("n_days must be an integer of at least 300.")
    if type(seed) is not int:
        raise ValueError("seed must be a non-boolean integer.")

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_days, name="Date")
    market = np.cumsum(rng.normal(0.0002, 0.009, n_days))

    def ar1(phi: float, sigma: float) -> np.ndarray:
        values = np.zeros(n_days)
        shocks = rng.normal(0.0, sigma, n_days)
        for index in range(1, n_days):
            values[index] = phi * values[index - 1] + shocks[index]
        return values

    tech_common = 4.5 + market + np.cumsum(rng.normal(0.0, 0.003, n_days))
    bank_common = 4.0 + 0.9 * market + np.cumsum(
        rng.normal(0.0, 0.0035, n_days)
    )
    log_prices = {
        # Each A/B pair shares a non-stationary common trend and differs by
        # stationary AR(1) noise, making its cointegration deliberate.
        "TECH_A": tech_common + ar1(0.90, 0.018),
        "TECH_B": 0.12 + 1.04 * tech_common + ar1(0.86, 0.015),
        "BANK_A": bank_common + ar1(0.91, 0.017),
        "BANK_B": -0.08 + 0.96 * bank_common + ar1(0.88, 0.016),
        # Independent random-walk components make these comparison assets
        # unsuitable as intentionally cointegrated partners.
        "TECH_C": 4.2
        + 0.7 * market
        + np.cumsum(rng.normal(0.0, 0.008, n_days)),
        "BANK_C": 3.8
        + 0.5 * market
        + np.cumsum(rng.normal(0.0, 0.009, n_days)),
    }
    prices = pd.DataFrame(
        {symbol: np.exp(values) for symbol, values in log_prices.items()},
        index=dates,
    )
    universe = {
        "Technology": ["TECH_A", "TECH_B", "TECH_C"],
        "Financials": ["BANK_A", "BANK_B", "BANK_C"],
    }
    return prices, universe
