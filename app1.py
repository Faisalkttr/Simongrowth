import numpy as np
import pandas as pd
import yfinance as yf


class SovereignExpectationsEngine:
    """
    Sovereign Expectations Engine v2.1 (hardened)

    A completely decoupled, standalone analytical entity that processes
    implied market expectations and priced-for-perfection risk profiles.
    Operates independently or alongside Macro and v1.2 Engine architectures.

    Changes from v2.0:
      - Fixed a tz_localize crash in the hard revenue fallback path.
      - Revenue-row lookup now checks quarterly AND annual statements
        instead of abandoning annual data whenever quarterly_financials
        is non-empty but lacks a recognizable revenue row.
      - Forward growth from analyst estimates is now clipped to the same
        [-50%, +150%] band as the historical fallback, so a bad/garbled
        analyst number can no longer blow out downstream math unchecked.
      - "+1y" row selection is stricter, avoiding accidental matches on
        "0y" (current year) or "-1y" (trailing) rows.
      - Recent-momentum calculation is now cadence-aware (quarterly vs.
        annual data get different, appropriately-sized windows) instead
        of blindly taking the last 4 observations regardless of cadence.
      - Revenue_TTM is forward-filled only; leading (pre-first-known-value)
        rows are dropped instead of back-filled, so P/S is never computed
        against a fabricated revenue figure.
      - Market cap is no longer flattened into a constant across 5 years
        of history when shares_outstanding is unavailable. Only the most
        recent day gets a market-cap value in that case, and the engine
        flags reduced confidence in the historical valuation anchors
        rather than silently distorting the P/S history.
      - Degenerate inputs (no revenue, no market cap, non-positive revenue)
        now raise clear, specific errors instead of crashing deep inside
        numpy/pandas or silently producing garbage.
    """

    EXPECTATIONS_STATUS_LEGEND = {
        "✅ Forward Expectations Manageable": "Forward growth appears sufficient relative to the valuation normalization burden.",
        "🟡 Execution-Dependent Premium": "The premium may be justified, but future returns depend on continued growth delivery.",
        "🟠 High Execution Burden": "The market appears to require substantial forward growth to justify current valuation.",
        "🔴 Priced-for-Perfection Risk": "Current valuation requires aggressive future growth assumptions and leaves limited room for disappointment."
    }

    GROWTH_CLIP_BOUNDS = (-0.50, 1.50)
    MIN_VALID_PS_OBSERVATIONS = 30  # below this, valuation anchors are flagged low-confidence

    def __init__(self, ticker: str, is_core: bool = False):
        if not ticker or not isinstance(ticker, str):
            raise ValueError("A non-empty string ticker is required.")
        self.ticker = ticker.upper().strip()
        self.is_core = is_core
        self.stock = yf.Ticker(self.ticker)

        # Output Metrics Container
        self.metrics = {}

    # ------------------------------------------------------------------
    # Parsing / extraction helpers
    # ------------------------------------------------------------------

    def _parse_estimate_number(self, value) -> float:
        """Converts raw or string analyst estimates (e.g., '12.5B') into pure floats."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return np.nan
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        if isinstance(value, str):
            text = value.strip().replace(",", "").upper()
            if text in ("", "N/A", "NA", "NAN", "-"):
                return np.nan
            multiplier = 1.0
            if text.endswith("T"):
                multiplier = 1e12
                text = text[:-1]
            elif text.endswith("B"):
                multiplier = 1e9
                text = text[:-1]
            elif text.endswith("M"):
                multiplier = 1e6
                text = text[:-1]
            elif text.endswith("K"):
                multiplier = 1e3
                text = text[:-1]
            try:
                return float(text) * multiplier
            except Exception:
                return np.nan
        return np.nan

    def _safe_get_df(self, attr_name: str) -> pd.DataFrame:
        """Safely extracts DataFrames from volatile yfinance endpoints."""
        try:
            attr = getattr(self.stock, attr_name)
            result = attr() if callable(attr) else attr
            if isinstance(result, pd.DataFrame):
                return result
        except Exception:
            pass
        return pd.DataFrame()

    def _align_index_tz(self, series: pd.Series, reference_tz) -> pd.Series:
        """
        Aligns a series' DatetimeIndex to a reference timezone (or naive),
        regardless of whether the series' own index started tz-aware or not.
        This is the single choke point for all tz handling in the engine,
        so no downstream code needs to guess about tz-awareness.
        """
        idx = pd.to_datetime(series.index)
        if reference_tz is not None:
            if idx.tz is None:
                idx = idx.tz_localize(reference_tz)
            else:
                idx = idx.tz_convert(reference_tz)
        else:
            if idx.tz is not None:
                idx = idx.tz_localize(None)
        series = series.copy()
        series.index = idx
        return series

    def _select_forward_revenue_row(self, index_labels) -> str:
        """
        Picks the analyst-estimate row that represents the NEXT fiscal year
        of revenue, avoiding accidental matches on the current year ("0y")
        or trailing year ("-1y") rows that a loose substring match could
        otherwise pick up.
        """
        labels = [str(x) for x in index_labels]

        exact_tags = ["+1y", "next year", "nextyear"]
        for tag in exact_tags:
            matches = [lbl for lbl in labels if tag in lbl.lower()]
            if matches:
                return matches[0]

        # "1y" alone is riskier (could theoretically appear elsewhere), so
        # it's tried after the more explicit tags but before the loose pass.
        matches = [lbl for lbl in labels if lbl.lower().replace(" ", "") in ("1y", "+1y")]
        if matches:
            return matches[0]

        # Loose fallback: contains "y" but is NOT the current-year ("0y")
        # or trailing-year ("-1y") row.
        loose_matches = [
            lbl for lbl in labels
            if "y" in lbl.lower()
            and "0y" not in lbl.lower().replace(" ", "")
            and "-1y" not in lbl.lower().replace(" ", "")
        ]
        if loose_matches:
            return loose_matches[0]

        return None

    # ------------------------------------------------------------------
    # Data hydration
    # ------------------------------------------------------------------

    def hydrate_standalone_data(self) -> pd.DataFrame:
        """
        Fallback internal data engine. Reconstructs necessary v1.2 historical
        pipelines (Revenue_TTM, Market_Cap, PS_Ratio) if the engine is
        executed completely solo.

        df_data.attrs["revenue_cadence"] is set to one of:
          "quarterly"     - TTM built by rolling 4 quarters of reported revenue
          "annual"        - annual statement revenue used directly
          "single_point"  - only a single current revenue figure was available
        This cadence flag lets downstream momentum calculations use an
        appropriately sized window instead of assuming quarterly spacing.

        df_data.attrs["market_cap_confidence"] is set to "high" when
        sharesOutstanding was available to reconstruct full history, or
        "low" when only a single current-day market cap value exists.
        """
        hist = self.stock.history(period="5y")
        if hist.empty:
            raise ValueError(f"{self.ticker}: Failed to download historical price data.")

        financials = self._safe_get_df("financials")
        quarterly_fin = self._safe_get_df("quarterly_financials")

        rev_labels = ["Total Revenue", "TotalRevenue", "Revenue"]
        quarterly_rev_row = next(
            (r for r in rev_labels if not quarterly_fin.empty and r in quarterly_fin.index), None
        )
        annual_rev_row = next(
            (r for r in rev_labels if not financials.empty and r in financials.index), None
        )

        cadence = None
        if quarterly_rev_row is not None:
            rev_series = (
                quarterly_fin.loc[quarterly_rev_row]
                .dropna()
                .sort_index()
                .rolling(window=4)
                .sum()
                .dropna()
            )
            if not rev_series.empty:
                cadence = "quarterly"

        if cadence is None and annual_rev_row is not None:
            rev_series = financials.loc[annual_rev_row].dropna().sort_index()
            if not rev_series.empty:
                cadence = "annual"

        if cadence is None:
            # Total hard fallback if financial tables fail completely.
            info = self.stock.info or {}
            fallback_value = info.get("totalRevenue")
            if fallback_value is None or fallback_value <= 0:
                raise ValueError(
                    f"{self.ticker}: No usable revenue data from financials, "
                    f"quarterly_financials, or info['totalRevenue']."
                )
            # Build with a tz-naive timestamp; _align_index_tz will reconcile
            # it against hist's own tz below, whatever that is.
            naive_ts = pd.Timestamp(hist.index.max()).tz_localize(None)
            rev_series = pd.Series([float(fallback_value)], index=[naive_ts])
            cadence = "single_point"

        rev_series = self._align_index_tz(rev_series, hist.index.tz)

        df_data = pd.DataFrame(index=hist.index)
        df_data["Close"] = hist["Close"]

        info = self.stock.info or {}
        shares_outstanding = info.get("sharesOutstanding")

        if shares_outstanding:
            df_data["Market_Cap"] = df_data["Close"] * shares_outstanding
            market_cap_confidence = "high"
        else:
            market_cap_now = info.get("marketCap")
            if not market_cap_now or market_cap_now <= 0:
                raise ValueError(
                    f"{self.ticker}: Unable to determine shares outstanding or "
                    f"market cap from info; cannot compute P/S history."
                )
            # Do NOT flatten the whole 5y history to today's market cap -
            # that fabricates a P/S history that never existed. Only the
            # most recent observation is trustworthy here.
            df_data["Market_Cap"] = np.nan
            df_data.loc[df_data.index[-1], "Market_Cap"] = float(market_cap_now)
            market_cap_confidence = "low"

        # Forward-fill revenue steps daily, but do NOT back-fill: rows
        # before the first known revenue print have no real TTM revenue
        # and should be dropped rather than fabricated.
        df_data = df_data.join(rev_series.to_frame(name="Revenue_TTM"), how="left")
        df_data["Revenue_TTM"] = df_data["Revenue_TTM"].ffill()
        df_data = df_data.dropna(subset=["Revenue_TTM"])

        if df_data.empty:
            raise ValueError(
                f"{self.ticker}: No overlapping price/revenue history remained "
                f"after alignment; cannot compute P/S ratios."
            )

        df_data["PS_Ratio"] = df_data["Market_Cap"] / df_data["Revenue_TTM"]

        df_data.attrs["revenue_cadence"] = cadence
        df_data.attrs["market_cap_confidence"] = market_cap_confidence
        return df_data

    # ------------------------------------------------------------------
    # Forward revenue estimation
    # ------------------------------------------------------------------

    def fetch_analyst_revenue_estimate(self) -> dict:
        """Priority 1: Parses institutional analyst forward expectations."""
        candidate_names = ["get_revenue_estimate", "revenue_estimate"]
        for name in candidate_names:
            df = self._safe_get_df(name)
            if df.empty:
                continue

            df_clean = df.copy()
            df_clean.columns = [str(c).lower() for c in df_clean.columns]

            avg_col = next((c for c in df_clean.columns if "avg" in c or "average" in c), None)
            if not avg_col:
                continue

            selected_idx = self._select_forward_revenue_row(df_clean.index)
            if selected_idx is None:
                continue

            val = self._parse_estimate_number(df_clean.loc[selected_idx, avg_col])
            if not np.isnan(val) and val > 0:
                return {"forward_revenue": val, "source": f"Yahoo analyst estimate row: {selected_idx}"}

        return {"forward_revenue": np.nan, "source": "No automated analyst revenue estimate available"}

    def estimate_historical_revenue_growth(self, df_data: pd.DataFrame) -> dict:
        """Priority 2-4: Multi-tier fallback calculations from realized data trends."""
        revenue = df_data["Revenue_TTM"].replace([np.inf, -np.inf], np.nan).dropna()
        if revenue.empty:
            return {"growth_estimate": np.nan, "source": "No historical data footprint available"}

        revenue_unique = revenue[~revenue.index.duplicated(keep="last")]
        revenue_unique = revenue_unique[revenue_unique != revenue_unique.shift(1)].dropna()

        if len(revenue_unique) < 2:
            return {"growth_estimate": np.nan, "source": "Insufficient historical distinct observations"}

        cadence = df_data.attrs.get("revenue_cadence", "unknown")

        latest_date = revenue_unique.index.max()
        current_revenue = revenue_unique.iloc[-1]
        growth_inputs = []
        source_parts = []

        # 1Y TTM Growth
        one_yr = latest_date - pd.DateOffset(years=1)
        rev_1y = revenue_unique[revenue_unique.index <= one_yr]
        if not rev_1y.empty and rev_1y.iloc[-1] > 0:
            g_1y = (current_revenue / rev_1y.iloc[-1]) - 1
            growth_inputs.append((g_1y, 0.50))
            source_parts.append(f"1Y Trailing Realized: {g_1y:.1%}")

        # 2Y CAGR
        two_yr = latest_date - pd.DateOffset(years=2)
        rev_2y = revenue_unique[revenue_unique.index <= two_yr]
        if not rev_2y.empty and rev_2y.iloc[-1] > 0:
            g_2y = (current_revenue / rev_2y.iloc[-1]) ** 0.5 - 1
            growth_inputs.append((g_2y, 0.30))
            source_parts.append(f"2Y Realized CAGR: {g_2y:.1%}")

        # Recent Momentum - cadence-aware window.
        # Quarterly TTM data: last 4 steps approximate ~1 year of momentum.
        # Annual data: 4 steps would span ~4 years, which is not "recent"
        # momentum at all, so use a much shorter window (and only if
        # there's enough data to make it meaningful).
        if cadence == "quarterly":
            window = 4
        elif cadence == "annual":
            window = 2
        else:
            window = 4  # unknown cadence: keep prior default behavior

        if len(revenue_unique) >= window + 1 and window >= 2:
            g_mom = revenue_unique.pct_change().tail(window).median()
            if not np.isnan(g_mom):
                growth_inputs.append((g_mom, 0.20))
                source_parts.append(f"Recent Momentum ({cadence}, window={window}): {g_mom:.1%}")

        if not growth_inputs:
            return {"growth_estimate": np.nan, "source": "No historical pipeline inputs valid"}

        blended = sum(g * w for g, w in growth_inputs) / sum(w for _, w in growth_inputs)
        clipped = float(np.clip(blended, *self.GROWTH_CLIP_BOUNDS))
        return {"growth_estimate": clipped, "source": " | ".join(source_parts)}

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, df_data: pd.DataFrame = None) -> dict:
        """
        Runs the complete forward calculations.
        If df_data is passed, it extracts metrics contextually.
        If df_data is None, it triggers standalone data hydration.
        """
        if df_data is None:
            df_data = self.hydrate_standalone_data()

        if "Revenue_TTM" not in df_data.columns or "Market_Cap" not in df_data.columns:
            raise ValueError(
                f"{self.ticker}: df_data must contain 'Revenue_TTM' and 'Market_Cap' columns."
            )

        market_cap_confidence = df_data.attrs.get("market_cap_confidence", "unknown")

        ps_series = df_data["PS_Ratio"].replace([np.inf, -np.inf], np.nan).dropna() \
            if "PS_Ratio" in df_data.columns else \
            (df_data["Market_Cap"] / df_data["Revenue_TTM"]).replace([np.inf, -np.inf], np.nan).dropna()

        current_revenue = float(df_data["Revenue_TTM"].iloc[-1])
        current_market_cap = float(df_data["Market_Cap"].iloc[-1])

        if current_revenue <= 0:
            raise ValueError(
                f"{self.ticker}: Current TTM revenue is non-positive ({current_revenue}); "
                f"valuation-burden math is undefined for this ticker."
            )
        if current_market_cap <= 0 or np.isnan(current_market_cap):
            raise ValueError(
                f"{self.ticker}: Current market cap is missing or non-positive; "
                f"cannot compute current P/S."
            )

        current_ps = float(current_market_cap / current_revenue)

        # Baseline Multiple Extraction
        anchor_confidence = "high"
        if ps_series.empty:
            median_ps = np.nan
            p75_ps = np.nan
            p90_ps = np.nan
            current_percentile = np.nan
            anchor_confidence = "none"
        else:
            median_ps = float(ps_series.median())
            p75_ps = float(ps_series.quantile(0.75))
            p90_ps = float(ps_series.quantile(0.90))
            current_percentile = float((ps_series <= current_ps).mean() * 100)
            if len(ps_series) < self.MIN_VALID_PS_OBSERVATIONS or market_cap_confidence == "low":
                anchor_confidence = "low"

        # Target Assignment Anchor Logic
        target_ps = p75_ps if self.is_core else median_ps
        target_label = "75th Percentile Scarcity Anchor" if self.is_core else "Historical Median Tactical Anchor"

        # Automated Cascade Data Pulls
        analyst = self.fetch_analyst_revenue_estimate()
        if not np.isnan(analyst["forward_revenue"]):
            forward_revenue = analyst["forward_revenue"]
            # Clip analyst-derived growth to the same bounds as the
            # historical fallback so a garbled/outlier estimate can't
            # blow out downstream math unchecked.
            forward_growth = float(
                np.clip((forward_revenue / current_revenue) - 1, *self.GROWTH_CLIP_BOUNDS)
            )
            # Recompute forward_revenue from the clipped growth so the two
            # stay internally consistent even if clipping engaged.
            forward_revenue = current_revenue * (1 + forward_growth)
            forward_source = f"Analyst Pipeline -> {analyst['source']}"
            forward_confidence = "High / Analyst Estimate Available"
        else:
            hist_growth = self.estimate_historical_revenue_growth(df_data)
            forward_growth = hist_growth["growth_estimate"]
            if np.isnan(forward_growth):
                forward_growth = 0.0
                forward_source = "Neutralized Baseline Fallback (Zero Visibility)"
                forward_confidence = "Low / Forward Data Unavailable"
            else:
                forward_growth = float(forward_growth)
                forward_source = f"Historical Pipeline Fallback -> {hist_growth['source']}"
                forward_confidence = "Medium / Historical Trend Fallback"

            forward_revenue = current_revenue * (1 + forward_growth)

        # Mathematical Valuation Burden Extractions
        forward_ps = float(current_market_cap / forward_revenue) if forward_revenue > 0 else np.nan
        required_revenue = float(current_market_cap / target_ps) if target_ps and target_ps > 0 else np.nan
        required_growth = float((required_revenue / current_revenue) - 1) if not np.isnan(required_revenue) else np.nan
        growth_gap = float(required_growth - forward_growth) if not np.isnan(required_growth) else np.nan

        # Years to Normalize Calculations
        if np.isnan(required_revenue) or required_revenue <= 0:
            years_to_normalise = np.nan
        elif required_revenue <= current_revenue:
            years_to_normalise = 0.0
        elif forward_growth <= 0:
            years_to_normalise = np.inf
        else:
            try:
                years_to_normalise = float(np.log(required_revenue / current_revenue) / np.log(1 + forward_growth))
            except Exception:
                years_to_normalise = np.nan

        # Continuous Score Compilation Engine (0-100 max points allocation)
        score = 0.0
        if not np.isnan(current_percentile):
            score += min(max((current_percentile - 50) / 50, 0), 1) * 25
        if not np.isnan(required_growth):
            score += min(max(required_growth, 0) / 1.00, 1) * 25
        if not np.isnan(growth_gap):
            score += min(max(growth_gap, 0) / 0.50, 1) * 25
        if not np.isnan(forward_ps) and target_ps and target_ps > 0:
            score += min(max(((forward_ps / target_ps) - 1), 0) / 1.00, 1) * 25

        # Posture Status Formatting
        if score >= 75:
            status = "🔴 Priced-for-Perfection Risk"
        elif score >= 55:
            status = "🟠 High Execution Burden"
        elif score >= 35:
            status = "🟡 Execution-Dependent Premium"
        else:
            status = "✅ Forward Expectations Manageable"

        self.metrics = {
            "Current Revenue TTM": current_revenue,
            "Current Market Cap": current_market_cap,
            "Current P/S": current_ps,
            "Historical Median P/S": median_ps,
            "Historical 75th Percentile P/S": p75_ps,
            "Historical 90th Percentile P/S": p90_ps,
            "Forward Revenue Estimate": forward_revenue,
            "Forward Revenue Growth Estimate": forward_growth,
            "Forward P/S": forward_ps,
            "Required Revenue to Normalise Valuation": required_revenue,
            "Required Revenue Growth": required_growth,
            "Growth Gap": growth_gap,
            "Years to Normalise Multiple": years_to_normalise,
            "Expectations Burden Score": score,
            "Expectations Classification": status,
            "Forward Confidence": forward_confidence,
            "Forward Source Pipeline": forward_source,
            "Target Multiple Value": target_ps,
            "Target Multiple Label": target_label,
            "Valuation Anchor Confidence": anchor_confidence,
            "Valuation Anchor Observation Count": int(len(ps_series)),
            "Revenue Data Cadence": df_data.attrs.get("revenue_cadence", "unknown"),
        }
        return self.metrics
