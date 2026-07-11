import numpy as np
import pandas as pd
import yfinance as yf

class SovereignExpectationsEngine:
    """
    Sovereign Expectations Engine v2.0
    
    A completely decoupled, standalone analytical entity that processes 
    implied market expectations and priced-for-perfection risk profiles.
    Operates independently or alongside Macro and v1.2 Engine architectures.
    """
    
    EXPECTATIONS_STATUS_LEGEND = {
        "✅ Forward Expectations Manageable": "Forward growth appears sufficient relative to the valuation normalization burden.",
        "🟡 Execution-Dependent Premium": "The premium may be justified, but future returns depend on continued growth delivery.",
        "🟠 High Execution Burden": "The market appears to require substantial forward growth to justify current valuation.",
        "🔴 Priced-for-Perfection Risk": "Current valuation requires aggressive future growth assumptions and leaves limited room for disappointment."
    }

    def __init__(self, ticker: str, is_core: bool = False):
        self.ticker = ticker.upper()
        self.is_core = is_core
        self.stock = yf.Ticker(self.ticker)
        
        # Output Metrics Container
        self.metrics = {}

    def _parse_estimate_number(self, value) -> float:
        """Converts raw or string analyst estimates (e.g., '12.5B') into pure floats."""
        if value is None or pd.isna(value):
            return np.nan
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        if isinstance(value, str):
            text = value.strip().replace(",", "").upper()
            multiplier = 1.0
            if text.endswith("T"): multiplier = 1e12; text = text[:-1]
            elif text.endswith("B"): multiplier = 1e9; text = text[:-1]
            elif text.endswith("M"): multiplier = 1e6; text = text[:-1]
            elif text.endswith("K"): multiplier = 1e3; text = text[:-1]
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

    def hydrate_standalone_data(self) -> pd.DataFrame:
        """
        Fallback internal data engine. Reconstructs necessary v1.2 historical pipelines 
        (Revenue_TTM, Market_Cap, PS_Ratio) if the engine is executed completely solo.
        """
        # Fetch pricing history
        hist = self.stock.history(period="5y")
        if hist.empty:
            raise ValueError(self.ticker, "Failed to download historical price data.")
            
        # Fetch financials
        financials = self._safe_get_df("financials")
        quarterly_fin = self._safe_get_df("quarterly_financials")
        
        # Combine annual and quarterly statements to find revenue lines
        df_fin = quarterly_fin if not quarterly_fin.empty else financials
        rev_labels = ["Total Revenue", "TotalRevenue", "Revenue"]
        rev_row = next((r for r in rev_labels if r in df_fin.index), None)
        
        if rev_row is not None:
            # Reconstruct trailing annual run-rate values
            rev_series = df_fin.loc[rev_row].dropna().sort_index()
            # If quarterly data is used, roll it to TTM
            if not quarterly_fin.empty and rev_row in quarterly_fin.index:
                rev_series = quarterly_fin.loc[rev_row].dropna().sort_index().rolling(window=4).sum()
        else:
            # Total hard fallback if financial tables fail completely
            info = self.stock.info or {}
            rev_series = pd.Series([info.get("totalRevenue", 1.0)], index=[hist.index.max()])

        # Reindex to match daily price data structure
        df_data = pd.DataFrame(index=hist.index)
        df_data["Close"] = hist["Close"]
        
        info = self.stock.info or {}
        shares_outstanding = info.get("sharesOutstanding", None)
        
        if shares_outstanding:
            df_data["Market_Cap"] = df_data["Close"] * shares_outstanding
        else:
            df_data["Market_Cap"] = info.get("marketCap", 1.0)

        # Forward fill clean TTM revenue steps daily
        rev_series.index = pd.to_datetime(rev_series.index).tz_localize(hist.index.tz)
        df_data = df_data.join(rev_series.to_frame(name="Revenue_TTM"), how="left")
        df_data["Revenue_TTM"] = df_data["Revenue_TTM"].ffill().bfill()
        
        # Build P/S Ratio arrays
        df_data["PS_Ratio"] = df_data["Market_Cap"] / df_data["Revenue_TTM"]
        return df_data

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

            preferred_rows = [idx for idx in df_clean.index if any(x in str(idx).lower() for x in ["+1y", "next year", "1y"])]
            if not preferred_rows:
                preferred_rows = [idx for idx in df_clean.index if "y" in str(idx).lower()]
                
            if preferred_rows:
                selected_idx = preferred_rows[0]
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

        # Recent Momentum
        if len(revenue_unique) >= 4:
            g_mom = revenue_unique.pct_change().tail(4).median()
            if not np.isnan(g_mom):
                growth_inputs.append((g_mom, 0.20))
                source_parts.append(f"Recent Momentum: {g_mom:.1%}")

        if not growth_inputs:
            return {"growth_estimate": np.nan, "source": "No historical pipeline inputs valid"}

        blended = sum(g * w for g, w in growth_inputs) / sum(w for _, w in growth_inputs)
        return {"growth_estimate": float(np.clip(blended, -0.50, 1.50)), "source": " | ".join(source_parts)}

    def execute(self, df_data: pd.DataFrame = None) -> dict:
        """
        Runs the complete forward calculations.
        If df_data is passed, it extracts metrics contextually. 
        If df_data is None, it triggers standalone data hydration.
        """
        if df_data is None:
            df_data = self.hydrate_standalone_data()

        ps_series = df_data["PS_Ratio"].replace([np.inf, -np.inf], np.nan).dropna()
        
        current_revenue = float(df_data["Revenue_TTM"].iloc[-1])
        current_market_cap = float(df_data["Market_Cap"].iloc[-1])
        current_ps = float(current_market_cap / current_revenue) if current_revenue > 0 else np.nan

        # Baseline Multiple Extraction
        median_ps = float(ps_series.median())
        p75_ps = float(ps_series.quantile(0.75))
        p90_ps = float(ps_series.quantile(0.90))
        current_percentile = float((ps_series <= current_ps).mean() * 100) if not ps_series.empty else np.nan

        # Target Assignment Anchor Logic
        target_ps = p75_ps if self.is_core else median_ps
        target_label = "75th Percentile Scarcity Anchor" if self.is_core else "Historical Median Tactical Anchor"

        # Automated Cascade Data Pulls
        analyst = self.fetch_analyst_revenue_estimate()
        if not np.isnan(analyst["forward_revenue"]):
            forward_revenue = analyst["forward_revenue"]
            forward_growth = (forward_revenue / current_revenue) - 1
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
        required_revenue = float(current_market_cap / target_ps) if target_ps > 0 else np.nan
        required_growth = float((required_revenue / current_revenue) - 1) if current_revenue > 0 else np.nan
        growth_gap = float(required_growth - forward_growth) if not np.isnan(required_growth) else np.nan

        # Years to Normalize Calculations
        if current_revenue <= 0 or required_revenue <= 0:
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
        if not np.isnan(forward_ps) and target_ps > 0:
            score += min(max(((forward_ps / target_ps) - 1), 0) / 1.00, 1) * 25

        # Posture Status Formatting
        if score >= 75: status = "🔴 Priced-for-Perfection Risk"
        elif score >= 55: status = "🟠 High Execution Burden"
        elif score >= 35: status = "🟡 Execution-Dependent Premium"
        else: status = "✅ Forward Expectations Manageable"

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
            "Target Multiple Label": target_label
        }
        return self.metrics
