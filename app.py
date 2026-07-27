import os
from typing import Optional, Dict, Any, List

import pandas as pd
import streamlit as st
import yfinance as yf


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Fundamental AI",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Fundamental AI")
st.caption(
    "Fundamental stock analysis for Indian and US markets. "
    "Educational analysis only — not investment advice."
)


# ============================================================
# HELPERS
# ============================================================

INDIA_SUFFIXES = (".NS", ".BO")


def normalize_ticker(ticker: str, market: str) -> str:
    """
    Converts common user input into a Yahoo Finance ticker.
    """
    ticker = ticker.strip().upper()

    if not ticker:
        return ""

    if market == "India":
        # Allow users to enter TCS or TCS.NS
        if ticker.endswith(INDIA_SUFFIXES):
            return ticker

        return f"{ticker}.NS"

    # US ticker
    return ticker


def safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None

        result = float(value)

        if pd.isna(result):
            return None

        return result
    except Exception:
        return None


def format_number(value, decimals: int = 2) -> str:
    value = safe_float(value)

    if value is None:
        return "N/A"

    return f"{value:,.{decimals}f}"


def format_currency(value, currency: str = "") -> str:
    value = safe_float(value)

    if value is None:
        return "N/A"

    if abs(value) >= 1_000_000_000_000:
        text = f"{value / 1_000_000_000_000:,.2f}T"
    elif abs(value) >= 1_000_000_000:
        text = f"{value / 1_000_000_000:,.2f}B"
    elif abs(value) >= 1_000_000:
        text = f"{value / 1_000_000:,.2f}M"
    else:
        text = f"{value:,.2f}"

    return f"{currency} {text}".strip()


def get_statement_value(
    statement: pd.DataFrame,
    row_names: List[str],
    column=None,
):
    """
    Safely gets a financial statement row.
    Handles missing rows and missing columns.
    """
    if statement is None or statement.empty:
        return None

    for row_name in row_names:
        try:
            if row_name in statement.index:
                row = statement.loc[row_name]

                if column is not None:
                    if column in row.index:
                        return safe_float(row[column])

                if len(row) > 0:
                    return safe_float(row.iloc[0])
        except Exception:
            pass

    return None


def get_year_columns(statement: pd.DataFrame):
    if statement is None or statement.empty:
        return []

    columns = list(statement.columns)

    # Newest first
    try:
        columns = sorted(columns, reverse=True)
    except Exception:
        pass

    return columns[:5]


def build_five_year_table(statement: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts a simple 5-year table from a yfinance statement.
    """
    if statement is None or statement.empty:
        return pd.DataFrame()

    columns = get_year_columns(statement)

    if not columns:
        return pd.DataFrame()

    metric_map = {
        "Revenue": ["Total Revenue", "Operating Revenue"],
        "Gross Profit": ["Gross Profit"],
        "Operating Income": ["Operating Income"],
        "EBIT": ["EBIT"],
        "Net Income": [
            "Net Income",
            "Net Income Common Stockholders",
            "Net Income Including Noncontrolling Interests",
        ],
        "EPS": ["Diluted EPS", "Basic EPS"],
    }

    data = {}

    for metric, rows in metric_map.items():
        values = []

        for col in columns:
            value = get_statement_value(statement, rows, col)
            values.append(value)

        data[metric] = values

    labels = []

    for col in columns:
        try:
            if hasattr(col, "year"):
                labels.append(str(col.year))
            else:
                labels.append(str(col)[:10])
        except Exception:
            labels.append(str(col))

    df = pd.DataFrame(data, index=labels).T
    df.index.name = "Metric"

    return df


def get_stock_data(ticker_symbol: str) -> Dict[str, Any]:
    """
    Fetch all required data.
    Important: returns only normal Python / pandas objects.
    This avoids cache serialization problems.
    """
    ticker = yf.Ticker(ticker_symbol)

    fast_info = {}
    info = {}

    # fast_info
    try:
        raw_fast = ticker.fast_info

        for key in [
            "currency",
            "lastPrice",
            "marketCap",
            "previousClose",
            "dayHigh",
            "dayLow",
            "yearHigh",
            "yearLow",
            "yearChange",
            "fiftyDayAverage",
            "twoHundredDayAverage",
        ]:
            try:
                fast_info[key] = safe_float(raw_fast[key])
            except Exception:
                try:
                    fast_info[key] = raw_fast.get(key)
                except Exception:
                    fast_info[key] = None

    except Exception:
        fast_info = {}

    # info
    try:
        raw_info = ticker.info

        if isinstance(raw_info, dict):
            # Only copy normal serializable values
            for key in [
                "longName",
                "shortName",
                "sector",
                "industry",
                "country",
                "currency",
                "exchange",
                "quoteType",
                "trailingPE",
                "forwardPE",
                "priceToBook",
                "returnOnEquity",
                "returnOnAssets",
                "debtToEquity",
                "currentRatio",
                "quickRatio",
                "profitMargins",
                "operatingMargins",
                "grossMargins",
                "revenueGrowth",
                "earningsGrowth",
                "dividendYield",
                "beta",
                "enterpriseValue",
                "freeCashflow",
                "operatingCashflow",
            ]:
                value = raw_info.get(key)

                if isinstance(value, (str, int, float, bool)) or value is None:
                    info[key] = value

    except Exception:
        info = {}

    # Financial statements
    income_stmt = pd.DataFrame()
    balance_sheet = pd.DataFrame()
    cashflow = pd.DataFrame()

    try:
        income_stmt = ticker.income_stmt
    except Exception:
        pass

    try:
        balance_sheet = ticker.balance_sheet
    except Exception:
        pass

    try:
        cashflow = ticker.cashflow
    except Exception:
        pass

    return {
        "ticker": ticker_symbol,
        "fast_info": fast_info,
        "info": info,
        "income_stmt": income_stmt,
        "balance_sheet": balance_sheet,
        "cashflow": cashflow,
    }


def metric(data: Dict[str, Any], name: str):
    info = data.get("info", {})
    fast = data.get("fast_info", {})

    # Prefer info
    if name in info and info[name] is not None:
        return safe_float(info[name])

    if name in fast and fast[name] is not None:
        return safe_float(fast[name])

    return None


# ============================================================
# FUNDAMENTAL SCORE
# ============================================================

def calculate_score(data: Dict[str, Any]) -> Dict[str, Any]:
    score = 0
    max_score = 0
    positives = []
    negatives = []

    # ROE
    roe = metric(data, "returnOnEquity")

    if roe is not None:
        max_score += 1

        if roe >= 0.15:
            score += 1
            positives.append("ROE is above 15%.")
        elif roe < 0.08:
            negatives.append("ROE is relatively low.")

    # Profit margin
    margin = metric(data, "profitMargins")

    if margin is not None:
        max_score += 1

        if margin >= 0.10:
            score += 1
            positives.append("Profit margin is above 10%.")
        elif margin < 0.05:
            negatives.append("Profit margin is relatively thin.")

    # Debt to equity
    debt_equity = metric(data, "debtToEquity")

    if debt_equity is not None:
        max_score += 1

        if debt_equity <= 50:
            score += 1
            positives.append("Debt-to-equity is relatively controlled.")
        elif debt_equity > 150:
            negatives.append("Debt-to-equity is relatively high.")

    # Current ratio
    current_ratio = metric(data, "currentRatio")

    if current_ratio is not None:
        max_score += 1

        if current_ratio >= 1:
            score += 1
            positives.append("Current ratio is at least 1.")
        elif current_ratio < 0.75:
            negatives.append("Current ratio is relatively weak.")

    # Free cash flow
    fcf = metric(data, "freeCashflow")

    if fcf is not None:
        max_score += 1

        if fcf > 0:
            score += 1
            positives.append("Free cash flow is positive.")
        else:
            negatives.append("Free cash flow is negative.")

    # Revenue growth
    revenue_growth = metric(data, "revenueGrowth")

    if revenue_growth is not None:
        max_score += 1

        if revenue_growth > 0:
            score += 1
            positives.append("Revenue growth is positive.")
        elif revenue_growth < -0.05:
            negatives.append("Revenue growth is negative.")

    # Valuation
    pe = metric(data, "trailingPE")

    if pe is not None:
        max_score += 1

        if 0 < pe <= 25:
            score += 1
            positives.append("Trailing P/E is within a moderate range.")
        elif pe > 50:
            negatives.append("Trailing P/E is elevated.")

    if max_score == 0:
        percentage = None
    else:
        percentage = score / max_score * 100

    if percentage is None:
        rating = "Insufficient data"
    elif percentage >= 80:
        rating = "Strong"
    elif percentage >= 60:
        rating = "Good"
    elif percentage >= 40:
        rating = "Mixed"
    else:
        rating = "Weak"

    return {
        "score": score,
        "max_score": max_score,
        "percentage": percentage,
        "rating": rating,
        "positives": positives,
        "negatives": negatives,
    }


# ============================================================
# RED FLAGS
# ============================================================

def get_red_flags(data: Dict[str, Any]) -> List[str]:
    flags = []

    pe = metric(data, "trailingPE")
    debt_equity = metric(data, "debtToEquity")
    roe = metric(data, "returnOnEquity")
    revenue_growth = metric(data, "revenueGrowth")
    profit_margin = metric(data, "profitMargins")
    fcf = metric(data, "freeCashflow")
    current_ratio = metric(data, "currentRatio")

    if pe is not None and pe > 50:
        flags.append(f"High valuation: trailing P/E is {pe:.1f}.")

    if debt_equity is not None and debt_equity > 150:
        flags.append(f"High leverage: debt-to-equity is {debt_equity:.1f}.")

    if roe is not None and roe < 0.08:
        flags.append(f"Low ROE: approximately {roe * 100:.1f}%.")

    if revenue_growth is not None and revenue_growth < 0:
        flags.append(
            f"Revenue growth is negative: approximately {revenue_growth * 100:.1f}%."
        )

    if profit_margin is not None and profit_margin < 0.05:
        flags.append(
            f"Low profit margin: approximately {profit_margin * 100:.1f}%."
        )

    if fcf is not None and fcf < 0:
        flags.append("Free cash flow is negative.")

    if current_ratio is not None and current_ratio < 0.75:
        flags.append(f"Weak current ratio: approximately {current_ratio:.2f}.")

    if not flags:
        flags.append("No major automated red flags were detected from the available metrics.")

    return flags


# ============================================================
# AI SUMMARY
# ============================================================

def generate_ai_summary(
    data: Dict[str, Any],
    score_data: Dict[str, Any],
) -> Optional[str]:
    """
    Optional OpenAI integration.

    The API key is read from:
    st.secrets["OPENAI_API_KEY"]

    If no key exists, the app simply shows a deterministic
    summary instead of crashing.
    """

    api_key = None

    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        info = data.get("info", {})
        fast = data.get("fast_info", {})

        company = (
            info.get("longName")
            or info.get("shortName")
            or data.get("ticker")
        )

        prompt = f"""
You are a financial analysis assistant.

Write a concise educational fundamental-analysis summary for:

Company: {company}
Ticker: {data.get("ticker")}

Price: {fast.get("lastPrice")}
Market Cap: {fast.get("marketCap")}
Currency: {fast.get("currency")}

Trailing P/E: {info.get("trailingPE")}
Price/Book: {info.get("priceToBook")}
ROE: {info.get("returnOnEquity")}
Debt/Equity: {info.get("debtToEquity")}
Current Ratio: {info.get("currentRatio")}
Profit Margin: {info.get("profitMargins")}
Revenue Growth: {info.get("revenueGrowth")}
Free Cash Flow: {info.get("freeCashflow")}

Fundamental score:
{score_data.get("score")} / {score_data.get("max_score")}
Rating: {score_data.get("rating")}

Write:
1. Overall business/fundamental picture
2. Main strengths
3. Main risks
4. Valuation observation
5. What an investor should investigate next

Do not give personalized financial advice.
Do not say buy, sell, or hold.
Clearly state that this is educational analysis.
"""

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
        )

        return response.output_text

    except Exception as exc:
        return f"AI summary unavailable: {exc}"


# ============================================================
# DETERMINISTIC SUMMARY
# ============================================================

def generate_basic_summary(
    data: Dict[str, Any],
    score_data: Dict[str, Any],
) -> str:

    info = data.get("info", {})
    fast = data.get("fast_info", {})

    company = (
        info.get("longName")
        or info.get("shortName")
        or data.get("ticker")
    )

    rating = score_data["rating"]

    roe = metric(data, "returnOnEquity")
    margin = metric(data, "profitMargins")
    debt_equity = metric(data, "debtToEquity")
    revenue_growth = metric(data, "revenueGrowth")

    parts = [
        f"**{company}** currently receives a fundamental rating of **{rating}** "
        f"based on the available automated metrics."
    ]

    if roe is not None:
        parts.append(f"ROE is approximately **{roe * 100:.1f}%**.")

    if margin is not None:
        parts.append(f"Profit margin is approximately **{margin * 100:.1f}%**.")

    if debt_equity is not None:
        parts.append(
            f"Debt-to-equity is approximately **{debt_equity:.1f}**."
        )

    if revenue_growth is not None:
        parts.append(
            f"Revenue growth is approximately **{revenue_growth * 100:.1f}%**."
        )

    parts.append(
        "This is an automated educational summary and should not be treated "
        "as personalized investment advice."
    )

    return " ".join(parts)


# ============================================================
# COMPANY ANALYSIS UI
# ============================================================

st.sidebar.header("Analysis Settings")

market = st.sidebar.selectbox(
    "Market",
    ["India", "US"],
    index=0,
)

ticker_input = st.sidebar.text_input(
    "Company ticker",
    value="TCS" if market == "India" else "AAPL",
    help=(
        "India examples: TCS, INFY, RELIANCE. "
        "US examples: AAPL, MSFT, GOOGL."
    ),
)

use_competitor = st.sidebar.checkbox(
    "Compare with competitor",
    value=False,
)

competitor_input = ""

if use_competitor:
    competitor_input = st.sidebar.text_input(
        "Competitor ticker",
        value="INFY" if market == "India" else "MSFT",
    )

analyze_button = st.sidebar.button(
    "🚀 Start Analysis",
    type="primary",
    use_container_width=True,
)


# ============================================================
# MAIN ANALYSIS
# ============================================================

if analyze_button:

    ticker_symbol = normalize_ticker(ticker_input, market)

    if not ticker_symbol:
        st.error("Please enter a valid company ticker.")
        st.stop()

    with st.spinner(f"Fetching data for {ticker_symbol}..."):

        try:
            company_data = get_stock_data(ticker_symbol)

        except Exception as exc:
            st.error("Data could not be retrieved.")
            st.code(str(exc))
            st.stop()

    info = company_data.get("info", {})
    fast = company_data.get("fast_info", {})

    # Basic validation
    price = fast.get("lastPrice")

    if price is None:
        st.error(
            "Data could not be retrieved for this ticker. "
            "Check the ticker/company name and try again."
        )
        st.stop()

    company_name = (
        info.get("longName")
        or info.get("shortName")
        or ticker_symbol
    )

    currency = (
        info.get("currency")
        or fast.get("currency")
        or ("INR" if market == "India" else "USD")
    )

    st.header(f"{company_name}")
    st.caption(f"{ticker_symbol} • {market} market")

    # ========================================================
    # TOP METRICS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Price",
            format_currency(price, currency),
        )

    with col2:
        st.metric(
            "Market Cap",
            format_currency(fast.get("marketCap"), currency),
        )

    with col3:
        st.metric(
            "Trailing P/E",
            format_number(info.get("trailingPE")),
        )

    with col4:
        st.metric(
            "Price / Book",
            format_number(info.get("priceToBook")),
        )

    # ========================================================
    # FUNDAMENTALS
    # ========================================================

    st.subheader("📌 Fundamental Metrics")

    metric_rows = [
        ("ROE", info.get("returnOnEquity"), "%"),
        ("ROA", info.get("returnOnAssets"), "%"),
        ("Profit Margin", info.get("profitMargins"), "%"),
        ("Operating Margin", info.get("operatingMargins"), "%"),
        ("Revenue Growth", info.get("revenueGrowth"), "%"),
        ("Earnings Growth", info.get("earningsGrowth"), "%"),
        ("Debt / Equity", info.get("debtToEquity"), ""),
        ("Current Ratio", info.get("currentRatio"), ""),
        ("Quick Ratio", info.get("quickRatio"), ""),
        ("Dividend Yield", info.get("dividendYield"), "%"),
    ]

    fundamentals = []

    for name, value, unit in metric_rows:

        value = safe_float(value)

        if value is None:
            display = "N/A"

        elif unit == "%":
            display = f"{value * 100:.2f}%"

        else:
            display = f"{value:.2f}"

        fundamentals.append(
            {
                "Metric": name,
                "Value": display,
            }
        )

    st.dataframe(
        pd.DataFrame(fundamentals),
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # 5 YEAR TABLE
    # ========================================================

    st.subheader("📅 5-Year Financial Table")

    income_stmt = company_data.get("income_stmt")

    five_year = build_five_year_table(income_stmt)

    if five_year.empty:
        st.info("5-year financial statement data is not available.")
    else:
        st.dataframe(
            five_year,
            use_container_width=True,
        )

    # ========================================================
    # SCORECARD
    # ========================================================

    st.subheader("🏆 Fundamental Scorecard")

    score_data = calculate_score(company_data)

    score_col1, score_col2, score_col3 = st.columns(3)

    with score_col1:
        if score_data["percentage"] is None:
            score_display = "N/A"
        else:
            score_display = f"{score_data['percentage']:.0f}%"

        st.metric(
            "Score",
            score_display,
        )

    with score_col2:
        st.metric(
            "Rating",
            score_data["rating"],
        )

    with score_col3:
        st.metric(
            "Checks Passed",
            f"{score_data['score']} / {score_data['max_score']}",
        )

    # ========================================================
    # STRENGTHS
    # ========================================================

    left, right = st.columns(2)

    with left:
        st.markdown("### ✅ Positive Signals")

        if score_data["positives"]:
            for item in score_data["positives"]:
                st.write("•", item)
        else:
            st.write("No strong positive signal detected.")

    # ========================================================
    # RED FLAGS
    # ========================================================

    with right:
        st.markdown("### 🚩 Red Flags")

        flags = get_red_flags(company_data)

        for flag in flags:
            st.write("•", flag)

    # ========================================================
    # AI SUMMARY
    # ========================================================

    st.subheader("🤖 AI-Written Summary")

    with st.spinner("Preparing summary..."):

        ai_summary = generate_ai_summary(
            company_data,
            score_data,
        )

    if ai_summary:
        st.markdown(ai_summary)
    else:
        st.markdown(
            generate_basic_summary(
                company_data,
                score_data,
            )
        )

        st.info(
            "AI API key is not configured, so the app is showing an "
            "automated summary instead. You can enable the AI summary "
            "through Streamlit Secrets later."
        )

    # ========================================================
    # COMPETITOR
    # ========================================================

    if use_competitor and competitor_input.strip():

        competitor_symbol = normalize_ticker(
            competitor_input,
            market,
        )

        st.subheader("⚔️ Competitor Comparison")

        with st.spinner(
            f"Fetching competitor data for {competitor_symbol}..."
        ):

            try:
                competitor_data = get_stock_data(
                    competitor_symbol
                )

            except Exception as exc:
                st.error(
                    f"Could not retrieve competitor data: {exc}"
                )
                competitor_data = None

        if competitor_data:

            company_score = calculate_score(company_data)
            competitor_score = calculate_score(competitor_data)

            comparison_rows = [
                {
                    "Metric": "Ticker",
                    ticker_symbol: ticker_symbol,
                    competitor_symbol: competitor_symbol,
                },
                {
                    "Metric": "Price",
                    ticker_symbol: format_currency(
                        company_data["fast_info"].get("lastPrice"),
                        currency,
                    ),
                    competitor_symbol: format_currency(
                        competitor_data["fast_info"].get("lastPrice"),
                        competitor_data["fast_info"].get("currency", currency),
                    ),
                },
                {
                    "Metric": "Market Cap",
                    ticker_symbol: format_currency(
                        company_data["fast_info"].get("marketCap"),
                        currency,
                    ),
                    competitor_symbol: format_currency(
                        competitor_data["fast_info"].get("marketCap"),
                        competitor_data["fast_info"].get("currency", currency),
                    ),
                },
                {
                    "Metric": "P/E",
                    ticker_symbol: format_number(
                        company_data["info"].get("trailingPE")
                    ),
                    competitor_symbol: format_number(
                        competitor_data["info"].get("trailingPE")
                    ),
                },
                {
                    "Metric": "Price / Book",
                    ticker_symbol: format_number(
                        company_data["info"].get("priceToBook")
                    ),
                    competitor_symbol: format_number(
                        competitor_data["info"].get("priceToBook")
                    ),
                },
                {
                    "Metric": "ROE",
                    ticker_symbol: (
                        f"{metric(company_data, 'returnOnEquity') * 100:.2f}%"
                        if metric(company_data, "returnOnEquity") is not None
                        else "N/A"
                    ),
                    competitor_symbol: (
                        f"{metric(competitor_data, 'returnOnEquity') * 100:.2f}%"
                        if metric(competitor_data, "returnOnEquity") is not None
                        else "N/A"
                    ),
                },
                {
                    "Metric": "Debt / Equity",
                    ticker_symbol: format_number(
                        company_data["info"].get("debtToEquity")
                    ),
                    competitor_symbol: format_number(
                        competitor_data["info"].get("debtToEquity")
                    ),
                },
                {
                    "Metric": "Profit Margin",
                    ticker_symbol: (
                        f"{metric(company_data, 'profitMargins') * 100:.2f}%"
                        if metric(company_data, "profitMargins") is not None
                        else "N/A"
                    ),
                    competitor_symbol: (
                        f"{metric(competitor_data, 'profitMargins') * 100:.2f}%"
                        if metric(competitor_data, "profitMargins") is not None
                        else "N/A"
                    ),
                },
                {
                    "Metric": "Fundamental Score",
                    ticker_symbol: (
                        f"{company_score['score']} / "
                        f"{company_score['max_score']}"
                    ),
                    competitor_symbol: (
                        f"{competitor_score['score']} / "
                        f"{competitor_score['max_score']}"
                    ),
                },
            ]

            st.dataframe(
                pd.DataFrame(comparison_rows),
                use_container_width=True,
                hide_index=True,
            )

    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.divider()

    st.caption(
        "⚠️ Disclaimer: This tool provides automated educational analysis "
        "using publicly available market and financial data. It is not "
        "personalized financial, investment, tax, or legal advice. "
        "Financial data may be delayed, incomplete, or inaccurate. "
        "Always verify important information using authoritative sources."
    )

else:

    st.info(
        "👈 Select a market, enter a ticker, and click "
        "**Start Analysis**."
    )

    st.markdown(
        """
### Supported examples

**🇮🇳 India**
- `TCS`
- `INFY`
- `RELIANCE`

**🇺🇸 US**
- `AAPL`
- `MSFT`
- `GOOGL`

Competitor comparison is **optional**. You can analyze a single company
without entering any competitor ticker.
"""
    )