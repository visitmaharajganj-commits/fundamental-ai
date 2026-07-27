import streamlit as st
import yfinance as yf
import pandas as pd
import math

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
    "Fundamental research dashboard — 5-Year Financials • Scorecard • "
    "Red Flags • Optional Competitor Comparison"
)

# ============================================================
# INPUTS
# ============================================================

company = st.text_input(
    "Company / Stock Ticker",
    placeholder="Example: TCS or TCS.NS",
)

market = st.selectbox(
    "Market",
    ["India", "US"],
)

analysis_mode = st.radio(
    "Analysis Mode",
    [
        "Single Company Analysis",
        "Company + Competitors",
    ],
    horizontal=True,
)

competitors_text = ""

if analysis_mode == "Company + Competitors":
    competitors_text = st.text_area(
        "Competitor Tickers — Optional",
        placeholder="INFY, WIPRO, HCLTECH",
        help="For India, tickers are automatically converted to .NS if needed.",
    )

st.caption(
    "💡 Competitor analysis is optional. For a normal fundamental analysis, "
    "just enter one company and select Single Company Analysis."
)

# ============================================================
# HELPERS
# ============================================================

def normalize_ticker(symbol: str, market: str) -> str:
    symbol = str(symbol).strip().upper()

    if not symbol:
        return symbol

    if market == "India":
        if not symbol.endswith(".NS") and not symbol.endswith(".BO"):
            symbol += ".NS"

    return symbol


def safe_float(value):
    try:
        if value is None:
            return None

        if isinstance(value, (list, tuple, dict)):
            return None

        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return None

        return value

    except Exception:
        return None


def safe_div(a, b):
    a = safe_float(a)
    b = safe_float(b)

    if a is None or b is None or b == 0:
        return None

    return a / b


def percentage(value):
    value = safe_float(value)

    if value is None:
        return None

    if abs(value) <= 1:
        return value * 100

    return value


def latest_value(df, possible_rows):
    if df is None or df.empty:
        return None

    for row in possible_rows:

        if row in df.index:

            series = df.loc[row].dropna()

            if not series.empty:
                return safe_float(series.iloc[0])

    return None


def sorted_periods(columns):

    cols = list(columns)

    try:
        return sorted(
            cols,
            key=lambda x: pd.to_datetime(x),
            reverse=True,
        )

    except Exception:
        return cols[::-1]


def format_period(period):

    try:
        return pd.to_datetime(period).strftime("%Y")

    except Exception:
        return str(period)[:10]


def build_5yr_table(df, row_map, max_periods=5):

    if df is None or df.empty:
        return None

    periods = sorted_periods(df.columns)[:max_periods]

    if not periods:
        return None

    output = {}

    for label, possible_rows in row_map.items():

        actual_row = None

        for row in possible_rows:

            if row in df.index:
                actual_row = row
                break

        if actual_row is None:
            continue

        values = []

        for period in periods:

            try:
                values.append(
                    safe_float(df.loc[actual_row, period])
                )

            except Exception:
                values.append(None)

        output[label] = values

    if not output:
        return None

    table = pd.DataFrame(
        output,
        index=[format_period(p) for p in periods],
    ).T

    return table


def cagr_from_row(df, possible_rows):

    if df is None or df.empty:
        return None

    for row in possible_rows:

        if row not in df.index:
            continue

        series = df.loc[row].dropna()

        if len(series) < 2:
            return None

        try:

            dates = pd.to_datetime(
                series.index,
                errors="coerce",
            )

            valid = ~pd.isna(dates)

            series = pd.Series(
                series.values[valid],
                index=dates[valid],
            ).sort_index()

            if len(series) < 2:
                return None

            start = safe_float(series.iloc[0])
            end = safe_float(series.iloc[-1])

            if start is None or end is None:
                return None

            if start <= 0 or end <= 0:
                return None

            years = max(
                (series.index[-1] - series.index[0]).days
                / 365.25,
                0.01,
            )

            return (end / start) ** (1 / years) - 1

        except Exception:
            return None

    return None


# ============================================================
# SCORING
# ============================================================

def score_high_good(value, bands):

    value = safe_float(value)

    if value is None:
        return None

    score = bands[0][1]

    for threshold, score_value in bands:

        if value >= threshold:
            score = score_value

    return score


def score_low_good(value, bands):

    value = safe_float(value)

    if value is None:
        return None

    for threshold, score_value in bands:

        if value <= threshold:
            return score_value

    return bands[-1][1]


# ============================================================
# DATA FETCH
# ============================================================

def fetch_bundle(symbol):

    ticker = yf.Ticker(symbol)

    # Convert everything into ordinary Python structures.
    # This avoids cache / serialization issues.
    try:
        info = dict(ticker.info or {})
    except Exception:
        info = {}

    try:
        fast_info = dict(ticker.fast_info or {})
    except Exception:
        fast_info = {}

    try:
        income = ticker.income_stmt.copy()
    except Exception:
        income = pd.DataFrame()

    try:
        balance = ticker.balance_sheet.copy()
    except Exception:
        balance = pd.DataFrame()

    try:
        cashflow = ticker.cashflow.copy()
    except Exception:
        cashflow = pd.DataFrame()

    return {
        "info": info,
        "fast": fast_info,
        "income": income,
        "balance": balance,
        "cashflow": cashflow,
    }


# ============================================================
# EXTRACT COMPANY METRICS
# ============================================================

def extract_metrics(symbol):

    bundle = fetch_bundle(symbol)

    info = bundle["info"]
    fast = bundle["fast"]

    income = bundle["income"]
    balance = bundle["balance"]
    cashflow = bundle["cashflow"]

    company_name = (
        info.get("longName")
        or info.get("shortName")
        or symbol
    )

    price = (
        safe_float(fast.get("lastPrice"))
        or safe_float(info.get("currentPrice"))
    )

    market_cap = (
        safe_float(fast.get("marketCap"))
        or safe_float(info.get("marketCap"))
    )

    currency = (
        fast.get("currency")
        or info.get("currency")
        or ""
    )

    pe = safe_float(info.get("trailingPE"))
    pb = safe_float(info.get("priceToBook"))

    roe = safe_float(info.get("returnOnEquity"))
    debt_equity = safe_float(info.get("debtToEquity"))

    eps = safe_float(info.get("trailingEps"))
    dividend_yield = safe_float(info.get("dividendYield"))

    revenue = latest_value(
        income,
        [
            "Total Revenue",
            "Operating Revenue",
        ],
    )

    operating_income = latest_value(
        income,
        [
            "Operating Income",
            "EBIT",
        ],
    )

    net_income = latest_value(
        income,
        [
            "Net Income",
            "Net Income Common Stockholders",
        ],
    )

    gross_profit = latest_value(
        income,
        ["Gross Profit"],
    )

    ebitda = latest_value(
        income,
        ["EBITDA"],
    )

    total_assets = latest_value(
        balance,
        ["Total Assets"],
    )

    total_debt = latest_value(
        balance,
        ["Total Debt"],
    )

    equity = latest_value(
        balance,
        [
            "Stockholders Equity",
            "Total Equity Gross Minority Interest",
            "Common Stock Equity",
        ],
    )

    current_assets = latest_value(
        balance,
        ["Current Assets"],
    )

    current_liabilities = latest_value(
        balance,
        ["Current Liabilities"],
    )

    cash = latest_value(
        balance,
        [
            "Cash And Cash Equivalents",
            "Cash Cash Equivalents And Short Term Investments",
        ],
    )

    operating_cash_flow = latest_value(
        cashflow,
        ["Operating Cash Flow"],
    )

    capex = latest_value(
        cashflow,
        ["Capital Expenditure"],
    )

    free_cash_flow = latest_value(
        cashflow,
        ["Free Cash Flow"],
    )

    profit_margin = safe_div(
        net_income,
        revenue,
    )

    if profit_margin is not None:
        profit_margin *= 100

    operating_margin = safe_div(
        operating_income,
        revenue,
    )

    if operating_margin is not None:
        operating_margin *= 100

    roe_calc = safe_div(
        net_income,
        equity,
    )

    if roe_calc is not None:
        roe_calc *= 100

    debt_to_equity_calc = safe_div(
        total_debt,
        equity,
    )

    current_ratio = safe_div(
        current_assets,
        current_liabilities,
    )

    asset_turnover = safe_div(
        revenue,
        total_assets,
    )

    fcf_margin = safe_div(
        free_cash_flow,
        revenue,
    )

    if fcf_margin is not None:
        fcf_margin *= 100

    revenue_cagr = cagr_from_row(
        income,
        [
            "Total Revenue",
            "Operating Revenue",
        ],
    )

    net_income_cagr = cagr_from_row(
        income,
        [
            "Net Income",
            "Net Income Common Stockholders",
        ],
    )

    ocf_cagr = cagr_from_row(
        cashflow,
        ["Operating Cash Flow"],
    )

    fcf_cagr = cagr_from_row(
        cashflow,
        ["Free Cash Flow"],
    )

    return {
        "symbol": symbol,
        "company_name": company_name,

        "price": price,
        "market_cap": market_cap,
        "currency": currency,

        "pe": pe,
        "pb": pb,
        "roe": roe,
        "roe_calc": roe_calc,

        "debt_equity": debt_equity,
        "debt_to_equity_calc": debt_to_equity_calc,

        "eps": eps,
        "dividend_yield": dividend_yield,

        "revenue": revenue,
        "operating_income": operating_income,
        "net_income": net_income,
        "gross_profit": gross_profit,
        "ebitda": ebitda,

        "total_assets": total_assets,
        "total_debt": total_debt,
        "equity": equity,

        "current_assets": current_assets,
        "current_liabilities": current_liabilities,
        "cash": cash,

        "operating_cash_flow": operating_cash_flow,
        "capex": capex,
        "free_cash_flow": free_cash_flow,

        "profit_margin": profit_margin,
        "operating_margin": operating_margin,
        "current_ratio": current_ratio,
        "asset_turnover": asset_turnover,
        "fcf_margin": fcf_margin,

        "revenue_cagr": revenue_cagr,
        "net_income_cagr": net_income_cagr,
        "ocf_cagr": ocf_cagr,
        "fcf_cagr": fcf_cagr,

        "income": income,
        "balance": balance,
        "cashflow": cashflow,
    }


# ============================================================
# SCORECARD
# ============================================================

def build_scorecard(m):

    # -------------------------
    # Growth
    # -------------------------

    growth_scores = []

    for value in [
        m["revenue_cagr"],
        m["net_income_cagr"],
        m["ocf_cagr"],
        m["fcf_cagr"],
    ]:

        score = score_high_good(
            value,
            [
                (0.00, 1),
                (0.05, 4),
                (0.10, 6),
                (0.15, 8),
                (0.20, 10),
            ],
        )

        if score is not None:
            growth_scores.append(score)

    growth = (
        round(sum(growth_scores) / len(growth_scores), 2)
        if growth_scores
        else None
    )

    # -------------------------
    # Profitability
    # -------------------------

    profitability_scores = []

    roe_pct = (
        percentage(m["roe"])
        if m["roe"] is not None
        else m["roe_calc"]
    )

    for value in [
        roe_pct,
        m["operating_margin"],
        m["profit_margin"],
    ]:

        score = score_high_good(
            value,
            [
                (0, 1),
                (5, 4),
                (10, 6),
                (15, 8),
                (20, 10),
            ],
        )

        if score is not None:
            profitability_scores.append(score)

    profitability = (
        round(
            sum(profitability_scores)
            / len(profitability_scores),
            2,
        )
        if profitability_scores
        else None
    )

    # -------------------------
    # Balance Sheet
    # -------------------------

    bs_scores = []

    if m["current_ratio"] is not None:

        bs_scores.append(
            score_high_good(
                m["current_ratio"],
                [
                    (1.0, 4),
                    (1.25, 6),
                    (1.5, 8),
                    (2.0, 10),
                ],
            )
        )

    if m["debt_to_equity_calc"] is not None:

        bs_scores.append(
            score_low_good(
                m["debt_to_equity_calc"],
                [
                    (0.5, 10),
                    (1.0, 8),
                    (2.0, 5),
                    (999999, 2),
                ],
            )
        )

    if (
        m["cash"] is not None
        and m["total_debt"] is not None
    ):

        cash_debt = safe_div(
            m["cash"],
            m["total_debt"],
        )

        if cash_debt is not None:

            bs_scores.append(
                score_high_good(
                    cash_debt,
                    [
                        (0.25, 4),
                        (0.5, 6),
                        (1.0, 8),
                        (2.0, 10),
                    ],
                )
            )

    balance_sheet = (
        round(sum(bs_scores) / len(bs_scores), 2)
        if bs_scores
        else None
    )

    # -------------------------
    # Cash Flow
    # -------------------------

    cf_scores = []

    if m["operating_cash_flow"] is not None:

        cf_scores.append(
            10 if m["operating_cash_flow"] > 0 else 2
        )

    if m["free_cash_flow"] is not None:

        cf_scores.append(
            10 if m["free_cash_flow"] > 0 else 2
        )

    if m["fcf_margin"] is not None:

        cf_scores.append(
            score_high_good(
                m["fcf_margin"],
                [
                    (0, 1),
                    (5, 5),
                    (10, 7),
                    (15, 9),
                ],
            )
        )

    cash_flow = (
        round(sum(cf_scores) / len(cf_scores), 2)
        if cf_scores
        else None
    )

    # -------------------------
    # Valuation
    # -------------------------

    valuation_scores = []

    if m["pe"] is not None:

        valuation_scores.append(
            score_low_good(
                m["pe"],
                [
                    (15, 10),
                    (25, 8),
                    (35, 5),
                    (50, 3),
                    (999999, 2),
                ],
            )
        )

    if m["pb"] is not None:

        valuation_scores.append(
            score_low_good(
                m["pb"],
                [
                    (3, 10),
                    (5, 8),
                    (8, 5),
                    (12, 3),
                    (999999, 2),
                ],
            )
        )

    valuation = (
        round(
            sum(valuation_scores)
            / len(valuation_scores),
            2,
        )
        if valuation_scores
        else None
    )

    # -------------------------
    # Overall
    # -------------------------

    components = [
        growth,
        profitability,
        balance_sheet,
        cash_flow,
        valuation,
    ]

    components = [
        x for x in components
        if x is not None
    ]

    overall = (
        round(sum(components) / len(components), 2)
        if components
        else None
    )

    return {
        "growth": growth,
        "profitability": profitability,
        "balance_sheet": balance_sheet,
        "cash_flow": cash_flow,
        "valuation": valuation,
        "overall": overall,
    }


# ============================================================
# CLASSIFICATION
# ============================================================

def classification(score):

    if score is None:
        return "⚪ Insufficient Data"

    if score >= 7.5:
        return "🟢 Strong Business — Worth Further Research"

    if score >= 6.0:
        return "🟡 Interesting — Needs More Evidence"

    if score >= 4.5:
        return "🟠 High Risk / Unclear"

    return "🔴 Major Red Flags — Research Carefully"


# ============================================================
# SUMMARY NOTES
# ============================================================

def generate_summary_notes(primary, scorecard):

    strengths = []
    concerns = []
    next_steps = []

    overall = scorecard["overall"]

    # Growth

    if primary["revenue_cagr"] is not None:

        if primary["revenue_cagr"] >= 0.10:
            strengths.append(
                "Revenue growth looks healthy."
            )

        elif primary["revenue_cagr"] <= 0:
            concerns.append(
                "Revenue growth is weak or negative."
            )

    if primary["net_income_cagr"] is not None:

        if primary["net_income_cagr"] >= 0.10:
            strengths.append(
                "Net profit growth is strong."
            )

        elif primary["net_income_cagr"] <= 0:
            concerns.append(
                "Net profit growth is weak or negative."
            )

    # Cash Flow

    if primary["operating_cash_flow"] is not None:

        if primary["operating_cash_flow"] > 0:
            strengths.append(
                "Operating cash flow is positive."
            )

        else:
            concerns.append(
                "Operating cash flow is negative."
            )

    if primary["free_cash_flow"] is not None:

        if primary["free_cash_flow"] > 0:
            strengths.append(
                "Free cash flow is positive."
            )

        else:
            concerns.append(
                "Free cash flow is negative."
            )

    # Debt

    if primary["debt_to_equity_calc"] is not None:

        if primary["debt_to_equity_calc"] > 2:
            concerns.append(
                "Debt-to-equity looks elevated."
            )

        elif primary["debt_to_equity_calc"] < 1:
            strengths.append(
                "Leverage appears manageable."
            )

    # Liquidity

    if primary["current_ratio"] is not None:

        if primary["current_ratio"] < 1:
            concerns.append(
                "Current ratio below 1 may indicate liquidity pressure."
            )

        elif primary["current_ratio"] >= 1.5:
            strengths.append(
                "Liquidity appears comfortable."
            )

    # ROE

    if primary["roe_calc"] is not None:

        if primary["roe_calc"] >= 15:
            strengths.append(
                "ROE looks strong."
            )

        elif primary["roe_calc"] < 10:
            concerns.append(
                "ROE is relatively weak."
            )

    # Valuation

    if (
        primary["pe"] is not None
        and primary["pb"] is not None
    ):

        if (
            primary["pe"] > 30
            or primary["pb"] > 8
        ):

            concerns.append(
                "Valuation appears relatively expensive."
            )

        elif (
            primary["pe"] < 20
            and primary["pb"] < 5
        ):

            strengths.append(
                "Simple valuation metrics look relatively reasonable."
            )

    # Next Steps

    if overall is not None:

        if overall >= 7.5:

            next_steps.extend(
                [
                    "Verify the thesis using the latest annual report.",
                    "Check the latest quarterly results.",
                    "Compare valuation with historical levels.",
                ]
            )

        elif overall >= 6:

            next_steps.extend(
                [
                    "Investigate the weakest scorecard section.",
                    "Read management commentary.",
                    "Check whether growth is sustainable.",
                ]
            )

        else:

            next_steps.extend(
                [
                    "Investigate the major red flags.",
                    "Check debt and cash-flow quality.",
                    "Review the latest annual and quarterly reports.",
                ]
            )

    if overall is None:
        summary = (
            "Not enough data is available to generate a strong automated conclusion."
        )

    elif overall >= 7.5:

        summary = (
            f"{primary['company_name']} currently screens as a "
            "relatively strong fundamental business. "
            "This is a screening result, not an investment recommendation."
        )

    elif overall >= 6:

        summary = (
            f"{primary['company_name']} has several positive signals, "
            "but the investment thesis needs additional evidence."
        )

    elif overall >= 4.5:

        summary = (
            f"{primary['company_name']} shows mixed fundamental signals. "
            "Further research is required."
        )

    else:

        summary = (
            f"{primary['company_name']} currently shows several weak areas. "
            "Extra caution and deeper research are warranted."
        )

    return {
        "summary": summary,
        "strengths": strengths[:6],
        "concerns": concerns[:6],
        "next_steps": next_steps[:6],
    }


# ============================================================
# MAIN ANALYSIS
# ============================================================

if st.button(
    "🔍 Analyze Company",
    type="primary",
    use_container_width=True,
):

    if not company.strip():

        st.warning(
            "Please enter a company name or stock ticker."
        )

        st.stop()

    primary_symbol = normalize_ticker(
        company,
        market,
    )

    competitor_symbols = []

    if (
        analysis_mode == "Company + Competitors"
        and competitors_text.strip()
    ):

        raw_items = (
            competitors_text
            .replace("\n", ",")
            .split(",")
        )

        for item in raw_items:

            item = item.strip()

            if item:
                competitor_symbols.append(
                    normalize_ticker(
                        item,
                        market,
                    )
                )

    # ========================================================
    # FETCH PRIMARY COMPANY
    # ========================================================

    with st.spinner(
        f"Fetching fundamental data for {primary_symbol}..."
    ):

        try:

            primary = extract_metrics(
                primary_symbol
            )

            # Basic validation
            if (
                primary["price"] is None
                and primary["revenue"] is None
                and primary["net_income"] is None
            ):

                st.error(
                    "No meaningful financial data was returned. "
                    "Please verify the ticker."
                )

                st.stop()

            primary_score = build_scorecard(
                primary
            )

        except Exception as e:

            st.error(
                "Data could not be retrieved."
            )

            st.caption(
                f"Technical detail: {type(e).__name__}: {e}"
            )

            st.stop()

    st.success(
        f"Data loaded successfully: {primary['company_name']}"
    )

    # ========================================================
    # COMPANY OVERVIEW
    # ========================================================

    st.subheader("🏢 Company Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Stock Price",
        (
            f"{primary['price']:,.2f}"
            if primary["price"] is not None
            else "N/A"
        ),
    )

    c2.metric(
        "Market Cap",
        (
            f"{primary['market_cap']:,.0f}"
            if primary["market_cap"] is not None
            else "N/A"
        ),
    )

    c3.metric(
        "Currency",
        primary["currency"] or "N/A",
    )

    c4.metric(
        "Overall Score",
        (
            f"{primary_score['overall']:.2f}/10"
            if primary_score["overall"] is not None
            else "N/A"
        ),
    )

    st.write(
        f"**Ticker:** `{primary['symbol']}`"
    )

    st.divider()

    # ========================================================
    # KEY METRICS
    # ========================================================

    st.subheader("📊 Key Fundamental Metrics")

    roe_display = (
        percentage(primary["roe"])
        if primary["roe"] is not None
        else primary["roe_calc"]
    )

    debt_display = (
        primary["debt_equity"]
        if primary["debt_equity"] is not None
        else primary["debt_to_equity_calc"]
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "P/E",
        (
            f"{primary['pe']:.2f}"
            if primary["pe"] is not None
            else "N/A"
        ),
    )

    m2.metric(
        "P/B",
        (
            f"{primary['pb']:.2f}"
            if primary["pb"] is not None
            else "N/A"
        ),
    )

    m3.metric(
        "ROE",
        (
            f"{roe_display:.2f}%"
            if roe_display is not None
            else "N/A"
        ),
    )

    m4.metric(
        "Debt / Equity",
        (
            f"{debt_display:.2f}"
            if debt_display is not None
            else "N/A"
        ),
    )

    n1, n2, n3, n4 = st.columns(4)

    n1.metric(
        "Revenue",
        (
            f"{primary['revenue']:,.0f}"
            if primary["revenue"] is not None
            else "N/A"
        ),
    )

    n2.metric(
        "Net Profit",
        (
            f"{primary['net_income']:,.0f}"
            if primary["net_income"] is not None
            else "N/A"
        ),
    )

    n3.metric(
        "Profit Margin",
        (
            f"{primary['profit_margin']:.2f}%"
            if primary["profit_margin"] is not None
            else "N/A"
        ),
    )

    n4.metric(
        "Current Ratio",
        (
            f"{primary['current_ratio']:.2f}"
            if primary["current_ratio"] is not None
            else "N/A"
        ),
    )

    # ========================================================
    # GROWTH
    # ========================================================

    st.divider()

    st.subheader("📈 Growth & Cash Flow Trends")

    g1, g2, g3, g4 = st.columns(4)

    g1.metric(
        "Revenue CAGR",
        (
            f"{primary['revenue_cagr'] * 100:.2f}%"
            if primary["revenue_cagr"] is not None
            else "N/A"
        ),
    )

    g2.metric(
        "Net Profit CAGR",
        (
            f"{primary['net_income_cagr'] * 100:.2f}%"
            if primary["net_income_cagr"] is not None
            else "N/A"
        ),
    )

    g3.metric(
        "Operating CF CAGR",
        (
            f"{primary['ocf_cagr'] * 100:.2f}%"
            if primary["ocf_cagr"] is not None
            else "N/A"
        ),
    )

    g4.metric(
        "FCF CAGR",
        (
            f"{primary['fcf_cagr'] * 100:.2f}%"
            if primary["fcf_cagr"] is not None
            else "N/A"
        ),
    )

    # ========================================================
    # 5 YEAR TABLES
    # ========================================================

    st.divider()

    st.subheader("📅 5-Year Financial Tables")

    income_rows = {
        "Revenue": [
            "Total Revenue",
            "Operating Revenue",
        ],
        "Gross Profit": [
            "Gross Profit",
        ],
        "Operating Income": [
            "Operating Income",
            "EBIT",
        ],
        "Net Income": [
            "Net Income",
            "Net Income Common Stockholders",
        ],
        "EPS": [
            "Diluted EPS",
            "Basic EPS",
        ],
    }

    balance_rows = {
        "Total Assets": [
            "Total Assets",
        ],
        "Total Debt": [
            "Total Debt",
        ],
        "Equity": [
            "Stockholders Equity",
            "Total Equity Gross Minority Interest",
            "Common Stock Equity",
        ],
        "Current Assets": [
            "Current Assets",
        ],
        "Current Liabilities": [
            "Current Liabilities",
        ],
    }

    cashflow_rows = {
        "Operating Cash Flow": [
            "Operating Cash Flow",
        ],
        "Capital Expenditure": [
            "Capital Expenditure",
        ],
        "Free Cash Flow": [
            "Free Cash Flow",
        ],
    }

    tab1, tab2, tab3 = st.tabs(
        [
            "Income Statement",
            "Balance Sheet",
            "Cash Flow",
        ]
    )

    with tab1:

        table = build_5yr_table(
            primary["income"],
            income_rows,
        )

        if table is not None:
            st.dataframe(
                table,
                use_container_width=True,
            )
        else:
            st.info(
                "Income statement data not available."
            )

    with tab2:

        table = build_5yr_table(
            primary["balance"],
            balance_rows,
        )

        if table is not None:
            st.dataframe(
                table,
                use_container_width=True,
            )
        else:
            st.info(
                "Balance sheet data not available."
            )

    with tab3:

        table = build_5yr_table(
            primary["cashflow"],
            cashflow_rows,
        )

        if table is not None:
            st.dataframe(
                table,
                use_container_width=True,
            )
        else:
            st.info(
                "Cash flow data not available."
            )

    # ========================================================
    # SCORECARD
    # ========================================================

    st.divider()

    st.subheader("🏆 Fundamental Scorecard")

    s1, s2, s3 = st.columns(3)

    s1.metric(
        "Growth",
        (
            f"{primary_score['growth']:.2f}/10"
            if primary_score["growth"] is not None
            else "N/A"
        ),
    )

    s2.metric(
        "Profitability",
        (
            f"{primary_score['profitability']:.2f}/10"
            if primary_score["profitability"] is not None
            else "N/A"
        ),
    )

    s3.metric(
        "Balance Sheet",
        (
            f"{primary_score['balance_sheet']:.2f}/10"
            if primary_score["balance_sheet"] is not None
            else "N/A"
        ),
    )

    s4, s5, s6 = st.columns(3)

    s4.metric(
        "Cash Flow",
        (
            f"{primary_score['cash_flow']:.2f}/10"
            if primary_score["cash_flow"] is not None
            else "N/A"
        ),
    )

    s5.metric(
        "Valuation",
        (
            f"{primary_score['valuation']:.2f}/10"
            if primary_score["valuation"] is not None
            else "N/A"
        ),
    )

    s6.metric(
        "Overall",
        (
            f"{primary_score['overall']:.2f}/10"
            if primary_score["overall"] is not None
            else "N/A"
        ),
    )

    st.info(
        f"**Verdict:** {classification(primary_score['overall'])}"
    )

    # ========================================================
    # AI-STYLE SUMMARY
    # ========================================================

    st.divider()

    st.subheader("🧠 AI-Style Fundamental Summary")

    notes = generate_summary_notes(
        primary,
        primary_score,
    )

    st.markdown(
        f"### Overall View\n{notes['summary']}"
    )

    col_a, col_b, col_c = st.columns(3)

    with col_a:

        st.markdown("### ✅ Strengths")

        if notes["strengths"]:

            for item in notes["strengths"]:
                st.success(item)

        else:
            st.info(
                "No major strengths automatically identified."
            )

    with col_b:

        st.markdown("### ⚠️ Concerns")

        if notes["concerns"]:

            for item in notes["concerns"]:
                st.warning(item)

        else:
            st.success(
                "No major concerns automatically identified."
            )

    with col_c:

        st.markdown("### 🔎 Next Steps")

        if notes["next_steps"]:

            for item in notes["next_steps"]:
                st.info(item)

        else:
            st.info(
                "No additional next steps generated."
            )

    # ========================================================
    # RED FLAGS
    # ========================================================

    st.divider()

    st.subheader("🚦 Red Flag Check")

    red_flags = []

    if (
        primary["current_ratio"] is not None
        and primary["current_ratio"] < 1
    ):
        red_flags.append(
            "Current ratio is below 1 — investigate short-term liquidity."
        )

    if (
        primary["debt_to_equity_calc"] is not None
        and primary["debt_to_equity_calc"] > 2
    ):
        red_flags.append(
            "Debt-to-equity is high — investigate leverage and interest burden."
        )

    if (
        primary["operating_cash_flow"] is not None
        and primary["operating_cash_flow"] < 0
    ):
        red_flags.append(
            "Operating cash flow is negative."
        )

    if (
        primary["free_cash_flow"] is not None
        and primary["free_cash_flow"] < 0
    ):
        red_flags.append(
            "Free cash flow is negative."
        )

    if (
        primary["net_income"] is not None
        and primary["net_income"] > 0
        and primary["operating_cash_flow"] is not None
        and primary["operating_cash_flow"] < 0
    ):
        red_flags.append(
            "Accounting profit is positive while operating cash flow is negative — investigate earnings quality."
        )

    if (
        primary["pe"] is not None
        and primary["pe"] > 40
    ):
        red_flags.append(
            "P/E is high — expectations embedded in valuation should be checked."
        )

    if red_flags:

        for flag in red_flags:
            st.warning(flag)

    else:

        st.success(
            "No obvious red flags detected by this automated screen."
        )

    # ========================================================
    # OPTIONAL COMPETITOR ANALYSIS
    # ========================================================

    if analysis_mode == "Company + Competitors":

        st.divider()

        st.subheader(
            "🏁 Optional Competitor Comparison"
        )

        if not competitor_symbols:

            st.info(
                "No competitors entered. The company analysis above is complete."
            )

        else:

            peer_rows = []

            # Primary company

            peer_rows.append(
                {
                    "Company": primary["company_name"],
                    "Ticker": primary["symbol"],
                    "P/E": primary["pe"],
                    "P/B": primary["pb"],
                    "ROE (%)": roe_display,
                    "Debt/Equity": debt_display,
                    "Revenue": primary["revenue"],
                    "Net Income": primary["net_income"],
                    "Operating CF": primary["operating_cash_flow"],
                    "Free CF": primary["free_cash_flow"],
                    "Overall Score": primary_score["overall"],
                }
            )

            # Competitors

            for symbol in competitor_symbols:

                try:

                    peer = extract_metrics(symbol)
                    peer_score = build_scorecard(peer)

                    peer_roe = (
                        percentage(peer["roe"])
                        if peer["roe"] is not None
                        else peer["roe_calc"]
                    )

                    peer_debt = (
                        peer["debt_equity"]
                        if peer["debt_equity"] is not None
                        else peer["debt_to_equity_calc"]
                    )

                    peer_rows.append(
                        {
                            "Company": peer["company_name"],
                            "Ticker": peer["symbol"],
                            "P/E": peer["pe"],
                            "P/B": peer["pb"],
                            "ROE (%)": peer_roe,
                            "Debt/Equity": peer_debt,
                            "Revenue": peer["revenue"],
                            "Net Income": peer["net_income"],
                            "Operating CF": peer["operating_cash_flow"],
                            "Free CF": peer["free_cash_flow"],
                            "Overall Score": peer_score["overall"],
                        }
                    )

                except Exception as e:

                    st.warning(
                        f"Could not retrieve competitor {symbol}: {e}"
                    )

            if len(peer_rows) > 1:

                peer_df = pd.DataFrame(
                    peer_rows
                )

                peer_df = peer_df.sort_values(
                    "Overall Score",
                    ascending=False,
                    na_position="last",
                )

                st.dataframe(
                    peer_df,
                    use_container_width=True,
                )

                st.caption(
                    "This is a simple screening comparison. "
                    "It is not a substitute for detailed industry-specific analysis."
                )

            else:

                st.info(
                    "No competitor data could be retrieved."
                )

    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.divider()

    with st.expander(
        "ℹ️ How to interpret this tool"
    ):

        st.write(
            """
            **Growth:** Revenue, profit and cash-flow growth.

            **Profitability:** ROE, operating margin and profit margin.

            **Balance Sheet:** Debt, liquidity and leverage.

            **Cash Flow:** Whether reported profits are supported by cash generation.

            **Valuation:** Simple P/E and P/B based screening.

            **Overall Score:** A blended screening score from 0–10.

            This tool is intended for research and education.
            It is not a Buy/Sell recommendation.
            """
        )

    st.caption(
        "⚠️ Always verify important figures against the latest annual report, "
        "quarterly results, exchange filings and company disclosures before making investment decisions."
    )