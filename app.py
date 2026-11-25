import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from google import genai
from google.genai import types


st.set_page_config(
    page_title="FinPilot — IA financière",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .live-market-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.35rem 0.7rem;
        margin: 0.25rem 0 0.75rem;
        border: 1px solid rgba(0, 212, 170, 0.32);
        border-radius: 999px;
        background: rgba(0, 212, 170, 0.08);
        color: #00D4AA;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
    }
    .st-key-metric_last_price, .st-key-metric_period_change,
    .st-key-metric_high, .st-key-metric_low, .st-key-metric_volatility {
        border: 1px solid #1E293B;
        border-radius: 14px;
        padding: 0.8rem 1rem;
        background: #131A29;
        min-height: 106px;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

SYSTEM_PROMPT = """Tu es FinPilot, une IA spécialisée en FinTech et analyse financière.
Réponds toujours en français, de façon structurée, rigoureuse et pédagogique.
Explique les mécanismes économiques et évalue systématiquement les risques,
les incertitudes et les limites des données. Distingue clairement les faits,
les hypothèses et les opinions. Ne présente jamais une projection comme une
certitude et ne prétends pas fournir un conseil financier personnalisé.
Quand un contexte de marché est fourni, appuie-toi dessus, précise sa période
et signale s'il est incomplet ou retardé. Si le contexte ne permet pas de
conclure, dis-le explicitement.
"""


@st.cache_data(ttl=300, show_spinner=False)
def load_market_data(symbol: str):
    """Fetch roughly one month of daily market data and compute a compact summary."""
    history = yf.Ticker(symbol).history(period="1mo", interval="1d", auto_adjust=True)
    if history.empty or "Close" not in history:
        raise ValueError("Aucune donnée disponible pour ce symbole.")

    close = history["Close"].dropna()
    if close.empty:
        raise ValueError("Aucun cours de clôture exploitable pour ce symbole.")

    current = float(close.iloc[-1])
    first = float(close.iloc[0])
    change_pct = ((current / first) - 1) * 100 if first else 0.0
    returns = close.pct_change().dropna()
    volatility = float(returns.std() * (252**0.5) * 100) if len(returns) > 1 else None
    dates = history.index
    if getattr(dates, "tz", None) is not None:
        dates = dates.tz_convert(None)

    close_history = close.rename("Cours de clôture").to_frame()
    if getattr(close_history.index, "tz", None) is not None:
        close_history.index = close_history.index.tz_localize(None)

    try:
        currency = yf.Ticker(symbol).fast_info.get("currency", "") or ""
    except Exception:
        currency = ""

    return {
        "symbol": symbol,
        "current": current,
        "change_pct": change_pct,
        "high": float(history["High"].max()),
        "low": float(history["Low"].min()),
        "volatility": volatility,
        "start_date": dates[0].strftime("%d/%m/%Y"),
        "end_date": dates[-1].strftime("%d/%m/%Y"),
        "currency": currency,
        "close_history": close_history,
    }


def format_price(value: float) -> str:
    return f"${value:,.2f}"


def get_default_api_key() -> str:
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except st.errors.StreamlitSecretNotFoundError:
        return ""


with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    api_key_override = st.text_input(
        "Clé API Gemini",
        type="password",
        placeholder="AIza…",
        help="Utilisée uniquement pour les requêtes de cette session.",
    )
    api_key = api_key_override or get_default_api_key()
    selected_model = st.selectbox(
        "Modèle Gemini",
        options=[
            "gemini-3.6-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.8-flash",
            "gemini-3.1-flash-lite",
        ],
        index=0,
    )
    st.markdown("#### Accès rapide")
    ticker_rows = [st.columns(2), st.columns(2)]
    quick_tickers = ["AAPL", "NVDA", "BTC-USD", "^FCHI"]
    for row, ticker_pair in zip(ticker_rows, (quick_tickers[:2], quick_tickers[2:])):
        for column, ticker in zip(row, ticker_pair):
            if column.button(ticker, key=f"ticker_{ticker}", width="stretch"):
                st.session_state["symbol"] = ticker
    symbol = st.text_input(
        "Symbole financier (facultatif)",
        placeholder="AAPL, BTC-USD, ^FCHI, EURUSD=X",
        help="Les données de marché sont fournies par Yahoo Finance et peuvent être retardées.",
        key="symbol",
    ).strip().upper()
    st.caption("Les données de marché sont indicatives et peuvent être retardées ou indisponibles.")

st.markdown('<div class="live-market-badge">● LIVE MARKET &amp; AI</div>', unsafe_allow_html=True)
st.title("FinPilot")
st.markdown("### Votre copilote IA pour la finance")
st.write(
    "Explorez les marchés et les concepts FinTech avec une IA qui privilégie "
    "la clarté, les faits et l’analyse des risques."
)

market_context = None
if symbol:
    with st.spinner(f"Chargement des données de {symbol}…"):
        try:
            market_context = load_market_data(symbol)
        except Exception as exc:
            st.warning(f"Données de marché indisponibles pour **{symbol}** : {exc}")

if market_context:
    currency = market_context["currency"]
    st.markdown(f"#### Marché · {market_context['symbol']} · 30 derniers jours")
    cols = st.columns(5)
    with cols[0].container(key="metric_last_price"):
        st.metric("Dernier cours", format_price(market_context["current"]))
    with cols[1].container(key="metric_period_change"):
        st.metric("Variation période", f"{market_context['change_pct']:+.2f}%")
    with cols[2].container(key="metric_high"):
        st.metric("Plus haut", format_price(market_context["high"]))
    with cols[3].container(key="metric_low"):
        st.metric("Plus bas", format_price(market_context["low"]))
    volatility = market_context["volatility"]
    with cols[4].container(key="metric_volatility"):
        st.metric("Volatilité annualisée", f"{volatility:.1f}%" if volatility is not None else "—")

    close_history = market_context["close_history"]
    price_chart = go.Figure(
        go.Scatter(
            x=close_history.index,
            y=close_history["Cours de clôture"],
            mode="lines",
            name="Cours de clôture",
            line={"color": "#00D4AA", "width": 2.5, "shape": "spline"},
            fill="tozeroy",
            fillgradient={
                "type": "vertical",
                "colorscale": [
                    [0, "rgba(0, 212, 170, 0.015)"],
                    [1, "rgba(0, 212, 170, 0.22)"],
                ],
            },
            hovertemplate=f"%{{x|%d %b %Y}}<br>%{{y:,.2f}} {currency}<extra></extra>",
        )
    )
    price_chart.update_layout(
        height=360,
        margin={"l": 12, "r": 12, "t": 24, "b": 12},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#F8FAFC"},
        hovermode="x unified",
        showlegend=False,
        xaxis={"showgrid": True, "gridcolor": "rgba(148, 163, 184, 0.12)", "zeroline": False},
        yaxis={"title": f"Cours de clôture ({currency})", "showgrid": True, "gridcolor": "rgba(148, 163, 184, 0.12)", "zeroline": False, "autorange": True, "rangemode": "normal"},
    )
    st.plotly_chart(
        price_chart,
        width="stretch",
        theme=None,
        alt=f"Évolution du cours de clôture de {market_context['symbol']} sur 30 jours",
    )
    st.caption(
        f"Clôtures ajustées du {market_context['start_date']} au {market_context['end_date']}. "
        "Variation calculée entre la première et la dernière clôture disponible ; "
        "volatilité annualisée à partir des rendements journaliers."
    )

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown("#### Questions rapides")
quick_cols = st.columns(3)
quick_prompt = None
if quick_cols[0].button("Analyser la volatilité", key="quick_volatility"):
    quick_prompt = "Analyser la volatilité"
if quick_cols[1].button("Identifier les risques clés", key="quick_risks"):
    quick_prompt = "Identifier les risques clés"
if quick_cols[2].button("Résumer la tendance", key="quick_trend"):
    quick_prompt = "Résumer la tendance"

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Posez une question sur la finance ou la FinTech…") or quick_prompt
if prompt:
    if not api_key:
        st.info("Pour démarrer la conversation, entrez votre clé API Gemini dans la barre latérale.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                client = genai.Client(api_key=api_key)
                system_message = SYSTEM_PROMPT
                if market_context:
                    system_message += (
                        "\n\nContexte de marché récupéré automatiquement (Yahoo Finance, cours ajustés) :\n"
                        f"Symbole : {market_context['symbol']}\n"
                        f"Période : {market_context['start_date']} au {market_context['end_date']}\n"
                        f"Dernière clôture : {market_context['current']} {market_context['currency']}\n"
                        f"Variation sur la période : {market_context['change_pct']:.2f}%\n"
                        f"Plus haut / plus bas : {market_context['high']} / {market_context['low']} "
                        f"{market_context['currency']}\n"
                        f"Volatilité annualisée estimée : {market_context['volatility']}%\n"
                        "Ces données peuvent être retardées ; indique leurs limites."
                    )

                with st.spinner("FinPilot prépare une réponse…"):
                    model_fallbacks = [
                        "gemini-3.6-flash",
                        "gemini-3.5-flash-lite",
                        "gemini-3.8-flash",
                        "gemini-3.1-flash-lite",
                    ]
                    models_to_try = [selected_model] + [
                        model for model in model_fallbacks if model != selected_model
                    ]
                    response = None
                    fallback_errors = []
                    last_fallback_error = None
                    for model_name in models_to_try:
                        try:
                            response = client.models.generate_content(
                                model=model_name,
                                contents=[
                                    {
                                        "role": "user" if message["role"] == "user" else "model",
                                        "parts": [{"text": message["content"]}],
                                    }
                                    for message in st.session_state.messages
                                ],
                                config=types.GenerateContentConfig(
                                    system_instruction=system_message,
                                ),
                            )
                            break
                        except Exception as exc:
                            last_fallback_error = exc
                            error_text = f"{exc} {exc!r}".upper()
                            if "503" in error_text or "UNAVAILABLE" in error_text:
                                fallback_errors.append("unavailable")
                                continue
                            if "404" in error_text or "NOT_FOUND" in error_text:
                                fallback_errors.append("not_found")
                                continue
                            raise

                if response is None:
                    st.error(f"Détail exact de la dernière erreur Gemini : {last_fallback_error}")
                    if fallback_errors and all(error == "unavailable" for error in fallback_errors):
                        st.warning(
                            "Les serveurs Gemini sont actuellement saturés. "
                            "Veuillez réessayer dans quelques instants."
                        )
                    else:
                        st.warning(
                            "Aucun des modèles Gemini disponibles n’a pu traiter la demande. "
                            "Veuillez vérifier leur disponibilité ou réessayer plus tard."
                        )
                else:
                    answer = response.text or "Je n’ai pas pu générer de réponse."
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as exc:
                st.error("La requête Gemini a échoué. Détails complets de l’erreur :")
                st.exception(exc)

if st.session_state.messages:
    if st.button("Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()
