import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# =========================
# CONFIG STREAMLIT
# =========================
st.set_page_config(page_title="Market Regime Analyzer", layout="centered")
st.title("📊 Market Regime Analyzer")

st.markdown("""
Análisis estadístico **integral** de rendimientos diarios:
- **Severidad y frecuencia de caídas**
- **Gaps reales entre eventos**
- **Comportamiento post-caída (5, 10, 15 días)**

No es un sistema predictivo.  
Es una **herramienta de contexto y gestión de riesgo**.
""")

# =========================
# INPUTS
# =========================
ticker = st.text_input("Ticker", value="AAPL")
start_date = st.date_input("Fecha inicio", value=pd.to_datetime("2015-01-01"))
run = st.button("Ejecutar análisis")

# =========================
# FUNCIONES
# =========================
def event_gap_stats(returns, threshold):
    events = returns[returns <= threshold]
    if len(events) < 2:
        return len(events), None, []
    gaps = events.index.to_series().diff().dt.days.dropna()
    return len(events), float(gaps.mean()), gaps.tail(3).astype(int).tolist()

def days_since_last_event(returns, threshold):
    events = returns[returns <= threshold]
    if len(events) == 0:
        return None
    return (returns.index[-1] - events.index[-1]).days

def post_event_stats(close, returns, threshold, horizons=(5, 10, 15)):
    events = returns[returns <= threshold].index
    stats = {}

    for h in horizons:
        vals = []
        for dt in events:
            if dt not in close.index:
                continue
            idx = close.index.get_loc(dt)
            if isinstance(idx, slice) or idx + h >= len(close):
                continue
            vals.append(close.iloc[idx + h] / close.iloc[idx] - 1)

        if len(vals) == 0:
            stats[h] = None
            continue

        vals = np.array(vals)
        stats[h] = {
            "prob_pos": float((vals > 0).mean()),
            "mean_pos": float(vals[vals > 0].mean()) if np.any(vals > 0) else None,
            "mean_neg": float(vals[vals < 0].mean()) if np.any(vals < 0) else None
        }

    return stats

# =========================
# MAIN
# =========================
if run:

    with st.spinner("Descargando datos diarios..."):
        df = yf.download(
            ticker,
            start=start_date,
            interval="1d",
            auto_adjust=True,
            progress=False
        )

    if df.empty or "Close" not in df.columns:
        st.error("No se pudieron descargar datos.")
        st.stop()

    close = df["Close"].squeeze()
    returns = close.pct_change().dropna().squeeze()

    if len(returns) < 100:
        st.error("Muy pocos datos para análisis estadístico.")
        st.stop()

    # =========================
    # ESTADÍSTICAS BASE
    # =========================
    mu = float(returns.mean())
    sigma = float(returns.std())

    th1 = mu - sigma
    th2 = mu - 2 * sigma
    th3 = mu - 3 * sigma

    # =========================
    # EVENTOS Y GAPS
    # =========================
    n1, g1, l1 = event_gap_stats(returns, th1)
    n2, g2, l2 = event_gap_stats(returns, th2)
    n3, g3, l3 = event_gap_stats(returns, th3)

    d1 = days_since_last_event(returns, th1)
    d2 = days_since_last_event(returns, th2)
    d3 = days_since_last_event(returns, th3)

    # =========================
    # POST-CAÍDA
    # =========================
    post1 = post_event_stats(close, returns, th1)
    post2 = post_event_stats(close, returns, th2)
    post3 = post_event_stats(close, returns, th3)

    # =========================
    # TABS
    # =========================
    tab1, tab2, tab3 = st.tabs(
        ["🧠 Régimen & Caídas", "📈 Post-Caída", "📉 Distribución"]
    )

    # =========================
    # TAB 1 — RÉGIMEN
    # =========================
    with tab1:
        st.subheader("📊 Estadísticas base")

        c1, c2 = st.columns(2)
        c1.metric("Media diaria", f"{mu*100:.3f}%")
        c2.metric("Volatilidad diaria", f"{sigma*100:.3f}%")

        st.markdown("## 📉 Caídas estadísticas")

        def render_regime(title, th, n, g, l, d):
            st.markdown(f"""
### {title}
- Valor de la caída: **{th*100:.2f}%**
- Eventos totales: **{n}**
- Frecuencia promedio: **cada {g:.1f} días**
- Últimos gaps reales: **{l}**
- Días desde la última: **{d}**
""")

        render_regime("🟡 μ − 1σ", th1, n1, g1, l1, d1)
        render_regime("🟠 μ − 2σ", th2, n2, g2, l2, d2)
        render_regime("🔴 μ − 3σ", th3, n3, g3, l3, d3)

    # =========================
    # TAB 2 — POST-CAÍDA
    # =========================
    with tab2:
        st.subheader("📈 Comportamiento post-caída")

        def render_post(title, post):
            st.markdown(f"### {title}")
            for h, s in post.items():
                if s is None:
                    st.write(f"{h} días → sin datos")
                else:
                    st.write(
                        f"{h} días → "
                        f"Prob.+ **{s['prob_pos']*100:.1f}%**, "
                        f"Ret.+ **{s['mean_pos']*100:.2f}%**, "
                        f"Ret.− **{s['mean_neg']*100:.2f}%**"
                    )

        render_post("🟡 μ − 1σ", post1)
        render_post("🟠 μ − 2σ", post2)
        render_post("🔴 μ − 3σ", post3)

    # =========================
    # TAB 3 — DISTRIBUCIÓN
    # =========================
    with tab3:
        st.subheader("Distribución de rendimientos diarios")

        fig, ax = plt.subplots()
        ax.hist(returns, bins=50, alpha=0.7)
        ax.axvline(th1, linestyle="--", label="μ − 1σ")
        ax.axvline(th2, linestyle="--", label="μ − 2σ")
        ax.axvline(th3, linestyle="--", label="μ − 3σ")
        ax.legend()
        ax.grid()
        st.pyplot(fig)

        st.markdown("""
**Interpretación**
- Gaps cortos consecutivos → deterioro acelerado
- Probabilidades post-caída muestran asimetría riesgo/beneficio
- No usar como señal directa, sino como **contexto de exposición**
""")

