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
Análisis estadístico **robusto** de rendimientos diarios.
Enfocado en **riesgo, régimen y frecuencia real de caídas**.
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
    """
    Devuelve:
    - valor del umbral
    - número de eventos
    - gap promedio entre eventos
    - últimos 3 gaps reales entre eventos
    """
    events = returns[returns <= threshold]

    if len(events) < 2:
        return len(events), None, []

    gaps = events.index.to_series().diff().dt.days.dropna()

    avg_gap = gaps.mean()
    last_3_gaps = gaps.tail(3).astype(int).tolist()

    return len(events), avg_gap, last_3_gaps

def days_since_last_event(returns, threshold):
    events = returns[returns <= threshold]
    if len(events) == 0:
        return None
    return (returns.index[-1] - events.index[-1]).days

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

    th_1 = mu - sigma
    th_2 = mu - 2 * sigma
    th_3 = mu - 3 * sigma

    # =========================
    # EVENTOS Y GAPS
    # =========================
    n1, gap1, last1 = event_gap_stats(returns, th_1)
    n2, gap2, last2 = event_gap_stats(returns, th_2)
    n3, gap3, last3 = event_gap_stats(returns, th_3)

    d1 = days_since_last_event(returns, th_1)
    d2 = days_since_last_event(returns, th_2)
    d3 = days_since_last_event(returns, th_3)

    # =========================
    # TABS
    # =========================
    tab1, tab2 = st.tabs(["🧠 Régimen y Caídas", "📉 Distribución"])

    # =========================
    # TAB 1
    # =========================
    with tab1:
        st.subheader("📊 Estadísticas base")

        c1, c2 = st.columns(2)
        c1.metric("Media diaria", f"{mu*100:.3f}%")
        c2.metric("Volatilidad diaria", f"{sigma*100:.3f}%")

        st.markdown("## 📉 Caídas estadísticas")

        st.markdown(f"""
### 🟡 Caída moderada (μ − 1σ)
- Valor: **{th_1*100:.2f}%**
- Eventos totales: **{n1}**
- Frecuencia promedio: **cada {gap1:.1f} días**
- Últimos gaps reales: **{last1}**
- Días desde la última: **{d1}**
""")

        st.markdown(f"""
### 🟠 Caída fuerte (μ − 2σ)
- Valor: **{th_2*100:.2f}%**
- Eventos totales: **{n2}**
- Frecuencia promedio: **cada {gap2:.1f} días**
- Últimos gaps reales: **{last2}**
- Días desde la última: **{d2}**
""")

        st.markdown(f"""
### 🔴 Caída muy fuerte (μ − 3σ)
- Valor: **{th_3*100:.2f}%**
- Eventos totales: **{n3}**
- Frecuencia promedio: **cada {gap3:.1f} días**
- Últimos gaps reales: **{last3}**
- Días desde la última: **{d3}**
""")

    # =========================
    # TAB 2
    # =========================
    with tab2:
        st.subheader("Distribución de rendimientos diarios")

        fig, ax = plt.subplots()
        ax.hist(returns, bins=50, alpha=0.7)
        ax.axvline(th_1, linestyle="--", label="μ − 1σ")
        ax.axvline(th_2, linestyle="--", label="μ − 2σ")
        ax.axvline(th_3, linestyle="--", label="μ − 3σ")
        ax.legend()
        ax.grid()
        st.pyplot(fig)

        st.markdown("""
**Interpretación**
- Los gaps muestran **cuánto tarda el mercado en volver a estresar**
- Gaps cortos consecutivos → régimen inestable
- Gaps largos → acumulación silenciosa de riesgo
""")
