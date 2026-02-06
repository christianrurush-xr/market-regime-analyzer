import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Return Regime Analyzer", layout="centered")
st.title("📊 Return Regime Analyzer")

st.markdown("""
Análisis **avanzado y estadístico** de rendimientos diarios.
Esta herramienta **NO predice**, sino que evalúa **riesgo, régimen y contexto de mercado**.
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
def days_since_event(returns, threshold):
    idx = returns[returns <= threshold].index
    if len(idx) == 0:
        return None
    return (returns.index[-1] - idx[-1]).days

def prob_positive_after_drop(returns, threshold):
    drops = returns[returns <= threshold]
    drops = drops[drops.index < returns.index[-1]]

    if len(drops) == 0:
        return None, 0

    next_returns = returns.shift(-1).loc[drops.index]
    return (next_returns > 0).mean(), len(drops)

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

    close = df["Close"]
    returns = close.pct_change().dropna()

    # =========================
    # STATS
    # =========================
    mu = returns.mean()
    sigma = returns.std()

    th_mod = mu - sigma
    th_fuerte = mu - 2 * sigma
    th_muy_fuerte = mu - 3 * sigma

    d_mod = days_since_event(returns, th_mod)
    d_fuerte = days_since_event(returns, th_fuerte)
    d_muy_fuerte = days_since_event(returns, th_muy_fuerte)

    p_mod, n_mod = prob_positive_after_drop(returns, th_mod)
    p_fuerte, n_fuerte = prob_positive_after_drop(returns, th_fuerte)
    p_muy_fuerte, n_muy_fuerte = prob_positive_after_drop(returns, th_muy_fuerte)

    # =========================
    # FEATURE ENGINEERING
    # =========================
    close = df["Close"].squeeze()
    returns = close.pct_change().dropna().squeeze()

    df_ret = returns.to_frame("ret")
    df_ret["weekday"] = df_ret.index.dayofweek
    df_ret["day"] = df_ret.index.day
    df_ret["month"] = df_ret.index.month


    weekday_map = {
        0: "Lunes", 1: "Martes", 2: "Miércoles",
        3: "Jueves", 4: "Viernes"
    }

    # =========================
    # TABS
    # =========================
    tab1, tab2, tab3 = st.tabs(
        ["🧠 Régimen y Riesgo", "📆 Patrones Temporales", "📉 Distribución"]
    )

    # =========================
    # TAB 1 — RIESGO
    # =========================
    with tab1:
        st.subheader("Régimen estadístico actual")

        c1, c2, c3 = st.columns(3)
        c1.metric("Media diaria", f"{mu*100:.3f}%")
        c2.metric("Desv. estándar", f"{sigma*100:.3f}%")
        c3.metric("Último retorno", f"{returns.iloc[-1]*100:.2f}%")

        st.markdown("### ⏱️ Días desde la última caída")

        st.write(f"🟡 Moderada (μ − 1σ): **{d_mod if d_mod is not None else 'Nunca'} días**")
        st.write(f"🟠 Fuerte (μ − 2σ): **{d_fuerte if d_fuerte is not None else 'Nunca'} días**")
        st.write(f"🔴 Muy fuerte (μ − 3σ): **{d_muy_fuerte if d_muy_fuerte is not None else 'Nunca'} días**")

        st.markdown("### 🔁 Probabilidad de rebote al día siguiente")

        if p_mod is not None:
            st.write(f"🟡 Moderada: **{p_mod*100:.1f}%**  (n={n_mod})")
        else:
            st.write("🟡 Moderada: sin eventos suficientes")

        if p_fuerte is not None:
            st.write(f"🟠 Fuerte: **{p_fuerte*100:.1f}%**  (n={n_fuerte})")
        else:
            st.write("🟠 Fuerte: sin eventos suficientes")

        if p_muy_fuerte is not None:
            st.write(f"🔴 Muy fuerte: **{p_muy_fuerte*100:.1f}%**  (n={n_muy_fuerte})")
        else:
            st.write("🔴 Muy fuerte: sin eventos suficientes")

        st.markdown("""
        **Interpretación rápida**  
        - >50% sugiere rebote estadístico (no garantía)  
        - <50% indica persistencia de debilidad  
        - Eventos extremos suelen aumentar volatilidad posterior  
        """)

    # =========================
    # TAB 2 — PATRONES
    # =========================
    with tab2:
        st.subheader("Rendimiento promedio por día de la semana")

        by_weekday = df_ret.groupby("weekday")["ret"].mean()
        by_weekday.index = by_weekday.index.map(weekday_map)

        fig1, ax1 = plt.subplots()
        by_weekday.plot(kind="bar", ax=ax1)
        ax1.set_ylabel("Retorno promedio")
        ax1.grid()
        st.pyplot(fig1)

        st.subheader("Rendimiento promedio por día del mes")

        by_day = df_ret.groupby("day")["ret"].mean()

        fig2, ax2 = plt.subplots()
        by_day.plot(ax=ax2)
        ax2.set_ylabel("Retorno promedio")
        ax2.grid()
        st.pyplot(fig2)

        st.subheader("Rendimiento promedio por mes")

        by_month = df_ret.groupby("month")["ret"].mean()

        fig3, ax3 = plt.subplots()
        by_month.plot(kind="bar", ax=ax3)
        ax3.set_ylabel("Retorno promedio")
        ax3.grid()
        st.pyplot(fig3)

    # =========================
    # TAB 3 — DISTRIBUCIÓN
    # =========================
    with tab3:
        st.subheader("Distribución de rendimientos diarios")

        fig, ax = plt.subplots()
        ax.hist(returns, bins=50, alpha=0.7)
        ax.axvline(th_mod, color="orange", linestyle="--", label="μ − 1σ")
        ax.axvline(th_fuerte, color="red", linestyle="--", label="μ − 2σ")
        ax.axvline(th_muy_fuerte, color="darkred", linestyle="--", label="μ − 3σ")
        ax.legend()
        ax.grid()
        st.pyplot(fig)

        st.markdown("""
        **Lectura avanzada**
        - Las colas muestran eventos raros pero críticos  
        - Mucho tiempo sin caídas extremas ≠ bajo riesgo  
        - Esta gráfica es clave para entender *colas gordas*
        """)

    # =========================
    # CONCLUSIÓN GLOBAL
    # =========================
    st.markdown("---")
    st.subheader("🧠 Conclusión general")

    if d_muy_fuerte is not None and d_muy_fuerte < 30:
        st.error("🔴 Régimen de **alto riesgo reciente**. Priorizar gestión de riesgo.")
    elif d_fuerte is not None and d_fuerte < 30:
        st.warning("🟠 Volatilidad elevada reciente. Precaución táctica.")
    else:
        st.success("🟢 Régimen estadísticamente estable.")
