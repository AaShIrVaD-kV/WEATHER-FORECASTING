"""
Weather Prediction System
A clean, minimal Streamlit app with 3-page navigation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Weather Prediction System",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  – Clean, minimal professional look
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
      font-family: 'Inter', sans-serif;
  }

  /* ── Background ── */
  .stApp {
      background-color: #f8f9fc;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
      background-color: #ffffff;
      border-right: 1px solid #e8eaf0;
  }
  [data-testid="stSidebar"] * { color: #374151 !important; }

  /* ── Navigation item active ── */
  .nav-item {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 10px 14px;
      border-radius: 10px;
      margin-bottom: 6px;
      cursor: pointer;
      font-weight: 500;
      font-size: 0.95rem;
      color: #6b7280;
      transition: all 0.2s ease;
  }
  .nav-item.active {
      background-color: #eff6ff;
      color: #2563eb !important;
      font-weight: 600;
  }
  .nav-item:hover {
      background-color: #f3f4f6;
  }

  /* ── Page title ── */
  .page-title {
      font-size: 1.9rem;
      font-weight: 700;
      color: #111827;
      margin-bottom: 4px;
  }
  .page-subtitle {
      font-size: 0.95rem;
      color: #6b7280;
      margin-bottom: 28px;
  }

  /* ── Divider ── */
  .divider {
      height: 1px;
      background: #e5e7eb;
      margin: 20px 0;
  }

  /* ── Input card ── */
  .input-section {
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 16px;
      padding: 28px 26px;
  }

  /* ── Result box ── */
  .result-box {
      background: linear-gradient(135deg, #eff6ff 0%, #f0fdf4 100%);
      border: 1.5px solid #bfdbfe;
      border-radius: 16px;
      padding: 32px 24px;
      text-align: center;
      margin-top: 20px;
  }
  .result-condition {
      font-size: 2rem;
      font-weight: 700;
      color: #1d4ed8;
      margin: 8px 0 4px;
  }
  .result-meta {
      font-size: 0.88rem;
      color: #6b7280;
  }

  /* ── Section heading ── */
  .section-heading {
      font-size: 1.05rem;
      font-weight: 600;
      color: #374151;
      margin-bottom: 16px;
      padding-bottom: 8px;
      border-bottom: 2px solid #e5e7eb;
  }

  /* ── About card ── */
  .about-card {
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 16px;
      padding: 32px;
      max-width: 680px;
  }

  /* ── Analytics stat card ── */
  .stat-card {
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 12px;
      padding: 18px 20px;
      text-align: center;
  }
  .stat-value {
      font-size: 1.5rem;
      font-weight: 700;
      color: #2563eb;
  }
  .stat-label {
      font-size: 0.82rem;
      color: #9ca3af;
      margin-top: 2px;
  }

  /* ── Button ── */
  .stButton > button {
      background: #2563eb;
      color: white;
      border: none;
      border-radius: 10px;
      padding: 11px 32px;
      font-weight: 600;
      font-size: 0.95rem;
      letter-spacing: 0.3px;
      transition: background 0.2s ease;
  }
  .stButton > button:hover {
      background: #1d4ed8;
  }

  /* ── Hide defaults ── */
  footer { visibility: hidden; }
  #MainMenu { visibility: hidden; }
  .block-container { padding-top: 2rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
CITIES = [
    "Mumbai", "Delhi", "Chennai", "Kolkata",
    "Bengaluru", "Hyderabad", "Pune", "Jaipur",
    "Lucknow", "Kochi",
]

SEASONS = ["Summer", "Winter", "Monsoon", "Spring"]

CONDITIONS = ["Sunny", "Cloudy", "Rainy", "Stormy", "Foggy"]

CONDITION_EMOJI = {
    "Sunny": "☀️", "Cloudy": "⛅", "Rainy": "🌧️",
    "Stormy": "⛈️", "Foggy": "🌫️",
}

CITY_PARAMS = {
    "Mumbai":    (29, 5,  78), "Delhi":     (25, 12, 55),
    "Chennai":   (31, 4,  75), "Kolkata":   (28, 7,  72),
    "Bengaluru": (24, 4,  65), "Hyderabad": (27, 6,  60),
    "Pune":      (26, 5,  62), "Jaipur":    (28, 11, 45),
    "Lucknow":   (26, 10, 58), "Kochi":     (30, 3,  85),
}

np.random.seed(42)


# ─────────────────────────────────────────────
# SYNTHETIC DATA & MODEL
# ─────────────────────────────────────────────
@st.cache_data
def generate_data(n: int = 3000) -> pd.DataFrame:
    records = []
    for _ in range(n):
        city   = np.random.choice(CITIES)
        season = np.random.choice(SEASONS)
        t_mean, t_std, h_mean = CITY_PARAMS[city]

        season_temp  = {"Winter": -8, "Spring": 0,  "Summer": 8,  "Monsoon": 2}
        season_humid = {"Winter": -10,"Spring": 0,  "Summer": -8, "Monsoon": 20}
        season_wind  = {"Winter": 10, "Spring": 14, "Summer": 18, "Monsoon": 22}

        temperature = round(np.random.normal(t_mean + season_temp[season], t_std), 1)
        humidity    = int(np.clip(np.random.normal(h_mean + season_humid[season], 12), 10, 100))
        wind_speed  = round(np.clip(np.random.normal(season_wind[season], 7), 0, 60), 1)

        # ── Improved Rule-based label (Indian climate aware) ──────────────
        # Northern cities prone to winter fog
        NORTHERN_CITIES = {"Delhi", "Jaipur", "Lucknow"}

        condition = None  # will be assigned below

        # ── MONSOON (July – September) ────────────────────────────────────
        if season == "Monsoon":
            if wind_speed > 30 and humidity > 80:
                condition = "Stormy"
            elif humidity > 70 or wind_speed > 20:
                condition = "Rainy"
            elif humidity > 55:
                condition = "Cloudy"
            else:
                # Dry spell within monsoon — still cloudy
                condition = "Cloudy"

        # ── SUMMER (March – June) ─────────────────────────────────────────
        elif season == "Summer":
            if temperature < 15:
                # Unusually cold for summer → cloudy / rainy
                condition = "Cloudy" if humidity < 70 else "Rainy"
            elif wind_speed > 32 and humidity > 65:
                condition = "Stormy"
            elif humidity > 75:
                condition = "Rainy"
            elif humidity > 55 or temperature < 25:
                condition = "Cloudy"
            else:
                # Hot, dry, low-humidity → Sunny
                condition = "Sunny"

        # ── WINTER (November – January) ───────────────────────────────────
        elif season == "Winter":
            if wind_speed > 30 and humidity > 75:
                condition = "Stormy"
            elif humidity > 80 and wind_speed < 12:
                condition = "Foggy"
            elif city in NORTHERN_CITIES and humidity > 65 and wind_speed < 18 and temperature < 15:
                # North Indian dense fog in cool, calm, humid conditions
                condition = "Foggy"
            elif temperature < 10 and humidity > 70:
                condition = "Foggy"
            elif humidity > 65 or temperature < 18:
                condition = "Cloudy"
            else:
                condition = "Sunny"

        # ── SPRING (February – March) ─────────────────────────────────────
        elif season == "Spring":
            if wind_speed > 30 and humidity > 70:
                condition = "Stormy"
            elif humidity > 75:
                condition = "Rainy"
            elif humidity > 55 or (temperature < 20):
                condition = "Cloudy"
            else:
                condition = "Sunny"

        # ── Safety fallback (should never reach here) ─────────────────────
        if condition is None:
            condition = "Cloudy"

        records.append({
            "City": city, "Season": season,
            "Temperature": temperature, "Humidity": humidity,
            "Wind_Speed": wind_speed, "Condition": condition,
        })

    return pd.DataFrame(records)


@st.cache_resource
def train_model(_df: pd.DataFrame):
    le_city   = LabelEncoder()
    le_season = LabelEncoder()
    le_cond   = LabelEncoder()

    df2 = _df.copy()
    df2["City_enc"]   = le_city.fit_transform(df2["City"])
    df2["Season_enc"] = le_season.fit_transform(df2["Season"])
    df2["Cond_enc"]   = le_cond.fit_transform(df2["Condition"])

    X = df2[["City_enc", "Season_enc", "Temperature", "Humidity", "Wind_Speed"]]
    y = df2["Cond_enc"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, le_city, le_season, le_cond


# ─────────────────────────────────────────────
# LOAD DATA & TRAIN
# ─────────────────────────────────────────────
df = generate_data()
model, le_city, le_season, le_cond = train_model(df)


# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-size:1.3rem;font-weight:700;color:#111827;"
        "margin-bottom:4px;'>🌤️ WeatherPredict</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:0.82rem;color:#9ca3af;margin-bottom:24px;'>"
        "ML-based weather prediction</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    pages = {
        "🏠  Home": "Home",
        "📊  Analytics": "Analytics",
        "ℹ️  About": "About",
    }

    if "page" not in st.session_state:
        st.session_state.page = "Home"

    for label, value in pages.items():
        active = "active" if st.session_state.page == value else ""
        if st.button(label, key=f"nav_{value}", use_container_width=True):
            st.session_state.page = value
            st.rerun()

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.78rem;color:#d1d5db;'>Built with Streamlit · scikit-learn</div>",
        unsafe_allow_html=True,
    )


page = st.session_state.page


# ══════════════════════════════════════════════
# PAGE 1 – HOME (PREDICTION)
# ══════════════════════════════════════════════
if page == "Home":
    st.markdown("<div class='page-title'>Weather Prediction System</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='page-subtitle'>Simple machine learning-based weather prediction tool.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Input form ──
    with st.container():
        col_left, col_right = st.columns([1, 1], gap="large")

        with col_left:
            st.markdown("<div class='section-heading'>Location & Season</div>", unsafe_allow_html=True)
            city   = st.selectbox("Select City", CITIES, key="city")
            season = st.selectbox("Season", SEASONS, key="season")

        with col_right:
            st.markdown("<div class='section-heading'>Weather Conditions</div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:0.85rem;font-weight:600;color:#374151;margin-bottom:2px;'>🌡️ Temperature (°C)</div>", unsafe_allow_html=True)
            temperature = st.number_input("Temperature (°C)", min_value=-5, max_value=50, value=28, step=1, label_visibility="collapsed")

            st.markdown("<div style='font-size:0.85rem;font-weight:600;color:#374151;margin-top:10px;margin-bottom:2px;'>💧 Humidity (%)</div>", unsafe_allow_html=True)
            humidity    = st.number_input("Humidity (%)", min_value=10, max_value=100, value=65, step=1, label_visibility="collapsed")

            st.markdown("<div style='font-size:0.85rem;font-weight:600;color:#374151;margin-top:10px;margin-bottom:2px;'>💨 Wind Speed (km/h)</div>", unsafe_allow_html=True)
            wind_speed  = st.number_input("Wind Speed (km/h)", min_value=0, max_value=60, value=12, step=1, label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)

    _, btn_col, _ = st.columns([1, 1, 1])
    with btn_col:
        predict_btn = st.button("Predict Weather", use_container_width=True)

    # ── Prediction result ──
    if predict_btn:
        city_enc   = le_city.transform([city])[0]
        season_enc = le_season.transform([season])[0]

        row = pd.DataFrame([{
            "City_enc":    city_enc,
            "Season_enc":  season_enc,
            "Temperature": temperature,
            "Humidity":    humidity,
            "Wind_Speed":  wind_speed,
        }])

        pred_enc   = model.predict(row)[0]
        prediction = le_cond.inverse_transform([pred_enc])[0]
        emoji      = CONDITION_EMOJI.get(prediction, "🌡️")

        st.markdown("<br>", unsafe_allow_html=True)

        _, res_col, _ = st.columns([1, 2, 1])
        with res_col:
            st.markdown(f"""
            <div class="result-box">
                <div style="font-size:3.2rem;">{emoji}</div>
                <div class="result-condition">{prediction}</div>
                <div class="result-meta">
                    Predicted condition for <strong>{city}</strong>
                    during <strong>{season}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE 2 – ANALYTICS
# ══════════════════════════════════════════════
elif page == "Analytics":
    st.markdown("<div class='page-title'>Analytics</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='page-subtitle'>A quick look at the dataset and key patterns.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Basic stats ──
    st.markdown("<div class='section-heading'>Basic Statistics</div>", unsafe_allow_html=True)

    num_cols = ["Temperature", "Humidity", "Wind_Speed"]
    stats = df[num_cols].agg(["mean", "min", "max"]).T

    s1, s2, s3 = st.columns(3)
    stat_blocks = [
        (s1, "Temperature (°C)", stats.loc["Temperature"]),
        (s2, "Humidity (%)",     stats.loc["Humidity"]),
        (s3, "Wind Speed (km/h)",stats.loc["Wind_Speed"]),
    ]
    for col, label, row in stat_blocks:
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div style="font-size:0.82rem;color:#6b7280;font-weight:500;margin-bottom:10px;">{label}</div>
                <div style="display:flex;justify-content:space-around;">
                    <div>
                        <div class="stat-value">{row['mean']:.1f}</div>
                        <div class="stat-label">Mean</div>
                    </div>
                    <div>
                        <div class="stat-value" style="color:#16a34a;">{row['min']:.1f}</div>
                        <div class="stat-label">Min</div>
                    </div>
                    <div>
                        <div class="stat-value" style="color:#dc2626;">{row['max']:.1f}</div>
                        <div class="stat-label">Max</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Dataset preview ──
    st.markdown("<div class='section-heading'>Dataset Preview (First 5 Rows)</div>", unsafe_allow_html=True)
    st.dataframe(df.head(5), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ──
    st.markdown("<div class='section-heading'>Visualizations</div>", unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2, gap="large")

    # Chart 1: Temperature vs Humidity Scatter
    with chart_col1:
        st.markdown(
            "<div style='font-size:0.92rem;font-weight:600;color:#374151;margin-bottom:10px;'>"
            "Temperature vs Humidity</div>",
            unsafe_allow_html=True,
        )

        condition_colors = {
            "Sunny": "#f59e0b", "Cloudy": "#6b7280",
            "Rainy": "#3b82f6", "Stormy": "#7c3aed", "Foggy": "#9ca3af",
        }

        fig, ax = plt.subplots(figsize=(5.5, 4))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#f8f9fc")

        for cond in CONDITIONS:
            subset = df[df["Condition"] == cond]
            ax.scatter(
                subset["Temperature"], subset["Humidity"],
                label=cond, color=condition_colors.get(cond, "#888"),
                alpha=0.55, s=18, edgecolors="none",
            )

        ax.set_xlabel("Temperature (°C)", fontsize=9, color="#374151")
        ax.set_ylabel("Humidity (%)",      fontsize=9, color="#374151")
        ax.tick_params(colors="#6b7280", labelsize=8)
        ax.legend(fontsize=7.5, framealpha=0.7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#e5e7eb")
        ax.yaxis.grid(True, color="#e5e7eb", linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Chart 2: City-wise average temperature
    with chart_col2:
        st.markdown(
            "<div style='font-size:0.92rem;font-weight:600;color:#374151;margin-bottom:10px;'>"
            "City-wise Average Temperature</div>",
            unsafe_allow_html=True,
        )

        city_avg = (
            df.groupby("City")["Temperature"]
            .mean()
            .sort_values(ascending=True)
        )

        fig, ax = plt.subplots(figsize=(5.5, 4))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#f8f9fc")

        bar_colors = ["#93c5fd" if v < city_avg.mean() else "#2563eb" for v in city_avg.values]
        ax.barh(city_avg.index, city_avg.values, color=bar_colors, edgecolor="none", height=0.6)

        ax.set_xlabel("Avg Temperature (°C)", fontsize=9, color="#374151")
        ax.tick_params(colors="#6b7280", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#e5e7eb")
        ax.xaxis.grid(True, color="#e5e7eb", linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════
# PAGE 3 – ABOUT
# ══════════════════════════════════════════════
elif page == "About":
    st.markdown("<div class='page-title'>About</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='page-subtitle'>Learn more about this project.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="about-card">
        <div style="font-size:1.15rem;font-weight:600;color:#111827;margin-bottom:14px;">
            🌤️ Weather Prediction System
        </div>
        <p style="color:#374151;line-height:1.75;font-size:0.95rem;">
            This application predicts weather conditions based on a few simple inputs
            such as city, season, temperature, humidity, and wind speed.
        </p>
        <p style="color:#374151;line-height:1.75;font-size:0.95rem;">
            It uses a <strong>Random Forest</strong> machine learning model — an algorithm
            that combines multiple decision trees to make reliable predictions.
            The model is trained on a synthetic dataset that reflects realistic
            Indian climate patterns across 10 major cities.
        </p>
        <p style="color:#374151;line-height:1.75;font-size:0.95rem;">
            The goal of this project is to demonstrate how machine learning can be
            applied to everyday problems in a simple, accessible way — without
            requiring complex infrastructure or large datasets.
        </p>
    </div>
    """, unsafe_allow_html=True)
