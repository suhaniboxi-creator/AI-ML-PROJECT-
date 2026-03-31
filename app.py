import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SafeHer — Women Safety Analytics",
    page_icon="🛡️",
    layout="wide"
)
st.markdown("""
<style>
.metric-card {
    background: #f8f9fa;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    border: 1px solid #dee2e6;
}
.risk-high   { background:#ffe8e8; border:2px solid #e53935; border-radius:10px; padding:14px; }
.risk-medium { background:#fff8e1; border:2px solid #fb8c00; border-radius:10px; padding:14px; }
.risk-low    { background:#e8f5e9; border:2px solid #43a047; border-radius:10px; padding:14px; }

.section-header {
    background: linear-gradient(90deg,#880e4f,#c2185b);
    color:white; padding:10px 18px;
    border-radius:8px; margin:10px 0;
    font-size:17px; font-weight:600;
}

button[kind="primary"] {
    background-color:#c2185b !important;
    color:white !important;
    border-radius:8px !important;
    font-weight:600;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# REAL NCRB DATASET LOADER
# Source: Kaggle — Crimes Against Women in India (2001–2021)
# Columns: State, Year, Rape, K&A, DD, AoW, AoM, DV, WT
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_crime_data():
    df = pd.read_csv("CrimesOnWomenData.csv")

    # Normalize state names (dataset uses UPPERCASE for 2001-2010, Title for 2011-2021)
    df['State'] = df['State'].str.title().str.strip()
    df['State'] = df['State'].replace({
        'A & N Islands':  'Andaman & Nicobar',
        'D&N Haveli':     'Dadra & NH',
        'D & N Haveli':   'Dadra & NH',
        'Delhi Ut':       'Delhi',
        'Jammu & Kashmir':'J & K'
    })

    # Rename columns to full descriptive names
    df = df.rename(columns={
        'K&A': 'Kidnapping & Abduction',
        'DD':  'Dowry Deaths',
        'AoW': 'Assault on Women',
        'AoM': 'Assault on Modesty',
        'DV':  'Domestic Violence',
        'WT':  'Women Trafficking'
    })

    # Drop index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Melt to long format: State | Year | Crime_Type | Cases
    crime_cols = ['Rape', 'Kidnapping & Abduction', 'Dowry Deaths',
                  'Assault on Women', 'Assault on Modesty', 'Domestic Violence', 'Women Trafficking']
    df_long = df.melt(id_vars=['State', 'Year'], value_vars=crime_cols,
                      var_name='Crime_Type', value_name='Cases')
    df_long['Cases'] = pd.to_numeric(df_long['Cases'], errors='coerce').fillna(0).astype(int)

    # Focus on major states (exclude UTs with very small numbers)
    exclude = ['Andaman & Nicobar', 'Dadra & NH', 'Daman & Diu',
               'Lakshadweep', 'Puducherry', 'Chandigarh', 'Sikkim',
               'Mizoram', 'Nagaland', 'Manipur', 'Meghalaya', 'Tripura',
               'Arunachal Pradesh', 'Goa', 'Himachal Pradesh']
    df_long = df_long[~df_long['State'].isin(exclude)]

    return df_long

# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC PERSONAL RISK DATASET (ML predictor — no public dataset exists)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def generate_personal_risk_data():
    np.random.seed(0)
    n = 3000
    time_of_day   = np.random.choice(["Morning","Afternoon","Evening","Night"], n, p=[0.25,0.25,0.25,0.25])
    location_type = np.random.choice(["Crowded Public","Isolated Road","Transport","Residential","Market"], n)
    alone         = np.random.choice([0,1], n, p=[0.4,0.6])
    phone_charged = np.random.choice([0,1], n, p=[0.3,0.7])
    known_area    = np.random.choice([0,1], n, p=[0.35,0.65])
    history       = np.random.choice([0,1], n, p=[0.7,0.3])
    lighting      = np.random.choice(["Well Lit","Poorly Lit","Dark"], n, p=[0.4,0.35,0.25])
    cctv          = np.random.choice([0,1], n, p=[0.45,0.55])

    risk_score = np.zeros(n)
    risk_score += np.where(time_of_day=="Night", 3, np.where(time_of_day=="Evening", 1.5, 0))
    risk_score += np.where(location_type=="Isolated Road", 3,
                  np.where(location_type=="Transport", 1.5,
                  np.where(location_type=="Crowded Public", -1, 0)))
    risk_score += alone * 2.5
    risk_score += (1 - phone_charged) * 1.5
    risk_score += (1 - known_area) * 1.5
    risk_score += history * 2
    risk_score += np.where(lighting=="Dark", 2.5, np.where(lighting=="Poorly Lit", 1, 0))
    risk_score += (1 - cctv) * 1.2
    risk_score += np.random.normal(0, 0.8, n)

    risk_label = np.where(risk_score < 3, "Low", np.where(risk_score < 6, "Medium", "High"))

    return pd.DataFrame({
        "Time_of_Day": time_of_day,
        "Location_Type": location_type,
        "Travelling_Alone": alone,
        "Phone_Charged": phone_charged,
        "Familiar_Area": known_area,
        "Past_Incident_Nearby": history,
        "Lighting_Condition": lighting,
        "CCTV_Present": cctv,
        "Risk_Level": risk_label
    })

# ══════════════════════════════════════════════════════════════════════════════
# TRAIN MODEL
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def train_model():
    df = generate_personal_risk_data()
    le_dict = {}
    df_enc = df.copy()
    for col in ["Time_of_Day","Location_Type","Lighting_Condition"]:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df[col])
        le_dict[col] = le
    X = df_enc.drop("Risk_Level", axis=1)
    y = df_enc["Risk_Level"]
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    acc    = accuracy_score(y_test, model.predict(X_test))
    report = classification_report(y_test, model.predict(X_test), output_dict=True)
    cm     = confusion_matrix(y_test, model.predict(X_test), labels=["Low","Medium","High"])
    feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return model, le_dict, acc, report, cm, feat_imp, X.columns.tolist()

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="background:linear-gradient(90deg,#880e4f,#c2185b,#e91e63);
padding:20px 24px;border-radius:12px;margin-bottom:20px;">
<h1 style='color:white;margin:0;font-size:28px;'>🛡️ SafeHer — Women Safety Analytics Dashboard</h1>
<p style='color:#fce4ec;margin:4px 0 0;font-size:14px;'>
Empowering women's safety through smart insights and informed choices.
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='background:#fff3f8;padding:12px;border-radius:10px;margin-bottom:15px;border:1px solid #f8bbd0'>
<b>Purpose of this system:</b> This dashboard analyzes real crime trends against women across India
using NCRB data (2001–2021) and predicts personal safety risk using machine learning,
helping users make safer decisions in real-world situations.
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA & MODEL
# ══════════════════════════════════════════════════════════════════════════════
df_crime = load_crime_data()
model, le_dict, acc, report, cm, feat_imp, feature_cols = train_model()

# ══════════════════════════════════════════════════════════════════════════════
# NAVIGATION TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📊 Crime Analysis Dashboard", "🤖 AI Risk Predictor", "📈 Model Performance"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CRIME ANALYSIS DASHBOARD (Real NCRB Data)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">📊 Crime Against Women — Pattern Analysis (NCRB 2001–2021)</div>', unsafe_allow_html=True)
    st.caption("Source: National Crime Records Bureau (NCRB) | Data covers 21 years across Indian states and UTs")

    # KPI cards
    total   = df_crime["Cases"].sum()
    worst   = df_crime.groupby("State")["Cases"].sum().idxmax()
    top_crime = df_crime.groupby("Crime_Type")["Cases"].sum().idxmax()
    yoy     = (df_crime[df_crime.Year==2021]["Cases"].sum() -
               df_crime[df_crime.Year==2001]["Cases"].sum())

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class="metric-card"><h2>📋</h2>
        <h3>{total:,}</h3><p>Total cases (2001–21)</p></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="metric-card"><h2>📍</h2>
        <h3>{worst}</h3><p>Highest crime state</p></div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="metric-card"><h2>⚠️</h2>
        <h3>{top_crime.split()[0]}</h3><p>Most common crime type</p></div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="metric-card"><h2>📈</h2>
        <h3>+{yoy:,}</h3><p>Rise from 2001 to 2021</p></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Row 1 — Top states bar + Crime type pie
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Top 10 States by Total Cases")
        state_totals = df_crime.groupby("State")["Cases"].sum().nlargest(10).reset_index()
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#c2185b" if i==0 else "#e91e63" if i<3 else "#f48fb1" for i in range(10)]
        bars = ax.barh(state_totals["State"][::-1], state_totals["Cases"][::-1], color=colors[::-1])
        ax.set_xlabel("Total Cases", fontsize=10)
        ax.tick_params(labelsize=9)
        for bar, val in zip(bars, state_totals["Cases"][::-1]):
            ax.text(bar.get_width()+5000, bar.get_y()+bar.get_height()/2,
                    f'{val:,}', va='center', fontsize=8)
        ax.spines[['top','right']].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        st.subheader("Crime Type Distribution")
        ct = df_crime.groupby("Crime_Type")["Cases"].sum().reset_index()
        ct["short"] = ct["Crime_Type"].apply(lambda x: x[:18]+"…" if len(x)>18 else x)
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        colors2 = ["#880e4f","#c2185b","#e91e63","#f06292","#f48fb1","#fce4ec","#ad1457"]
        wedges, texts, autotexts = ax2.pie(
            ct["Cases"], labels=ct["short"], autopct="%1.1f%%",
            colors=colors2, startangle=140,
            textprops={"fontsize": 7}, pctdistance=0.82
        )
        for at in autotexts:
            at.set_fontsize(7)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # Row 2 — Year trend + Heatmap
    c3, c4 = st.columns([2, 3])
    with c3:
        st.subheader("Year-wise Trend (2001–2021)")
        yt = df_crime.groupby("Year")["Cases"].sum().reset_index()
        fig3, ax3 = plt.subplots(figsize=(5, 3.5))
        ax3.plot(yt["Year"], yt["Cases"], color="#c2185b", linewidth=2.5, marker="o", markersize=5)
        ax3.fill_between(yt["Year"], yt["Cases"], alpha=0.15, color="#e91e63")
        ax3.set_xlabel("Year")
        ax3.set_ylabel("Total Cases")
        ax3.tick_params(axis='x', rotation=45, labelsize=7)
        ax3.spines[['top','right']].set_visible(False)
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close()

    with c4:
        st.subheader("State × Crime Type Heatmap")
        top10_states = state_totals["State"].tolist()
        pivot = df_crime[df_crime["State"].isin(top10_states)].groupby(
            ["State","Crime_Type"])["Cases"].sum().unstack(fill_value=0)
        pivot = pivot.reindex(top10_states)
        fig4, ax4 = plt.subplots(figsize=(9, 4.5))
        sns.heatmap(pivot, cmap="RdPu", ax=ax4, linewidths=0.3,
                    cbar_kws={"shrink": 0.7}, annot=False)
        ax4.set_xlabel("")
        ax4.set_ylabel("")
        ax4.tick_params(axis='x', rotation=30, labelsize=7)
        ax4.tick_params(axis='y', labelsize=8)
        fig4.tight_layout()
        st.pyplot(fig4)
        plt.close()

    # Row 3 — Filters
    st.markdown("---")
    st.subheader("🔍 Filter & Explore")
    f1, f2 = st.columns(2)
    with f1:
        sel_state = st.selectbox("Select State", sorted(df_crime["State"].unique()))
    with f2:
        sel_crime = st.selectbox("Select Crime Type", sorted(df_crime["Crime_Type"].unique()))

    fc1, fc2 = st.columns(2)
    with fc1:
        state_df = df_crime[df_crime["State"]==sel_state].groupby("Year")["Cases"].sum().reset_index()
        fig5, ax5 = plt.subplots(figsize=(5, 3))
        ax5.bar(state_df["Year"], state_df["Cases"], color="#e91e63")
        ax5.set_title(f"{sel_state} — All crimes by year", fontsize=10)
        ax5.tick_params(axis='x', rotation=45, labelsize=7)
        ax5.spines[['top','right']].set_visible(False)
        fig5.tight_layout()
        st.pyplot(fig5)
        plt.close()

    with fc2:
        crime_df = df_crime[df_crime["Crime_Type"]==sel_crime].groupby(
            "State")["Cases"].sum().nlargest(8).reset_index()
        fig6, ax6 = plt.subplots(figsize=(5, 3))
        ax6.barh(crime_df["State"][::-1], crime_df["Cases"][::-1], color="#c2185b")
        ax6.set_title(f"{sel_crime[:30]} — Top 8 states", fontsize=10)
        ax6.spines[['top','right']].set_visible(False)
        ax6.tick_params(labelsize=8)
        fig6.tight_layout()
        st.pyplot(fig6)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — AI RISK PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">🤖 AI Personal Safety Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown("Fill in your current situation and the Random Forest model will assess your safety risk.")

    col_a, col_b = st.columns(2)
    with col_a:
        time_of_day   = st.selectbox("⏰ Time of day", ["Morning","Afternoon","Evening","Night"])
        location_type = st.selectbox("📍 Location type", ["Crowded Public","Isolated Road","Transport","Residential","Market"])
        lighting      = st.selectbox("💡 Lighting condition", ["Well Lit","Poorly Lit","Dark"])

    with col_b:
        alone         = st.radio("🚶 Travelling alone?", ["No","Yes"])
        phone_charged = st.radio("🔋 Phone charged (>20%)?", ["Yes","No"])
        known_area    = st.radio("🗺️ Familiar area?", ["Yes","No"])
        cctv          = st.radio("📷 CCTV cameras visible?", ["Yes","No"])
        history       = st.radio("⚠️ Past incidents in area?", ["No","Yes"])

    st.markdown("---")

    if st.button("🔍 Predict My Safety Risk", use_container_width=True):
        inp = {
            "Time_of_Day":        le_dict["Time_of_Day"].transform([time_of_day])[0],
            "Location_Type":      le_dict["Location_Type"].transform([location_type])[0],
            "Travelling_Alone":   1 if alone == "Yes" else 0,
            "Phone_Charged":      1 if phone_charged == "Yes" else 0,
            "Familiar_Area":      1 if known_area == "Yes" else 0,
            "Past_Incident_Nearby": 1 if history == "Yes" else 0,
            "Lighting_Condition": le_dict["Lighting_Condition"].transform([lighting])[0],
            "CCTV_Present":       1 if cctv == "Yes" else 0,
        }

        X_in  = pd.DataFrame([inp])[feature_cols]
        pred  = model.predict(X_in)[0]
        proba = model.predict_proba(X_in)[0]
        classes = model.classes_

        if pred == "High":
            st.markdown("""
            <div class="risk-high">
            <h2>🔴 HIGH RISK</h2>
            <p><b>Your situation shows strong danger indicators.</b></p>
            <ul>
            <li>Move to a crowded and well-lit place immediately</li>
            <li>Call a trusted contact and stay connected</li>
            <li>Be ready to dial emergency number <b>112</b></li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        elif pred == "Medium":
            st.markdown("""
            <div class="risk-medium">
            <h2>🟡 MEDIUM RISK</h2>
            <p><b>Some risk factors detected.</b></p>
            <ul>
            <li>Stay alert and aware of surroundings</li>
            <li>Share your live location with a trusted contact</li>
            <li>Avoid isolated areas</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="risk-low">
            <h2>🟢 LOW RISK</h2>
            <p><b>Situation appears relatively safe.</b></p>
            <ul>
            <li>Stay aware of surroundings</li>
            <li>Keep your phone charged</li>
            <li>Trust your instincts</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        # Probability chart
        st.markdown("#### Confidence Breakdown")
        prob_df = pd.DataFrame({"Risk Level": classes, "Probability": proba}).sort_values("Risk Level")
        bar_colors = {"Low":"#43a047","Medium":"#fb8c00","High":"#e53935"}

        fig_p, ax_p = plt.subplots(figsize=(6, 2))
        for i, row in prob_df.iterrows():
            ax_p.barh(row["Risk Level"], row["Probability"],
                      color=bar_colors.get(row["Risk Level"], "gray"))
            ax_p.text(row["Probability"]+0.01, i,
                      f'{row["Probability"]:.1%}', va='center', fontsize=10)
        ax_p.set_xlim(0, 1.15)
        ax_p.set_xlabel("Confidence")
        ax_p.spines[['top','right']].set_visible(False)
        fig_p.tight_layout()
        st.pyplot(fig_p)
        plt.close()

        with st.expander("📞 Emergency Helplines (India)"):
            st.markdown("""
| Helpline | Number |
|---|---|
| Emergency | **112** |
| Women's Helpline | **1091** |
| Police | **100** |
| Cyber Crime | **1930** |
| Ambulance | **108** |
            """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.info("This section evaluates how accurately the machine learning model predicts safety risk.")
    st.markdown('<div class="section-header">📈 ML Model Performance & Evaluation</div>', unsafe_allow_html=True)
    st.markdown(f"**Model:** Random Forest Classifier | Training: 2400 | Test: 600 | Accuracy: `{acc:.2%}`")

    st.info("""
**How the model works:**
- Uses Random Forest (ensemble machine learning technique)
- Considers 8 personal safety factors as input features
- Predicts risk as Low, Medium, or High
- Trained on domain-informed synthetic data (no public dataset exists for personal safety situations)
""")

    m1, m2 = st.columns(2)
    with m1:
        st.subheader("Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        labels = ["Low","Medium","High"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="RdPu",
                    xticklabels=labels, yticklabels=labels,
                    ax=ax_cm, linewidths=0.5)
        ax_cm.set_xlabel("Predicted", fontsize=11)
        ax_cm.set_ylabel("Actual", fontsize=11)
        fig_cm.tight_layout()
        st.pyplot(fig_cm)
        plt.close()

    with m2:
        st.subheader("Feature Importance")
        feat_labels = {
            "Time_of_Day":         "Time of Day",
            "Location_Type":       "Location Type",
            "Travelling_Alone":    "Travelling Alone",
            "Phone_Charged":       "Phone Charged",
            "Familiar_Area":       "Familiar Area",
            "Past_Incident_Nearby":"Past Incidents Nearby",
            "Lighting_Condition":  "Lighting Condition",
            "CCTV_Present":        "CCTV Present"
        }
        fi_renamed = feat_imp.rename(index=feat_labels)
        fig_fi, ax_fi = plt.subplots(figsize=(5, 4))
        colors_fi = ["#880e4f" if v == fi_renamed.max() else
                     "#e91e63" if v > fi_renamed.mean() else
                     "#f48fb1" for v in fi_renamed.values]
        ax_fi.barh(fi_renamed.index[::-1], fi_renamed.values[::-1], color=colors_fi[::-1])
        ax_fi.set_xlabel("Importance Score")
        ax_fi.spines[['top','right']].set_visible(False)
        ax_fi.tick_params(labelsize=9)
        fig_fi.tight_layout()
        st.pyplot(fig_fi)
        plt.close()

    st.subheader("Per-Class Precision, Recall & F1-Score")
    metrics_rows = []
    for cls in ["Low","Medium","High"]:
        if cls in report:
            r = report[cls]
            metrics_rows.append({
                "Risk Level": cls,
                "Precision":  f"{r['precision']:.2f}",
                "Recall":     f"{r['recall']:.2f}",
                "F1-Score":   f"{r['f1-score']:.2f}",
                "Support":    str(int(r['support']))
            })
    st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True, hide_index=True)

    st.info("""
**What these metrics mean:**
- **Accuracy** — % of predictions the model got right overall
- **Precision** — when model says HIGH risk, how often it's correct
- **Recall** — how many actual HIGH risk cases the model caught
- **F1-Score** — balance of precision and recall (most important for imbalanced classes)
- **Confusion Matrix** — shows which risk levels the model confuses with each other
    """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr>
<div style='text-align:center;color:gray;font-size:13px;'>
<b>SafeHer — Women Safety Analytics System</b><br>
Crime data: National Crime Records Bureau (NCRB) 2001–2021<br>
AI Risk Predictor: Random Forest Classifier<br>
Designed for awareness and educational purposes<br><br>
Emergency: <b>112</b> | Women Helpline: <b>1091</b>
</div>
""", unsafe_allow_html=True)
