# AI-ML-PROJECT-
# 🛡️ SafeHer — Women Safety Analytics Dashboard

> Built to make 21 years of NCRB crime data actually useful — and to give
> women a real-time tool to assess personal safety risk.

SafeHer is an interactive Streamlit dashboard that does two things:
analyses real government crime data across India (2001–2021), and uses
a trained ML model to predict your personal safety risk based on your
current situation.

---

## What all can be done by it?

- **Crime Analysis Dashboard** — 21 years of NCRB data across 7 crime
  categories and 20+ Indian states. State comparisons, year-wise trends,
  crime type breakdowns, and an interactive heatmap.

- **AI Risk Predictor** — Describe your situation (time, location,
  lighting, whether you're alone) and get an instant Low / Medium / High
  risk classification from a trained Random Forest model.

- **Model Performance Tab** — Confusion matrix, feature importance, and
  per-class metrics so you know exactly how the model works and where it
  might be wrong.

---

## Tech stack

| Library | Used for |
|---|---|
| `streamlit` | Web app framework |
| `scikit-learn` | Random Forest classifier, label encoding, metrics |
| `pandas` | Data loading and cleaning |
| `numpy` | Numerical operations |
| `matplotlib` | Charts and visualizations |
| `seaborn` | Heatmaps |

---

## Project structure
```
safeher/
├── app.py                  # Everything — data, ML, and UI in one file
├── CrimesOnWomenData.csv   # Real NCRB dataset (2001–2021)
├── requirements.txt        # Python dependencies
└── README.md               # You're reading it
```

> ⚠️ `CrimesOnWomenData.csv` must be in the **same folder** as `app.py`.

---

## Setup

**Prerequisites:** Python 3.8+, pip
```bash
# 1. Clone the repo
git clone https://github.com/your-username/safeher.git
cd safeher

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the app
```bash
streamlit run app.py
```

Opens at **http://localhost:8501** automatically.

---

## How to use it

**Tab 1 — Crime Analysis**
- KPI cards at the top give you the headline numbers at a glance
- Scroll down for bar charts, pie chart, 21-year trend line, and a
  state × crime heatmap
- Use the Filter & Explore dropdowns to dig into any state or crime type

**Tab 2 — AI Risk Predictor**
1. Set your situation using the dropdowns: time of day, location type,
   lighting, whether you're alone, phone charge, familiarity with the
   area, CCTV visibility, past incidents nearby
2. Hit **"🔍 Predict My Safety Risk"**
3. Get a color-coded result (🔴 High / 🟡 Medium / 🟢 Low) with
   actionable advice
4. A confidence chart shows how certain the model is
5. Expand the **📞 Emergency Helplines** section for India's emergency
   numbers

**Tab 3 — Model Performance**
- Confusion matrix, feature importance chart, and a per-class metrics
  table with plain-English explanations

---

## Data source

| | |
|---|---|
| **Dataset** | Crimes Against Women in India |
| **Source** | NCRB via Kaggle |
| **Years** | 2001–2021 |
| **States** | 20+ Indian states and UTs |
| **Crime types** | Rape, Kidnapping & Abduction, Dowry Deaths, Assault on Women, Assault on Modesty, Domestic Violence, Women Trafficking |

---

## Model details

| | |
|---|---|
| Algorithm | Random Forest Classifier |
| Training samples | 2,400 |
| Test samples | 600 |
| Overall accuracy | ~80.33% |
| High-risk F1 | 0.89 |
| Class balancing | `class_weight="balanced"` |

**Features used:** Time of Day, Location Type, Travelling Alone, Phone
Charged, Familiar Area, Past Incidents Nearby, Lighting Condition, CCTV
Present

**Top predictors (by importance):**
1. Location Type
2. Time of Day
3. Travelling Alone
4. Lighting Condition

> **Note:** The risk predictor is trained on domain-informed synthetic
> data — no public dataset exists for personal situational safety
> assessments.

---

## Emergency helplines

| Service | Number |
|---|---|
| Emergency | **112** |
| Women's Helpline | **1091** |
| Police | **100** |
| Cyber Crime | **1930** |
| Ambulance | **108** |

---

## Disclaimer

For awareness and educational purposes only. Crime data is from NCRB via
Kaggle. The AI risk predictor uses a simulation model — it is not a
substitute for professional safety advice or emergency services.

---

*SafeHer — NCRB data 2001–2021 | Random Forest Classifier | Built for awareness*
