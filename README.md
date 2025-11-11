# CFPB-Complaints Analytics Suite
Capstone-Style Project on the Consumer Financial Protection Bureau (CFPB) Consumer Complaint Database

---

## 🔎 Executive Summary
**Data:** Public CFPB Consumer Complaint Database with free-text complaint narratives and structured fields (product, issue, company, state, dates, response).  
**Challenge:** Unstructured narratives, evolving label definitions, class imbalance, and calendar effects.  
**Solution:** A four-notebook workflow that explores the data (EDA), builds a **response classification** model, performs **sentiment analysis** on narratives, and **forecasts complaint volume** with time-series methods.  
**Highlights**
- End-to-end, notebook-first workflow for analysis → modeling → evaluation → forecasting.
- Text pipeline centered on complaint narratives; balanced training batches and stratified splits where applicable.
- Clear outputs: confusion matrices, feature importances, sentiment distributions, and monthly forecasts with confidence intervals.

---

## 📦 Data & Taxonomy
**Source:** CFPB Consumer Complaint Database (public).  
**Core fields used (typical):**  
`complaint_id`, `date_received`, `product`, `issue`, `sub_issue`, `consumer_complaint_narrative`, `company`, `state`, `submitted_via`, `date_sent_to_company`, `company_response_to_consumer`, `timely_response`, `consumer_disputed`.  
**Targets (illustrative):**
- **Response Classification:** `company_response_to_consumer` (multi-class; e.g., *Closed with explanation*, *Closed with monetary relief*, *In progress*).
- **Sentiment Analysis:** polarity score over `consumer_complaint_narrative` (e.g., negative/neutral/positive).
- **Time Series:** monthly complaint counts (overall and/or by product/state).  


---

## 🧪 Notebooks & What They Do
1) **`CFPB_Complaints_EDA.ipynb`**  
   - Load and clean data; sanity checks and missingness.  
   - Distribution plots for *product, issue, state, company*.  
   - Narrative length stats; word clouds / top n-grams (optional).  
   - Calendar views: monthly trend, seasonality, and outliers.

2) **`CFPB_Complaints_Response_Classification.ipynb`**  
   - Text featurization (e.g., TF–IDF over unigrams/bigrams; optional Transformer embeddings).  
   - Baselines (Logistic Regression / Linear SVM) with class weights; optional Gradient Boosting.  
   - Train/validation/test with **stratified** splits; hyperparameter search (grid or randomized).  
   - **Metrics:** Accuracy, macro-Precision/Recall/F1; per-class confusion matrices and classification report.  
   - Export predictions and model artifacts (vectorizer + model) for reuse.

3) **`CFPB_Complaints_Sentiment_Analysis.ipynb`**  
   - Rule-based (e.g., VADER) and/or ML-based sentiment on `consumer_complaint_narrative`.  
   - Compare distributions by product, company, channel, and geography.  
   - Correlate sentiment with downstream labels (e.g., response type, timely_response).  
   - Visualize **sentiment drift** across time and cohorts.

4) **`CFPB_Complaints_Time_Series_Forecasting.ipynb`**  
   - Aggregate to monthly counts; handle reporting lags and missing months.  
   - Benchmarks: naïve seasonal, moving averages; models: SARIMA/Prophet (pick one or both).  
   - Backtests with expanding or rolling windows; prediction intervals.  
   - Scenario views: overall, by product, and top states/companies.

---

## 🧰 Modeling Details
**Text preprocessing:** lowercasing, punctuation/stopword handling, min-df thresholds; optional lemmatization.  
**Vectorization:** TF–IDF (uni/bi-grams), optional Transformer embeddings for advanced variants.  
**Class imbalance:** class-weighted losses and/or minority oversampling in training folds.  
**Validation:** stratified k-fold or train/val/test split with fixed random seeds.  
**Explainability (optional):** linear model coefficients, permutation importances, and example reviews near decision boundaries.

---

## 📊 Evaluation
**Classification:** Accuracy, macro-F1, per-class metrics, confusion matrix, ROC/PR curves for one-vs-rest (optional).  
**Sentiment:** distribution summaries, stability across time, agreement between rule-based and ML variants.  
**Forecasting:** MAE/MAPE/RMSE on backtests; coverage of prediction intervals; seasonal decomposition checks.

> Tip: lock a **held-out test month** for time-series (no leakage), and hold out **recent months** to mimic production.

---

## ⚙️ Setup
**Python:** 3.9+ recommended • **GPU:** optional (for Transformer variants)

```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly nltk textblob statsmodels prophet==1.1.*
python -c "import nltk; nltk.download('punkt'); nltk.download('vader_lexicon')"
```

- If using Prophet on some systems: `pip install prophet` may require a recent `pystan`/`cmdstanpy` stack.  
- If using Transformer embeddings: also `pip install transformers torch accelerate` and ensure a compatible PyTorch build.

---

## 🚀 Usage

### Open notebooks
```bash
jupyter lab
# or
jupyter notebook
```

### Typical flows
- **EDA:** run sections top-to-bottom to profile data health and distributions.  
- **Response Classification:** edit the *label set* and TF–IDF parameters; run training, then evaluate on a held-out set.  
- **Sentiment:** choose VADER or a model-based approach; compare across cohorts and time.  
- **Forecasting:** pick aggregation level (overall/product/state), run model fit + backtests, export future-month forecasts.
