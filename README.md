# CFPB- Consumer Complaints Analytics 
Project on the Consumer Financial Protection Bureau (CFPB) Consumer Complaint Database

---

## 🔎 Executive Summary
**Problem Statement:** The financial services industry has been receiving complaints from consumers regarding various issues such as
fraudulent transactions, poor customer service, and unresponsive companies.
These complaints, collected in the CFPB Consumer Complaint Database, need to be analyzed to identify
patterns and trends that can inform better regulatory decisions and provide companies with actionable insights.
- Problem:
How can we leverage data analytics and machine learning to identify trends and patterns in consumer financial
complaints to improve regulatory oversight and enhance consumer protection?
This problem is significant for policymakers, financial institutions, consumer rights advocates, and consumers
themselves. CFPB regulators can use the analysis to monitor the performance of financial institutions, while
companies can identify areas to improve customer service. The sheer volume of structured and unstructured
data, including the complaint narratives, product categories, and company responses, makes it a task best suited
for data analytics. Machine learning models can help categorize complaints, predict future trends, and provide
actionable insights from both structured and text data.


---

## 📦 Data & Taxonomy
**Data:** I am using the Consumer Complaint Database from the CFPB, which is updated daily and
includes complaints related to various financial products. We are using the data from 01/01/2023 to 10/04/2024.
- Data Attributes:
1. Date CFPB received the complaint (Date)
2. Product/sub-product: The financial product the consumer identified in the complaint (Categorical)
3. Issue/sub-issue: The specific issue the consumer reported (Categorical)
4. State: State of residence of the consumer (Categorical)
5. ZIP code: ZIP code provided by the consumer (Categorical)
6. Company name: Company the complaint is about (Categorical)
7. Timely response: Whether the company provided a timely response (Boolean)
8. Company response: How the company responded to the complaint (Categorical)
9. Public company response: Optional public response from the company (Text)
10. Date complaint sent to company (Date)
11. Consumer consent to publish narrative: Whether the consumer consented to publishing their
narrative (Boolean)
12. Submission method: How the complaint was submitted to CFPB (Categorical)
13. Tags: Additional tags for categorization (Categorical)

- Dataset Statistics:
  - Number of records: ~6 million complaints, but reduced to 3.1 million complaints, because of storage
constraints.
  - Number of features: 13 attributes, but after one-hot-encoding its increased to 455 features.
  - Data source: Consumer Complaint Database (publicly available data.
    
**Targets:**
- **Response Classification:** `timely_response` (binary; yes or no).
- **Sentiment Analysis:** polarity tone over `consumer_complaint_narrative` (e.g., negative/neutral/positive).
- **Time Series:** monthly complaint counts (overall and/or by product/state).  


---

## 🧪 Notebooks & What They Do
1) **`CFPB_Complaints_EDA.ipynb`**  
   - Load and clean data; sanity checks and missingness.  
   - Distribution plots for *product, issue, state, company*.  
   - Narrative length stats; word clouds.  
   - Calendar views: monthly trend, seasonality, and outliers.
  
   Preprocessing Steps
   Data Cleaning:
    - Since the dataset was pretty large, like approx. 6 million complaints, we had to chunk our data into
      400,000 instances for each chunk. Each chunk (400000, 18) was loaded into the pandas DataFrame and
      then concatenated row-wise.
    - There were 16 categorical columns and 2 numerical features, and these categorical features were
      converted to “category,” and the date features were converted to “date-time.”
    - We calculated the “NaN” values and found out there were 4 features where “NaN” values had more than
      50%. These are subsequently dropped. There were also some features where we had to add a category
      because there were no values at all.

   Visualizations:
    - Yearly complaint count (insert pic)
    - Monthly complaint count (insert pic)
    - Top 10 compalaint by products (insert pic)
    - Complaints by state (insert pic)
    - Company response (insert pic)
    - Word Cloud (insert pic)
    - Top Complaint by companies (insert pic)

3) **`CFPB_Complaints_Response_Classification.ipynb`**  
   - Text featurization (e.g. One-hot encoding, Standardization of data, data resampling, feature selection)
   - Baselines (XGBoost, 
   - Train/test splits; hyperparameter search using Optuna.  
   - Metrics:Accuracy, Precision/Recall/F1; per-class confusion matrices and classification report.  

  In this section, we analyze the Consumer Financial Protection Bureau (CFPB) complaints data to predict whether
  a financial institution will provide a timely response to a consumer complaint. Our analysis focuses on
  developing a classification model to forecast the likelihood of a timely response, which is crucial for
  understanding and improving customer service in the financial sector. By leveraging various features from the
  complaint data, such as the type of financial product, the company involved, and the nature of the complaint, we
  aim to create a predictive model that can help identify factors influencing response times and potentially assist
  both consumers and financial institutions in managing expectations and improving complaint resolution
  processes.

4) **`CFPB_Complaints_Sentiment_Analysis.ipynb`**  
   - Rule-based (e.g., VADER) and/or ML-based sentiment on `consumer_complaint_narrative`.  
   - Compare distributions by product, company, channel, and geography.  
   - Correlate sentiment with downstream labels (e.g., response type, timely_response).  
   - Visualize **sentiment drift** across time and cohorts.

5) **`CFPB_Complaints_Time_Series_Forecasting.ipynb`**  
   - Aggregate to monthly counts; handle reporting lags and missing months.  
   - Benchmarks: naïve seasonal, moving averages; models: SARIMA/Prophet (pick one or both).  
   - Backtests with expanding or rolling windows; prediction intervals.  
   - Scenario views: overall, by product, and top states/companies.

---

## 🧰 Modeling Details
**Text preprocessing:** lowercasing, punctuation/stopword handling, optional lemmatization.  
**Vectorization:** TF–IDF (uni/bi-grams), optional Transformer embeddings for advanced variants.  
**Class imbalance:** class-weighted losses and/or minority oversampling in training folds.  
**Validation:** stratified k-fold or train/val/test split with fixed random seeds.  
**Explainability (optional):** linear model coefficients, permutation importances, and example reviews near decision boundaries.

---

## 📊 Evaluation
**Classification:** Accuracy, precison, recall, f1-score (all metrics based on confusion matrix).
**Sentiment:** sentiment of the complaint (positive/negative/neutral).
**Forecasting:** coverage of prediction intervals; seasonal decomposition checks.

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
