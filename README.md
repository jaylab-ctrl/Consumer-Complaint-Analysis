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

    **Feature selection** is crucial in analyzing consumer complaints data, addressing challenges like high
    dimensionality and potential overfitting. We reduced 446 columns to 400 features, improving model
    performance and generalization, using `SelectKBest` from `chi_sq` hypothesis
  
    **NearMiss Sampling**:
    It is an undersampling technique used in machine learning to balance imbalanced datasets by removing majority class samples. It works by selecting majority class instances that are closest to the minority class samples, focusing          on the boundary region between the two classes to preserve important information for classification. It is initialized with `version=1` and `n_neighbors=3`.
    `Version 1` of NearMiss selects majority class samples that have the smallest average distance to the k-nearest
    minority class samples. Setting `n_neighbors=3` means it considers the 3 nearest neighbors when making its
    selection. This process creates a new, balanced dataset by undersampling the majority class while keeping all
    minority class samples. By balancing the dataset, the subsequent machine learning models are less likely to
    be biased towards the majority class, potentially improving overall predictive performance for the consumer
    complaints analysis.

    (insert pic)

    **Model Selection** (tl;dr):
    - XGBoost: High efficiency and strong performance in classification tasks through gradient boosting with advanced regularization.
    - LightGBM: Fast training speed and memory efficiency with histogram-based learning, ideal for large-scale datasets.
    - Logistic Regression: Simple yet effective linear model providing interpretable probability estimates for binary and multiclass classification.
    - Random Forest: Robust ensemble method combining multiple decision trees to reduce overfitting and improve generalization.
    - Decision Tree: Intuitive tree-based model offering clear interpretability and easy visualization of decision-making rules.
  
    **Performance for each of the model shown in the form of confusion matrix**:
     - XgBoost:
       (insert pic)
     - LightGBM:
       (insert pic)
     - Logistc Regression:
       (insert pic)
     - Random Forest:
       (insert pic)
     - Decision Tree:
       (insert pic)

    **Hyper Paramter Tuning**:
     I employed Optuna, an open-source hyperparameter optimization framework, to fine-tune multiple
     machine learning models for the CFPB consumer complaints dataset. This approach automates the search for
     optimal hyperparameters, enhancing model performance and efficiency.

     (insert pic)

     Optimization Process:
     - Define Hyperparameter Search Space: We specified the range of hyperparameters to explore for each model.
     - Create Optuna Study with TPE Sampler:Optuna initializes a study object using the TPE sampler, which employs Bayesian optimization to model the relationship between hyperparameters and model performance. Unlike random or grid              search, TPE learns from past trials to focus computational resources on promising hyperparameter regions.
     - Iterative Trial Process: The optimization follows an adaptive cycle:
          a. Suggest Hyperparameters (Trial n): Optuna's TPE sampler proposes a new set of hyperparameters, Early trials explore the space broadly, Later trials exploit promising regions based on previous results

          b. Train Model with Suggested Parameters: The model is trained using the suggested hyperparameter configuration, Training is performed on the training dataset with cross-validation

          c. Evaluate Performance: The trained model is evaluated on the validation set, Common metrics used: Accuracy, F1-Score, ROC-AUC, Precision-Recall, The performance metric is reported back to Optuna

          d. Update Knowledge: Optuna updates its internal probabilistic model, The TPE algorithm adjusts its understanding of which hyperparameter regions yield better performance, This informs the next trial's hyperparameter suggestions

     - Convergence Check: After each trial, Optuna checks if the maximum number of trials has been reached (typically 50-200 trials). If not, the process loops back to suggest new hyperparameters. If yes, optimization concludes.
     - Return Best Hyperparameters: Upon completion, Optuna returns:
          a. The best hyperparameter configuration found
          b. The corresponding performance score
          c. Complete trial history for analysis
     
    
5) **`CFPB_Complaints_Sentiment_Analysis.ipynb`**  
   - Rule-based (e.g., VADER) and/or ML-based sentiment on `consumer_complaint_narrative`.  
   - Compare distributions by product, company, channel, and geography.  
   - Correlate sentiment with downstream labels (e.g., response type, timely_response).  
   - Visualize **sentiment drift** across time and cohorts.

    In this section of the report, sentiment analysis is conducted on a dataset of consumer complaints to uncover
    patterns and themes in the textual narratives. The analysis begins with preprocessing steps like handling missing
    values and analyzing the distribution of text lengths. Word frequency analysis and visualizations such as word
    clouds and bar charts are employed to highlight recurring terms, followed by sentiment classification using
    FinBERT, a model fine-tuned for financial text, ensuring robust handling of domain-specific language. These
    insights provide a foundation for addressing consumer concerns and enhancing customer service strategies.

    **Word Cloud Analysis**:
    The word cloud serves as an effective exploratory tool, providing a quick overview of key concerns and topics
    within the complaints. By highlighting common terms, it underscores the focus on financial and
    consumer-related matters, such as inaccurate information, payments, and balances. This visualization
    complements detailed quantitative analyses by offering a more accessible summary, making it easier to identify
    central themes that could inform targeted interventions or improvements in services. Despite the presence of
    anonymized data like "xxxx," the word cloud demonstrates that meaningful insights about consumer concerns
    can still be extracted, showcasing the robustness of the preprocessing and sentiment analysis workflow.

    (insert pic)

   **Model Implementation**:
    The sentiment analysis was performed using FinBERT, a financial text-specific model, integrated into a Hugging
    Face pipeline. This setup employs BertForSequenceClassification and BertTokenizer to categorize text into
    positive, neutral, or negative sentiments. The process is optimized for large datasets through batch processing,
    analyzing 32 records at a time. To accommodate FinBERT's 512-token limit, text is truncated, balancing
    efficiency with information preservation.

  (insert pic)

7) **`CFPB_Complaints_Time_Series_Forecasting.ipynb`**  
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
