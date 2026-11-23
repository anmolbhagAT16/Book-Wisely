#  BookWisely – Train Arrival Delay Prediction

BookWisely is a machine learning-based train arrival delay prediction project developed using Python, Pandas, and Scikit-learn.

The project analyzes historical railway data, identifies train routes, and predicts estimated arrival delays at destination stations using three regression algorithms.

The application provides a route-based interface where users can select source and destination stations, explore matching train services, and compare predicted arrival delays from different models.

**Note:** BookWisely provides historical-data-based machine learning estimates. It does not provide live train-running status or guaranteed predictions.

---

## 1. Project Objectives

The main objectives of BookWisely are:

- Analyze historical train-running data using Pandas.
- Perform data cleaning and exploratory data analysis.
- Identify patterns affecting train arrival delays.
- Develop and evaluate three regression models.
- Compare prediction performance using multiple evaluation metrics.
- Identify matching train services between selected stations.
- Provide a user-friendly application for predicting arrival delays.

---

## 2. Dataset Description

The project uses a 10% sample of a larger railway dataset.

The dataset contains information about train services, station sequences, scheduled arrival and departure times, actual running information, and arrival delays.

### Dataset overview

| Description | Details |
|---|---:|
| Route-stop records prepared | 607,069 |
| Distinct station codes in route data | 8,475 |
| Distinct train categories | 20 |
| Modeling observations | 398,575 |
| Modeling train runs | 32,121 |
| Target variable | arrival_delay_min |

The raw dataset and large intermediate files are maintained locally and are not included in the GitHub repository.

### Important data preparation steps

- Selected completed train runs.
- Excluded origin-station records from arrival-delay modeling because their arrival information is naturally unavailable.
- Examined duplicate records and missing values.
- Investigated extremely early arrival records and corrected suspected scheduled-arrival date inconsistencies based on station-sequence evidence.
- Prepared the cleaned modeling dataset for regression analysis.

---

## 3. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the distribution of train arrival delays and identify useful patterns.

The analysis included:

- Arrival delay summary statistics.
- Arrival delay distribution and histograms.
- Delay analysis by train category.
- Scheduled arrival hour analysis using Indian Standard Time (IST).
- Station-wise arrival delay analysis.
- Delay changes between consecutive stations.
- Analysis of arrival delay progression across station sequences.

The EDA findings indicated that arrival delays are right-skewed, with a smaller number of observations experiencing considerably larger delays.

---

## 4. Machine Learning Methodology

### Prediction Objective

The objective is to predict train arrival delay at a particular destination station.

**Target variable:** `arrival_delay_min`

Since arrival delay is a continuous numerical value measured in minutes, the project is treated as a regression problem.

### Selected Features

| Feature | Description |
|---|---|
| category | Train service category |
| seq | Destination station sequence |
| station_code | Destination station code |
| scheduled_arrival_hour | Scheduled arrival hour in IST |

Actual arrival and current target-delay information are not used as prediction inputs.

### Train-Test Split

A chronological train-test split was used to evaluate model performance on later train runs.

| Dataset | Observations | Train runs |
|---|---:|---:|
| Training | 312,577 | 25,062 |
| Testing | 85,998 | 7,059 |

Training data ends on 10 November 2026, while testing data begins on 11 November 2026.

Train runs are kept together to prevent observations from the same run appearing in both datasets.

### Feature Preprocessing

Categorical features are converted into numerical representations using encoding techniques.

- One-Hot Encoding is used for Linear Regression and Random Forest.
- Ordinal Encoding is used within the HistGradientBoosting pipeline.

The preprocessing encoder is fitted using training data only.

---

## 5. Models Evaluated

Three regression algorithms were evaluated.

### 1. Linear Regression

A regression algorithm that models the relationship between input features and the continuous target variable.

### 2. Random Forest Regressor

An ensemble learning algorithm that combines predictions from multiple decision trees.

### 3. HistGradientBoosting Regressor

A gradient boosting algorithm that builds decision trees sequentially to improve prediction performance.

A Dummy Regressor using the median strategy was also implemented as a baseline benchmark. It is not counted as one of the three main models.

---

## 6. Model Evaluation Results

The models were evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score.

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Dummy Baseline | 33.06 | 76.34 | -0.0943 |
| Linear Regression | **31.67** | **63.19** | **0.2503** |
| Random Forest | 37.85 | 70.86 | 0.0573 |
| HistGradientBoosting | 33.24 | 64.51 | 0.2185 |

Linear Regression achieved the lowest MAE and RMSE and the highest R² score among the three regression models.

Its average absolute prediction error is approximately 31.67 minutes.

### Prediction Performance Within Time Tolerances

The percentage of predictions falling within different time tolerances was also calculated.

| Model | ±5 min | ±10 min | ±15 min | ±30 min | ±60 min |
|---|---:|---:|---:|---:|---:|
| Linear Regression | 18.58% | 34.48% | 46.68% | 68.81% | 87.26% |
| Random Forest | 6.41% | 13.29% | 21.20% | 58.84% | 89.98% |
| HistGradientBoosting | 15.61% | 30.67% | 41.93% | 67.14% | 87.43% |
| Dummy Baseline | 20.34% | 40.43% | 59.86% | 76.14% | 87.01% |

Linear Regression remains the best model according to MAE, RMSE, and R², although the Dummy Baseline and Random Forest performed better at certain prediction tolerances.

---

## 7. Final Model Selection

Linear Regression was selected as the best-performing regression model based on the primary regression evaluation metrics.

However, its R² score of 0.2503 indicates that considerable variation in arrival delays remains unexplained.

Further improvements may be possible through additional feature engineering, historical delay information, better categorical representations, and model optimization.

---

## 8. BookWisely Application

The project includes a Streamlit application for route-based arrival delay prediction.

### Application Features

- Searchable source station dropdown.
- Searchable destination station dropdown.
- Train category selection with category descriptions.
- Automatic destination station sequence identification.
- Automatic scheduled arrival hour identification.
- Multiple matching train services.
- Prediction using Linear Regression.
- Prediction using Random Forest.
- Prediction using HistGradientBoosting.
- Option to compare all three models.
- Historical route record information.

### Route Identification

The route lookup data is prepared using historical completed train runs.

A train service is considered a matching route when:

1. The source station is present in the train run.
2. The destination station is present in the same train run.
3. The destination station sequence is greater than the source station sequence.

Matching train services are grouped by train number, train name, and category.

The application displays the number of distinct historical start dates represented by the matching service records.

The destination sequence and scheduled arrival hour are used as inputs to the prediction models.

**Important:** Historical routes are derived from the available 10% sample. They should not be interpreted as a complete or current railway timetable. Predictions do not use live source-station delay information.

---

## 9. Project Structure

```text
MLProject/
│
├── Data/
│   ├── raw/
│   │
│   ├── interim/
│   │   └── railway_sample_10_percent.parquet
│   │
│   └── processed/
│       ├── modeling_data_clean.parquet
│       └── route_stops.parquet
│
├── Models/
│   └── bookwisely_all_models.joblib
│
├── Notebooks/
│   └── main.ipynb
│
├── Output/
│   ├── figures/
│   └── reports/
│
├── src/
│   └── app.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

The original raw dataset, 10% sample, and cleaned modeling Parquet file are maintained locally and excluded from GitHub.

The saved model bundle and route-stop lookup file are required to run the application.

---

## 10. Installation and Setup

### Step 1: Clone the repository

```bash
git clone <repository-url>
```

### Step 2: Open the project directory

```bash
cd MLProject
```

### Step 3: Create a virtual environment

```bash
python -m venv .venv
```

### Step 4: Activate the environment

For Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 5: Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 11. Running the Application

Ensure the following files are available:

- `Models/bookwisely_all_models.joblib`
- `Data/processed/route_stops.parquet`

Run the following command from the project root directory:

```bash
python -m streamlit run src/app.py
```

The BookWisely application will open in your browser.

---

## 12. Running the Jupyter Notebook

To explore the machine learning workflow:

```bash
jupyter notebook
```

Open:

```text
Notebooks/main.ipynb
```

To rerun the complete notebook, place the original 10% sample at:

```text
Data/interim/railway_sample_10_percent.parquet
```

This dataset is intentionally excluded from GitHub because of its size.

---

## 13. Limitations and Future Improvements

### Current Limitations

- The project uses a 10% sample rather than the complete railway dataset.
- Route coverage is limited to the historical records available in the sample.
- The application does not provide live train-running status.
- The prediction uses only four selected input features.
- Linear Regression does not capture all patterns affecting large arrival delays.
- HistGradientBoosting uses ordinal encoding for categorical features, which may introduce artificial ordering.

### Future Improvements

- Include more relevant train and station features.
- Incorporate historical train-delay information without data leakage.
- Explore improved categorical encoding techniques.
- Optimize model parameters.
- Improve large-delay prediction performance.
- Integrate reliable live railway information if an appropriate data source becomes available.

---

## 14. Conclusion

BookWisely demonstrates the application of data analysis and machine learning techniques to historical railway arrival delay prediction.

The project includes data preparation, exploratory data analysis, three regression algorithms, model evaluation, route identification, and a Streamlit-based prediction interface.

Linear Regression achieved the strongest overall performance according to MAE, RMSE, and R² among the evaluated regression models.

The project provides a foundation for further development of railway delay prediction systems using additional historical and operational information.
