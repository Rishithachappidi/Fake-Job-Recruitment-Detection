# Fake Job Recruitment Detection Using Machine Learning

A machine-learning project for identifying potentially fraudulent online job advertisements from structured job-posting information. The project explores data preprocessing, feature engineering, encoding, scaling, train/test splitting, and comparison of **Logistic Regression, Linear SVM, and Random Forest** classifiers.

## Project Objective

The objective is to classify job advertisements into two categories:

- **Genuine** (`f` / `0`)
- **Fraudulent** (`t` / `1`)

The notebook also includes exploratory analysis and visual outputs generated during the project workflow.

## Dataset

The repository contains **one dataset only**:

`data/Nextera_dataset.csv`

It contains **20,434 rows and 18 columns**.

### Target distribution

The `fraudulent` column contains:

| Label | Meaning | Rows |
|---|---|---:|
| `f` | Genuine job posting | 16,994 |
| `t` | Fraudulent job posting | 866 |
| Missing | No target label supplied | 2,574 |
| **Total** | | **20,434** |

The 2,574 rows without a target label are not usable as supervised-learning examples unless labels are obtained separately. The notebook removes rows with a missing `fraudulent` value before model training.

### Dataset columns

| Column | Description |
|---|---|
| `title` | Job title or position name |
| `location` | Job location |
| `department` | Department associated with the position |
| `salary_range` | Advertised salary range |
| `company_profile` | Information provided about the company |
| `description` | Main job advertisement description |
| `requirements` | Requirements or qualifications for the job |
| `benefits` | Benefits offered by the employer |
| `telecommuting` | Indicates whether remote/telecommuting is supported |
| `has_company_logo` | Indicates whether a company logo is present |
| `has_questions` | Indicates whether screening questions are included |
| `employment_type` | Employment type, such as full-time or part-time |
| `required_experience` | Experience level requested |
| `required_education` | Education level requested |
| `industry` | Industry associated with the job |
| `function` | Functional area of the job |
| `fraudulent` | Target label: `f` = genuine, `t` = fraudulent |
| `in_balanced_dataset` | Indicates membership in the balanced subset supplied with the data |

### Missing values

The original dataset contains missing values in several fields. The largest missing-value counts include:

- `salary_range`: 17,510
- `department`: 14,022
- `required_education`: 10,668
- `benefits`: 9,750
- `required_experience`: 9,605
- `function`: 9,024
- `industry`: 7,470

The notebook handles missing values differently depending on the feature type rather than simply deleting every incomplete row.

## Project Workflow

The notebook follows this general process:

```text
Dataset
   ↓
Exploratory Data Analysis / Visualization
   ↓
Missing-Value Inspection
   ↓
Missing-Value Handling
   ↓
Duplicate Inspection and Removal
   ↓
Feature Engineering
   ↓
Categorical Encoding
   ↓
Numerical Feature Scaling
   ↓
Remove Rows Without Target Labels
   ↓
Train/Test Split (80/20, stratified)
   ↓
Model Training
   ├── Logistic Regression
   ├── Linear SVM
   └── Random Forest
   ↓
Model Evaluation
   ├── Accuracy
   ├── Precision
   ├── Recall
   └── F1 Score
   ↓
Comparison / Visual Results
```

### 1. Exploratory analysis

The notebook begins with visualization and inspection of the job-posting data to understand the dataset and its class distribution.

### 2. Missing-value handling

Text columns are filled with an empty string when missing. Categorical columns are filled with `Unknown`, while `salary_range` is also filled with `Unknown`.

### 3. Duplicate handling

Duplicate rows are checked and removed using `drop_duplicates()`.

### 4. Feature engineering

The notebook creates numerical features based on the amount of text in a job advertisement:

- `title_length`
- `description_length`
- `requirements_length`
- `benefits_length`
- `total_text_length`

These features provide a simple representation of the amount of information contained in different parts of a job posting.

### 5. Encoding

Categorical variables are converted into numerical features using one-hot encoding with `pandas.get_dummies(..., drop_first=True)`.

### 6. Scaling

The engineered numerical text-length features are standardized using `StandardScaler`. In the final model section, the scaler is fitted on the training data and then applied to the test data.

### 7. Target preparation

The target is converted from the original labels:

```text
f → 0 → Genuine
 t → 1 → Fraudulent
```

Rows with missing target labels are removed before supervised training.

### 8. Train/test split

The final model section uses an **80/20 train/test split** with `random_state=42` and stratification so that the class proportions are preserved as closely as possible.

### 9. Models

#### Logistic Regression

A linear baseline classifier using class balancing.

#### Linear SVM

A linear support-vector classifier using class balancing.

#### Random Forest

An ensemble of decision trees using 200 estimators and class balancing.

### 10. Evaluation

The notebook calculates:

- Accuracy
- Precision
- Recall
- F1 score
- Classification reports
- Model comparison outputs
- Confusion-matrix/visual outputs where generated by the notebook

## Results

All image outputs generated from the supplied notebook are stored in:

`results/figures/`

This folder contains the extracted figures and plots that were already embedded in the executed notebook.

## Reference Paper

The project includes the reference paper:

**Naudé, M., Adebayo, K. J., & Nanda, R. (2023). _A machine learning approach to detecting fraudulent job types_. AI & Society, 38, 1013–1024.**

DOI: **10.1007/s00146-022-01469-0**

- [Reference paper PDF in this repository](reference/reference_paper.pdf)
- [Publisher/DOI page](https://doi.org/10.1007/s00146-022-01469-0)

The paper investigates machine-learning approaches for identifying types of fraudulent job advertisements and discusses lexical, syntactic, semantic, and contextual features.

## Repository Structure

```text
Nextera-Fake-Job-Recruitment-Detection/
│
├── data/
│   └── Nextera_dataset.csv
│
├── notebooks/
│   └── Nextera_project_phase.ipynb
│
├── reference/
│   └── reference_paper.pdf
│
├── results/
│   └── figures/
│       ├── figure_01_cell3.png
│       ├── figure_02_cell4.png
│       ├── ...
│       └── figure_14_cell73.png
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Important Reproducibility Note

This repository reflects the supplied project notebook. It should not be treated as a perfectly cleaned production ML pipeline yet. In particular, the notebook contains development-stage/repeated preprocessing and modeling sections, and some earlier preprocessing operations occur before the final train/test split. Therefore, the figures and model outputs should be understood as outputs of the supplied notebook rather than as independently reproduced benchmark results.

For a stronger final version, preprocessing should be fitted only on training data using a reproducible `Pipeline`/`ColumnTransformer`, and text features such as TF-IDF could be evaluated alongside the current structured features.

## Requirements

Install the dependencies listed in `requirements.txt` and open the notebook:

```bash
pip install -r requirements.txt
jupyter notebook
```

Then open:

```text
notebooks/Nextera_project_phase.ipynb
```

## Project Files

- **Dataset:** `data/Nextera_dataset.csv`
- **Main notebook:** `notebooks/Nextera_project_phase.ipynb`
- **Reference paper:** `reference/reference_paper.pdf`
- **Generated figures:** `results/figures/`
