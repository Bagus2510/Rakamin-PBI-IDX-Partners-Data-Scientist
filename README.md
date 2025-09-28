# 🏦 CREDIT RISK PREDICTION MODEL

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-red.svg)](https://xgboost.readthedocs.io/)

A comprehensive machine learning project for predicting loan approval decisions using advanced data science techniques. This project analyzes 466,285+ loan records from 2007-2014 to build a robust credit risk assessment model.

## 📊 Project Overview

This project develops an automated credit risk prediction system that helps financial institutions make data-driven loan approval decisions. Using advanced machine learning algorithms and comprehensive feature engineering, the model achieves high accuracy in distinguishing between approved and defaulted loans.

### 🎯 Key Objectives
- **Analyze** large-scale loan dataset with comprehensive EDA
- **Engineer** meaningful features from raw financial data  
- **Develop** and compare multiple ML algorithms
- **Optimize** model performance through hyperparameter tuning
- **Evaluate** model reliability using professional metrics

## 🗂️ Project Structure

```
├── predictive_model_loan.ipynb    # Main analysis notebook
├── loan_data_2007_2014.csv       # Primary dataset
├── LCDataDictionary.xlsx         # Data dictionary reference
├── README.md                      # Project documentation
└── requirements.txt               # Python dependencies
```

## 🛠️ Technical Architecture

### 1. **Data Understanding**
- **Dataset**: 466,285 loan records with 74+ features
- **Time Period**: 2007-2014 lending data
- **Target Variable**: Binary loan approval classification
- **Data Quality**: Comprehensive missing value and outlier analysis

### 2. **Feature Engineering**
#### 📅 Date Features (6 new features)
```python
- credit_history_length      # Duration of credit history in years
- days_since_last_payment   # Days since last payment received
- issue_year                # Year when loan was issued
- issue_month              # Month when loan was issued  
- issue_season             # Season when loan was issued
```

#### 💰 Financial Ratios (7 new features)
```python
- loan_to_income_ratio         # Loan burden relative to income
- installment_to_income_ratio  # Monthly payment burden
- total_debt_service_ratio     # Total debt burden
- credit_util_ratio           # Credit utilization rate
- income_per_inquiry          # Income per credit inquiry
- account_diversity           # Variety of credit accounts
- risk_score                  # Composite risk indicator
```

### 3. **Model Development**
```python
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(),           # 🏆 Best Performance
    'LightGBM': LGBMClassifier()
}
```

### 4. **Data Preprocessing**
- **Missing Value Imputation**: Strategic median/mode imputation
- **Outlier Handling**: IQR-based detection and capping
- **Feature Scaling**: StandardScaler and MinMaxScaler
- **Encoding Strategy**: Label encoding for high-cardinality, One-Hot for low-cardinality
- **Class Balancing**: SMOTE oversampling technique

## 📈 Model Performance

### 🏆 Best Model: XGBoost Classifier

| Metric | Score |
|--------|--------|
| **Accuracy** | 99.91% |
| **ROC-AUC** | 0.9991 |
| **Precision** | 99.92% |
| **Recall** | 99.89% |
| **F1-Score** | 99.91% |

### 📊 Cross-Validation Results
- **5-Fold CV Mean**: 0.9987
- **CV Standard Deviation**: ±0.0003
- **Stability Score**: Excellent consistency

## 🔍 Key Insights

### 💡 Data Analysis Findings
- **Class Distribution**: 87.9% approved vs 12.1% not approved loans
- **Dominant Loan Purpose**: Debt consolidation (60%+)
- **Interest Rate Impact**: Higher rates strongly correlate with defaults
- **Income Distribution**: Most borrowers earn $40K-$100K annually

### 🏆 Top Predictive Features
1. **Interest Rate** - Primary risk indicator
2. **DTI Ratio** - Debt-to-income relationship
3. **Loan Grade** - Credit risk assessment
4. **Loan-to-Income Ratio** - Engineered feature
5. **Credit Utilization** - Available credit usage
6. **Annual Income** - Borrower's earning capacity
7. **Employment Length** - Job stability indicator
8. **Credit History Length** - Experience with credit
9. **Revolving Balance** - Outstanding credit debt
10. **Number of Accounts** - Credit portfolio diversity

## 🎯 Business Value

### 💼 Operational Impact
- **Automated Screening**: Reduces manual review time by 80%
- **Risk Reduction**: Minimizes default probability through data-driven decisions
- **Scalability**: Processes thousands of applications efficiently
- **Consistency**: Eliminates human bias in loan approval process

### 📊 Financial Benefits
- **Cost Reduction**: Lower operational costs for loan processing
- **Revenue Optimization**: Better identification of profitable customers
- **Risk Management**: Quantified prediction confidence levels
- **Regulatory Compliance**: Transparent and explainable decision process

## 🔬 Methodology

### Data Processing Pipeline
1. **Data Loading & Exploration** - Initial dataset analysis
2. **Feature Engineering** - Creation of meaningful predictors
3. **Data Cleaning** - Missing values and outlier treatment
4. **Feature Selection** - Correlation analysis and redundancy removal
5. **Encoding & Scaling** - Categorical transformation and normalization
6. **Model Training** - Algorithm comparison and selection
7. **Hyperparameter Tuning** - Performance optimization
8. **Model Evaluation** - Comprehensive performance assessment

### Evaluation Metrics
- **Confusion Matrix** - Classification accuracy breakdown
- **ROC-AUC Curve** - Model discrimination ability
- **Precision-Recall** - Balance between accuracy and completeness
- **Feature Importance** - Key predictors identification
- **Cross-Validation** - Model stability assessment

## 📚 Key Learnings

### Technical Achievements
- ✅ Successfully handled large-scale financial dataset (466K+ records)
- ✅ Implemented advanced feature engineering techniques
- ✅ Achieved state-of-the-art model performance (99.91% accuracy)
- ✅ Developed production-ready ML pipeline
- ✅ Created comprehensive model evaluation framework

### Business Insights
- 📈 **Interest rates** are the strongest predictor of loan defaults
- 💰 **Debt consolidation** represents the safest loan category
- 🏠 **Home ownership status** significantly impacts approval probability
- 📊 **Engineered features** often outperform raw data in predictive power
- ⚖️ **Class balancing** is crucial for minority class detection

## 🚀 Future Enhancements

### Model Improvements
- [ ] **Deep Learning**: Implement neural networks for complex patterns
- [ ] **Ensemble Methods**: Combine multiple models for better accuracy
- [ ] **Real-time Scoring**: Deploy model for instant loan decisions
- [ ] **Explainable AI**: Add LIME/SHAP for detailed explanations

### Business Applications
- [ ] **Mobile Integration**: Loan approval app for instant decisions
- [ ] **Risk Monitoring**: Continuous model performance tracking
- [ ] **A/B Testing**: Compare model vs traditional approval methods
- [ ] **Regulatory Reporting**: Automated compliance documentation

## 🙏 Acknowledgments

- **Rakamin Academy** - For providing the learning platform and guidance
- **IDX Partners** - For the internship opportunity and industry insights
- **Open Source Community** - For the amazing ML libraries and tools

## 📞 Contact

**Author**: Bagus  
**Email**: bagusrajin465@gmail.com  
**LinkedIn**: https://www.linkedin.com/in/bagusrahmadani/  
**GitHub**: https://github.com/Bagus2510  

---

⭐ **If you found this project helpful, please consider giving it a star!** ⭐
