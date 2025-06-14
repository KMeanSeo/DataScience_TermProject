# Data Science Term Project (Student Data Analysis Project: Sleep Patterns and Productivity)

## Team Roles
- 신채운: Preporcessing / Presentation
- 서지호: Evalutation
- 노민지: Data Exploration
- 김민서: Data Exploration / Data Preprocessing / Data Analysis and Modeling / Model Evaluation / PPT preparation / Github READ writing and organizaiton / Final report preparation

## Code
[Student Data Analysis Project](datascienc.ipynb)

## Installation
Install the required packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Comprehensive Report
### Key Findings Summary
This Project integrates and processess three independent datasets to analyze the multi-faceted correlations between student's sleep patterns, productivity, and stress levels, building a preprocessing, model train, predict and evaluate pipeline from end to end. There are 893 lines of Python code implement the pipeline lifecycle from data preprocessing to model evaluation, leveraging 15 regression models and 20 classifiers for multivariate analysis. Notably, the feature engineering stage combines over 30 scikit-learn components to imporve data quality.

Project Scale

- Computational Complexity Analysis  
    Code Metrics and Complexity:
  - Total Lines of Code: Approximately 400~500 lines including comments and markdown
  - Cyclomatic Complexity: Medium (branching in preprocessing logic and model evaluation)
  - Data processing Volume: ~6,400 total records across three datasets
  - Feature Engineering Operations: More than 15 transformation functions
  - Model Training Instances: 3 primary models + evaluation models

Advanced Analytics Implementation:

- SMOTE-based Data Augmentation: Synthetic minority oversampling for imbalanced datasets using imbalanced-learn library
    - Multi-scaling Strategy: Dataset-specific scaler optimization (StandardScaler, RobustScaler, MinMaxScaler)
    - Ensemble Architecture: Soft voting classifier combining 3 heterogeneous algorithms (LogisticRegression, KNN, RandomForest)
    - Hybrid Modeling: Simultaneous regression and classification pipeline for comprehensive analysis
    - Clustering Integration: KMeans clustering with silhouette score validation for unsupervised learning insights
- Technical Implementation Scope  
    Data Engineering Pipeline Complexity:
  - Multi-dataset Integration: Complex schema alignment and feature harmonization
  - Temporal Data Processing: Custom datetime conversion algorithms \[datetime_to_float() function)
  - Missing Value Strategies: Conditional imputation based on grouping variables with domain specific logic
  - Feature Selection: Statistical significance testing with chi-square and SelectKBest implementation

Enhanced Data Processing Pipeline:

- Imbalanced Data Handling: SMOTE (Synthetic Minority Oversampling Technique) implementation for class balance optimization
    - Advanced Feature Engineering: Custom transformation functions for temporal data conversion and categorical encoding
    - Data Quality Assurance: Multi-level validation checks for data consistency and integrity
    - Cross-Dataset Harmonization: Sophisticated schema standardization across heterogeneous data sources

Machine Learning Architecture:

- Ensemble Methods: Gradient Boosting with 100 estimators (default)
    - Cross-Dataset Prediction: Transfer learning approach between datasets
    - Multi-target Modeling: Separate model optimization for each wellbeing metric
    - Performance Evaluation: Comparative analysis framework

Machine Learning Implementation:

- Stratified Cross-Validation: StratifiedKFold 5-fold implementation ensuring population representative sampling
    - Voting Ensemble System: VotingClassifier with soft voting mechanism combining multiple base learners
    - Multi-metric Evaluation Framework: Comprehensive assessment using R2, MAE, MSE, RMSE, Accuracy and F1-score
    - Clustering Analysis: KMeans clustering with silhouette coefficient optimization for pattern discovery
    - Data Augmentation Validation: Comparative performance analysis between original and SMOTE-enhanced datasets

Statistical Strict Enhancement:

- Hypothesis Testing: Chi-square statistical significance testing for feature relevance
    - Cluster Validation: Silhouette analysis for optimal cluster number determination
    - Cross-Validation Robustness: Stratified sampling ensure balanced representation across folds
    - Performance Significance: Statistical validation of model performance differences across datasets

Technical Skill Level

- Programming Language:  
    Python 3.x  

Core Libraries and Packages:

- pandas: Data manipulation and analysis
    - numpy: Numerical computing
    - scikit-learn: Machine learning algorithms and utilities
    - matplotlib, seaborn: Data visualization
    - datetime: Date and time handling
    - scipy: Statistical functions and hypothesis testing
    - imblearn: Handling imbalanced datasets via SMOTE and other resampling techniques
    - sklearn.model_selection.StratifiedKFold: Class-aware cross-validation for robust model evaluation  

Implemented Algorithms:

- Feature Selection: Chi-square test, SelectKBest
    - Preprocessing: StandardScaler, RobustScaler, MinMaxSclaer
    - Machine Learning:
        1. GradientBoosting Regressor: Gradient boosting for regression tasks
        2. LinearRegression: Linear modeling for regression
        3. RandomForestRegressor: Soft voting ensemble combining multiple base learners (LogisticRegression, KNN, RandomForest)
    - Clustering: KMeans clustering with silhouette score validation for unsupervised pattern discovery
    - Cross-Validation:
        1. train_test_split: Simple hold-out validation
        2. StratifiedKFold: 5-fold stratified cross-validation for robust evaluation

Evaluation Metrics:

- Regression Metrics: R2, MAE, MSE, RMSE
    - Classification Metrics: Accuracy, F1-score, Confusion Matrix
    - Clustering Metrics: Silhouette Score for cluster quality assessment

Object-Oriented Programming Application:

- Class Usage: Proper instantiation of scikit-learn estimators and custom transformation
    - Method Chaining: Pandas method changing for efficient data transformation
    - Error Handling: errors=’ignore’ parameter usage for robust column operations

Advanced Implementation Techniques:

- SMOTE-based Data Augmentation: Synthetic minority oversampling for handling class imbalance
    - Multi-target Modeling: Separate model optimization for each wellbeing metric
    - Cross-Dataset Learning: Transfer learning approach between datasets
    - Performance Evaluation Framework: Comparative analysis between original and SMOTE-augmented datasets, multi-metric evaluation

Libraries and Packages Version:

- pandas: 2.2.3
    - numpy: 2.2.6
    - scikit-learn: 1.6.1
    - matplotlib: 3.10.3
    - seaborn: 0.13.2
    - scipy: 1.15.3
    - imbalanced-learn: 0.13.0
- Programming Proficiency Demonstration  
    Advanced Python Techniques:

Object-Oriented Programming Application:

- Class Usage: Proper instantiation of scikit-learn estimators
    - Method Chaining: Pandas method changing for efficient data transformation
    - Error Handling: errors=’ignore’ parameter usage for robust column operations
- Statistical and Mathematical Competency  
    Statistical Methods Implementation:
  - Chi-square Testing: Proper application for categorical feature selection
  - Scaling Algorithms: Understanding of when to use StandardScaler and RobustSclaer
  - Cross-Validation Ready: Code structure supports k-fold validation implementation
  - Missing Data Theory: Group-based imputation using statistical mode

Mathematical Understanding:

- Gradient Boosting Algorithm: Appropriate algorithm selection for regression tasks
    - Feature Space Transformation: Dimensional alignment across datasets
    - Outlier Detection: Clipping operations for chi-square compatibility
- Data Science Methodology  
    Practices Implementation:
  - Reproducible Research: Random state setting (random_state = 42)
  - Data Leakage Prevention: Proper train/test separation principles
  - Feature Engineering: Domain knowledge application in transformations
  - Model Validation: Comparative evaluation framework

Completeness

- Functional Requirements Fulfillment  
    Implemented Components:  
    <br/>Data Integration:
  - Three dataset loading and initial exploration
  - Schema standardization across heterogeneous data sources
  - Feature alignment for cross-dataset compatibility
  - Data type consistency enforcement
  - Advanced Data Harmonization
  - Multi-source Validation

Preprocessing Pipeline:

- Missing value imputation with domain logic
    - Categorical variable encoding (one-hot, ordinal)
    - Temporal data conversion to numerical format
    - Outlier handling through robust scaling
    - Comprehensive Data Quality Framework
    - SMOTE-based Data Augmentation
    - Multi-scaling Strategy

Machine Learning Implementation:

- Multi-target regression modeling
    - Feature selection using statistical methods
    - Cross-dataset prediction capability
    - Data augmentation through prediction
    - Default Parameter Strategy
    - Complete Cross-Validation Framework
    - Ensemble Architecture
    - Clustering Analysis

Evaluation Framework:

- Comparative analysis setup (original vs augmented data)
    - Multiple model evaluation preparation
    - Performance metrics calculation
    - Statistical Validation Framework
    - Comprehensive Feature Analysis
- Advanced Implementation Components  
    <br/>Model validation and Testing:
  - Stratified Cross-Validation: 5-fold StratifiedKFold implementation ensuring population representative sampling
  - Performance Robustness Testing: Cross-validation with SMOTE integration for balanced evaluation
  - Multi-metric Assessment: Comprehensive evaluation using regression (R2, MAE, MSE, RMSE) and classification (Accuracy, F1-Score) metrics
  - Clustering validation: Silhouette analysis for optimal cluster determination and quality assessment

Data Augmentation and Enhancement:

- SMOTE Implementation: Synthetic Minority Oversampling Technique for addressing class imbalance
    - Cross-Dataset Learning: Transfer learning approach between heterogeneous sleep pattern datasets
    - Predictive Data Augmentation: Model-based student data enhancement with performance comparison
    - Quality Assurance: Original vs Augmented data performance validation

Statistical Strict and Validation:

- Hypothesis Testing: Chi-square statistical significance attesting for feature relevance determination
    - Cluster Quality Assessment: Silhouette coefficient calculation for clustering validation
    - Cross-Validation Robustness: Stratified sampling ensuring balanced representation across validation folds
    - Performance Significance Analysis: Statistical comparison between original and SMOTE-enhanced datasets

Comments

- Documentation Quality Assessment  
    Comment Density and Distribution:
  - Inline Comments: Code lines contain explanatory comments
  - Docstring Usage: Function-level documentation for complex operations
  - Section Headers: Markdown cells providing structural organization
  - Parameter Explanations: Detailed rationale for transformation choices

Technical Documentation Standards:

Algorithmic Explanation Comments:

Process Flow Documentation:

- Project Overview and Results  
    Project purpose and Scope:
  - Developed a comprehensive data science system that analyzes student sleep patterns to predict productivity, mood and stress levels. Integrating three independent sleep-related datasets with total of 64,000 records, project utilized various machine learning techniques to derive insights for improving student welfare

Experiment:

- Data Integration Success
        1.  Successfully integrated three datasets with different structures and completed schema standardization
        2.  Built high-quality dataset through missing value handling, outlier removal, and data type consistency assurance
        3.  Preprocessing by converting temporal data into numerical format for analysis
    - Modeling Performance
        1. Implemented regression models (productivity and mood prediction) and classification models (stress level prediction)
        2. Solve imbalance data problems and improved model performance through SMOTE-based data augmentation
        3. Enhanced prediction accuracy compared to single models by utilizing ensemble methodology
    - Validation and Evaluation
        1. Ensure model reliability through StratifiedKFold 5-fold cross validation
        2. Conducted comprehensive performance evaluation using multiple metrics (R2, MAE, MSE, RMSE, Accuracy, F1-Score)
        3. Validated data augmentation effectiveness through performance comparison between original and augmented data
- Major Technical Implementation  
    Data Processing Techniques:
  - SMOTE-based Data Augmentation
        1. Implemented Synthetic Minority Oversampling Technique using the imbalanced-learn library
        2. Solve class imbalance problems and improved model generalization performance
        3. Quantitiatively validated augmentation effects through comparative analysis of original and augmented data performance

Model Architecture:

- Ensemble methodology
        1.  Implement soft voting classifier combining LogisticRegression, KNN and RandomForest using VotingClassifier
        2.  Use multiple regression model utilizing GradientBoostingRegressor, LinearRegression and RandomForestRegressor
        3.  Achieved improved performance compared to single models by combining the strengths of each model
    - Cross-Validation Framework
        1. Implemented 5-fold cross-validation using StratifiedKFold
        2. Maintained consistent class ratios in each fold through stratified sampling
        3. Evaluate model performance while preventing data leakage by integrate SMOTE with cross-validation
    - Feature Selection and Dimensionality Reduction
        1. Implemented statistical feature selection using chi-square testing
        2. Automatically selected most important feature through SelectKBest
        3. Improve model interpretability through statistically significant feature selection
    - Clustering Analysis
        1. Discover hidden structure in sleep patterns through KMeans clustering
        2. Determine optimal cluster numbers through Silhouette analysis
        3. Built hybrid analysis framework combining supervised and unsupervised learning
- Future Improvements  
    Model Performance Enhancement:
  - Hyperparameter Optimization: Maximize model performance through GridSearchCV, RandomizedSearchCV, and Bayesian Optimization
  - Advanced Ensemble: Implement XGBoost and Stacking/Blending techniques
  - Model diversity expansion: Improve prediction accuracy by integrating various algorithms