## Common Commands

- Install dependencies: `pip install -r requirements.txt`
- This is a pure Python library, no build step required. Import modules directly for use in modeling workflows.

## High-Level Architecture

This repository is a modular machine learning toolset designed for credit risk modeling and tabular data analysis, following scikit-learn compatible interface design (all components implement `fit()` / `transform()` methods for pipeline integration):

### Core Modules

1. **Base Modules** (root directory):

   - `data.py`: Data loading and management (DataHelper class handles train/test split, automatic feature type detection)
   - `utils.py`: General utilities (memory optimization, timing functions)
   - `pipeline.py`: Custom sklearn pipeline components (feature selection, constant removal)
   - `metrics.py`: Model evaluation metrics (KS, Gini, RMSE, etc.)
   - `estimators.py`: Custom statistical estimators (target encoding implementations)
2. **Preprocessing** (`Preprocessing/`):

   - Feature encoding (category, count, likelihood, percentile, dummy encoding)
   - Missing value handling, scaling, feature stability validation between train/test datasets
3. **Feature Engineering** (`Feature_Engineer/`):

   - Feature combination, time-related feature generation
   - Groupby statistical feature aggregation
   - GBDT-based feature encoding
4. **Modeling** (`Model/`):

   - K-fold cross-validation wrappers for XGBoost, LightGBM, CatBoost, DNN
   - Bayesian optimization hyperparameter tuning
   - Model parsing and utility functions
5. **Feature Selection** (`FeatureSelector/`):

   - Greedy stepwise feature selection
   - Feature importance-based threshold selection
   - Forward/backward recursive selection
6. **Ensemble** (`Ensemble/`): Stacking model fusion implementation
7. **Supplementary Modules**:

   - `ScoreCard/`: Credit scorecard development tools
   - `Evalutor/`: Model performance evaluation utilities
   - `AutoModel.py`: End-to-end automated modeling workflow

## Key Usage Notes

- All components are designed to work with scikit-learn Pipeline, as demonstrated in the README example
- Primary use case: binary classification risk modeling (supports AUC, KS as main evaluation metrics)
- Library is compatible with Python 3.6+ (based on dependency versions)
