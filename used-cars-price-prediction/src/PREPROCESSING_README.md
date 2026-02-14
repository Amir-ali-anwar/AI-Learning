# Production-Ready Data Preprocessing Pipeline

This directory contains a production-ready preprocessing pipeline for the used cars price prediction project.

## 📁 File Structure

```
src/
├── config.py                      # Configuration file with all parameters
├── data_validator.py              # Data validation utilities
├── preprocessing_pipeline.py      # Main preprocessing pipeline
├── categorical_imputer.py         # Categorical imputation module
└── ...

notebooks/
├── 02_data_preprocessing_production.ipynb  # Production-ready notebook
└── 02_data_preprocessing.ipynb            # Original notebook (deprecated)
```

## 🚀 Quick Start

### Option 1: Use the Notebook

Open and run `notebooks/02_data_preprocessing_production.ipynb` for an interactive experience with visualizations and detailed logging.

### Option 2: Use the Pipeline Programmatically

```python
import sys
sys.path.append('../src')

from config import *
from preprocessing_pipeline import DataPreprocessor, split_data
import pandas as pd

# Load data
df = pd.read_csv('../data/raw/vehicles.csv')

# Create configuration
config = {
    'COLUMNS_TO_DROP': COLUMNS_TO_DROP,
    'PRICE_FILTER': PRICE_FILTER,
    'YEAR_FILTER': YEAR_FILTER,
    'ODOMETER_FILTER': ODOMETER_FILTER,
    'IMPUTATION_STRATEGY': IMPUTATION_STRATEGY,
    'CONSTANT_VALUES': CONSTANT_VALUES,
    'TARGET_COLUMN': TARGET_COLUMN,
    'OUTLIER_CONFIG': OUTLIER_CONFIG,
    'NUMERICAL_COLUMNS': NUMERICAL_COLUMNS,
    'ENCODING_CONFIG': ENCODING_CONFIG,
    'SCALING_CONFIG': SCALING_CONFIG,
    'TRAIN_TEST_SPLIT': TRAIN_TEST_SPLIT
}

# Initialize and run pipeline
preprocessor = DataPreprocessor(config)
df_processed = preprocessor.fit_transform(df)

# Split data
X_train, X_test, y_train, y_test = split_data(df_processed, config)

# Save preprocessor for later use
import pickle
with open('../models/preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)
```

### Option 3: Transform New Data

```python
import pickle

# Load saved preprocessor
with open('../models/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Transform new data
df_new_processed = preprocessor.transform(df_new)
```

## 🔧 Configuration

All preprocessing parameters are centralized in `src/config.py`:

### Key Configuration Sections

1. **Data Paths**
   - Raw and processed data locations

2. **Column Management**
   - Columns to drop
   - Numerical and categorical columns
   - Target column

3. **Domain Filters**
   - Price range: $500 - $200,000
   - Year range: 1990 - 2022
   - Odometer range: 0 - 500,000 miles

4. **Missing Value Handling**
   - Strategy per column (mode, constant, median, mean)
   - Constant values for specific columns

5. **Outlier Detection**
   - Method: IQR or Z-score
   - Handling: remove, cap, or keep
   - Configurable thresholds

6. **Feature Encoding**
   - Ordinal encoding for ordered categories
   - One-hot encoding for nominal categories
   - Label encoding for high-cardinality columns

7. **Feature Scaling**
   - Method: standard, minmax, or robust
   - Columns to scale

8. **Train-Test Split**
   - Test size: 20%
   - Random seed for reproducibility

## 📊 Pipeline Steps

The preprocessing pipeline follows these steps:

1. **Drop Unnecessary Columns**
   - Removes IDs, URLs, and non-predictive columns

2. **Filter by Domain Knowledge**
   - Applies reasonable ranges for price, year, and odometer

3. **Remove Duplicates**
   - Identifies and removes duplicate rows

4. **Handle Missing Values**
   - Uses configured strategies per column
   - Drops rows with missing target values

5. **Detect and Handle Outliers**
   - Uses IQR or Z-score method
   - Caps, removes, or keeps outliers

6. **Encode Categorical Variables**
   - Ordinal encoding for ordered categories
   - One-hot encoding for nominal categories
   - Label encoding for remaining object columns

7. **Scale Numerical Features**
   - Standardizes or normalizes numerical columns

8. **Split Data**
   - Creates train and test sets
   - Validates split for data leakage

## ✅ Data Validation

The pipeline includes comprehensive validation:

### Pre-Processing Validation
- Minimum row count check
- Duplicate detection
- Missing value percentage check
- Target column validation
- Data type validation

### Post-Processing Validation
- No object columns remaining
- No missing values
- Train-test split validation
- Data leakage detection

### Quality Reports
- Generates detailed quality reports before and after processing
- Includes statistics for all columns
- Saves reports as CSV files

## 📈 Outputs

The pipeline generates:

### Data Files
- `cleaned_data.csv` - Full processed dataset
- `X_train.csv` - Training features
- `X_test.csv` - Test features
- `y_train.csv` - Training target
- `y_test.csv` - Test target

### Model Files
- `preprocessor.pkl` - Fitted preprocessor for reuse

### Reports
- `data_quality_report_raw.csv` - Quality report for raw data
- `data_quality_report_processed.csv` - Quality report for processed data

### Visualizations
- `missing_values_raw.png` - Missing value visualization
- `target_distribution.png` - Target distribution comparison

### Logs
- `preprocessing.log` - Detailed execution log

## 🎯 Key Features

### ✅ Production-Ready
- Modular, reusable code
- Comprehensive error handling
- Extensive logging
- Configuration-driven approach

### ✅ Data Quality
- Multi-stage validation
- Quality reporting
- Outlier detection and handling
- Missing value strategies

### ✅ Reproducibility
- Fixed random seeds
- Saved preprocessor
- Detailed logging
- Version control friendly

### ✅ Flexibility
- Easy to modify configuration
- Pluggable components
- Extensible architecture
- Supports new data transformation

## 🔍 Validation Checks

The pipeline performs these validation checks:

1. **DataFrame Validation**
   - Empty dataframe check
   - Minimum rows requirement
   - Duplicate detection
   - Missing value thresholds

2. **Target Column Validation**
   - Column existence
   - Missing values in target
   - Minimum unique values
   - Negative value detection (for price)

3. **Numerical Column Validation**
   - Data type verification
   - Infinite value detection

4. **Categorical Column Validation**
   - Single value column detection

5. **Train-Test Split Validation**
   - Shape matching
   - Column consistency
   - Data leakage detection

## 📝 Logging

The pipeline uses Python's logging module:

- **Level**: INFO
- **Format**: Timestamp - Logger - Level - Message
- **Handlers**: File and console output
- **Log File**: `../logs/preprocessing.log`

Example log output:
```
2026-02-13 13:54:53 - __main__ - INFO - Starting preprocessing pipeline...
2026-02-13 13:54:53 - __main__ - INFO - Initial shape: (426880, 26)
2026-02-13 13:54:54 - __main__ - INFO - Dropped 8 columns: ['id', 'url', ...]
2026-02-13 13:54:55 - __main__ - INFO - Price filter: Removed 12345 rows
...
```

## 🛠️ Customization

### Adding New Preprocessing Steps

1. Add configuration to `config.py`
2. Implement method in `DataPreprocessor` class
3. Call method in `fit_transform()` pipeline
4. Update documentation

### Modifying Existing Steps

1. Update configuration in `config.py`
2. Modify method implementation if needed
3. Test with validation checks

### Adding New Validation

1. Add method to `DataValidator` class
2. Call in notebook or pipeline
3. Handle validation results

## 📚 Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- logging (built-in)
- pickle (built-in)

## 🤝 Best Practices

1. **Always validate** data before and after preprocessing
2. **Save the preprocessor** for consistent transformation
3. **Review logs** to understand transformations
4. **Check quality reports** for data insights
5. **Version control** configuration changes
6. **Test on sample** data before full dataset
7. **Document assumptions** in configuration

## 🐛 Troubleshooting

### Issue: Column not found
- **Solution**: Check if column exists in raw data
- **Solution**: Update `COLUMNS_TO_DROP` or column lists in config

### Issue: Too many rows removed
- **Solution**: Adjust domain filter ranges in config
- **Solution**: Review outlier handling method

### Issue: Validation fails
- **Solution**: Check validation thresholds in config
- **Solution**: Review data quality report for issues

### Issue: Memory error
- **Solution**: Process data in chunks
- **Solution**: Reduce dataset size for testing

## 📞 Support

For issues or questions:
1. Check the logs in `../logs/preprocessing.log`
2. Review quality reports in `../data/processed/`
3. Validate configuration in `src/config.py`
4. Check notebook outputs for detailed information

## 🔄 Version History

- **v1.0** (2026-02-13): Initial production-ready version
  - Modular pipeline architecture
  - Comprehensive validation
  - Configuration-driven approach
  - Extensive logging and reporting

## 📖 Next Steps

After preprocessing:
1. **Feature Engineering**: Create additional features
2. **Feature Selection**: Select most important features
3. **Model Training**: Train ML models
4. **Model Evaluation**: Evaluate performance
5. **Hyperparameter Tuning**: Optimize models

## 🎓 Learning Resources

- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Pandas Data Cleaning](https://pandas.pydata.org/docs/user_guide/missing_data.html)
- [Feature Engineering Guide](https://www.kaggle.com/learn/feature-engineering)
