"""
Configuration file for data preprocessing pipeline.
"""

# Data paths
RAW_DATA_PATH = '../data/raw/vehicles.csv'
PROCESSED_DATA_PATH = '../data/processed/'

# Target column
TARGET_COLUMN = 'price'

# Columns to drop (not useful for modeling)
COLUMNS_TO_DROP = [
    'id',           # Unique identifier, not predictive
    'url',          # URL, not predictive
    'region_url',   # URL, not predictive
    'VIN',          # Too many unique values, high cardinality
    'image_url',    # URL, not predictive
    'description',  # Text data, requires separate NLP processing
    'county',       # Too many missing values and redundant with region/state
    'posting_date'  # Could be engineered but skipping for now
]

# Numerical columns for outlier detection
NUMERICAL_COLUMNS = ['price', 'year', 'odometer']

# Categorical columns for imputation
CATEGORICAL_COLUMNS = [
    'manufacturer', 'model', 'condition', 'cylinders',
    'fuel', 'title_status', 'transmission', 'drive',
    'type', 'paint_color', 'region', 'state'
]

# Imputation strategies for different columns
IMPUTATION_STRATEGY = {
    'manufacturer': 'mode',
    'model': 'constant',  # Use 'unknown' for missing models
    'condition': 'mode',
    'cylinders': 'mode',
    'fuel': 'mode',
    'title_status': 'mode',
    'transmission': 'mode',
    'drive': 'mode',
    'type': 'mode',
    'paint_color': 'constant',  # Use 'unknown' for missing colors
    'region': 'mode',
    'state': 'mode'
}

# Constant values for constant imputation
CONSTANT_VALUES = {
    'model': 'unknown',
    'paint_color': 'unknown'
}

# Outlier handling configuration
OUTLIER_CONFIG = {
    'method': 'iqr',  # 'iqr' or 'zscore'
    'iqr_multiplier': 1.5,
    'zscore_threshold': 3,
    'handle_method': 'cap'  # 'remove', 'cap', or 'keep'
}

# Price filtering (domain knowledge)
# Tightened from $500-$200K to remove extreme outliers that inflate MAPE
PRICE_FILTER = {
    'min': 2000,     # Below $2K = mostly junk/salvage cars with misleading features
    'max': 60000     # Above $60K = rare luxury cars, insufficient data to learn
}

# Year filtering (domain knowledge)
YEAR_FILTER = {
    'min': 1995,     # Minimum reasonable year (tightened from 1990)
    'max': 2022      # Maximum reasonable year (adjust based on data collection date)
}

# Odometer filtering (domain knowledge)
ODOMETER_FILTER = {
    'min': 100,      # 0-mile "used" cars are likely data entry errors
    'max': 300000    # 300K miles is a practical upper limit (tightened from 500K)
}

# High cardinality columns (need special encoding)
HIGH_CARDINALITY_COLUMNS = ['model', 'region']
HIGH_CARDINALITY_THRESHOLD = 50  # Number of unique values to consider high cardinality

# Encoding configuration
ENCODING_CONFIG = {
    'ordinal_columns': {
        'condition': ['salvage', 'fair', 'good', 'excellent', 'like new', 'new']
    },
    'onehot_columns': [
        'manufacturer', 'fuel', 'title_status', 'transmission',
        'drive', 'type', 'paint_color', 'cylinders'
    ],
    'target_encoding_columns': ['model', 'region'],  # High cardinality columns
    'drop_first': True  # Drop first category in one-hot encoding
}

# Train-test split configuration
TRAIN_TEST_SPLIT = {
    'test_size': 0.2,
    'random_state': 42,
    'stratify': None  # Can stratify by a column if needed
}

# Feature scaling configuration
SCALING_CONFIG = {
    'method': 'standard',  # 'standard', 'minmax', or 'robust'
    'columns': ['year', 'odometer', 'lat', 'long']
}

# Data validation thresholds
VALIDATION_CONFIG = {
    'max_missing_percentage': 50,  # Maximum percentage of missing values allowed per column
    'min_rows': 1000,              # Minimum number of rows required
    'min_unique_target': 10        # Minimum unique values in target column
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': '../logs/preprocessing.log'
}

# Random seed for reproducibility
RANDOM_SEED = 42
