#-------------------------------------------------------------------------------#
# ARTIFICIAL NEURAL NETWORK (ANN)
# DIABETES DIAGNOSIS PREDICTION
#-------------------------------------------------------------------------------#

#-------------------#
# Business Problem
#-------------------#
# Diabetes is a chronic disease that can lead to severe complications
# if not detected early. Manual diagnosis is time-consuming and
# error-prone. The goal is to build an ANN-based system that can
# predict whether a patient has diabetes using medical diagnostic data.

#--------------------#
# High Level Solution
#--------------------#
# 1. Collect patient diagnostic data from NIDDK dataset
# 2. Perform data preprocessing and cleaning
# 3. Normalize numerical medical features
# 4. Train an Artificial Neural Network (ANN)
# 5. Predict diabetes outcome (Diabetic / Non-Diabetic)

#---------------------#
# Business Objective
#---------------------#
# Binary Classification:
# Outcome = 1 -> Diabetic
# Outcome = 0 -> Non-Diabetic

#---------------------#
# Business Constraints
#---------------------#
# - Limited dataset size
# - Presence of zero / missing medical values
# - Risk of overfitting
# - High accuracy required for healthcare applications

#---------------------#
# Success Criteria
#---------------------#
# Business Success:
# - Early diagnosis of diabetes
# - Support clinical decision-making
#
# ML Success:
# - Accuracy >= 85%
# - Low false-negative rate
#
# Economic Success:
# - Reduced healthcare costs
# - Efficient screening process

#-------------------------------------------------------------------------------#
# HIGH LEVEL DESIGN
#        |
# DATA ARCHITECTURE
#        |
# MODEL ARCHITECTURE
#-------------------------------------------------------------------------------#

#-------------------------------#
# DATA UNDERSTANDING
#-------------------------------#

# Data Source:
# - National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)

# Data Format:
# - CSV file

#-------------------#
# Dataset Features
#-------------------#
# Pregnancies               : Number of pregnancies
# Glucose                   : Plasma glucose concentration
# BloodPressure             : Diastolic blood pressure (mm Hg)
# SkinThickness             : Triceps skin fold thickness (mm)
# Insulin                   : Serum insulin (mu U/ml)
# BMI                       : Body Mass Index
# DiabetesPedigreeFunction  : Genetic influence factor
# Age                       : Patient age (years)
# Outcome                   : Target variable (0 = No Diabetes, 1 = Diabetes)

#------------------------#
# Meta Data Description
#------------------------#
# - Glucose and BMI are strong predictors
# - Zero values indicate missing medical measurements
# - All features are numerical
# - Outcome is the dependent variable

#------------------------#
# CODE MODULARITY
#------------------------#


import pandas as pd                             # Data loading and manipulation
import numpy as np                              # Numerical operations
import matplotlib.pyplot as plt                 # Basic visualization
import seaborn as sns                           # Statistical plots

# Scikit-Learn: Preprocessing & Evaluation
from sklearn.model_selection import train_test_split  # Split data
from sklearn.preprocessing import StandardScaler, OneHotEncoder # Scaling/Encoding
from sklearn.compose import ColumnTransformer          # Column-wise transforms
from sklearn.pipeline import Pipeline                 # Workflow management
from sklearn.metrics import confusion_matrix, classification_report # Metrics
from sklearn.utils.class_weight import compute_class_weight        # Imbalance handling

# TensorFlow/Keras: Deep Learning
from tensorflow.keras.models import Sequential        # Model architecture
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization # Layers
from tensorflow.keras.optimizers import Adam          # Training optimizer
from tensorflow.keras.callbacks import EarlyStopping  # Overfitting prevention
from tensorflow.keras.metrics import Recall, AUC      # Performance metrics

#-------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------#
# STEP 1: EXPLORATORY DATA ANALYSIS (EDA)
#-------------------------------------------------------------------------------#

# Load Dataset
df = pd.read_csv("data/diabetes.csv")

# Dataset Inspection
print("Shape of dataset:", df.shape)
df.info()
print(df.head())
print(df.describe())

# Target Variable Analysis
print("\nOutcome Distribution:")
print(df["Outcome"].value_counts())
sns.countplot(x="Outcome", data=df)
plt.title("Diabetes Outcome Distribution")
plt.show()

# Invalid Zero Value Check
invalid_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
print("\nCount of zero values in each feature:")
for col in invalid_columns:
    print(f"{col}: {(df[col] == 0).sum()}")

# Visualizing Distributions and Correlations
df.hist(figsize=(14, 10))
plt.suptitle("Feature Distributions")
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.show()

#===============================================================================#
# EDA INSIGHTS & OBSERVATIONS
#===============================================================================#

#-------------------------------------------------------------------------------#
# 1. Dataset Overview
#-------------------------------------------------------------------------------#
# - Total records: 768 patients
# - Features: 9 columns, all numerical (int64 / float64)
# - No explicit NULL values
# - Memory footprint is low → suitable for ANN processing

#-------------------------------------------------------------------------------#
# 2. Target Variable Analysis
#-------------------------------------------------------------------------------#
# Outcome Distribution:
# - Non-Diabetic (0): 500 samples (~65%)
# - Diabetic (1): 268 samples (~35%)
# - Dataset is moderately imbalanced
# - Class imbalance handling may improve ANN performance

#-------------------------------------------------------------------------------#
# 3. Zero / Invalid Value Analysis
#-------------------------------------------------------------------------------#
# Physiologically invalid zero values present in several medical attributes:
# - Glucose: 5 zeros
# - BloodPressure: 35 zeros
# - SkinThickness: 227 zeros
# - Insulin: 374 zeros
# - BMI: 11 zeros
# Interpretation:
# - Zero values indicate missing medical measurements
# - SkinThickness and Insulin have very high missing proportions
# - These features require careful imputation during preprocessing

#-------------------------------------------------------------------------------#
# 4. Feature Distribution Insights
#-------------------------------------------------------------------------------#
# - Pregnancies:
#   - Right-skewed
#   - Most patients have fewer pregnancies
# - Glucose:
#   - Approximately normal distribution
#   - Clear separation between diabetic and non-diabetic cases
#   - Strong predictive feature
# - BloodPressure:
#   - Slight right skew
#   - Some extreme values present
# - SkinThickness & Insulin:
#   - Highly skewed distributions
#   - Large number of zeros
#   - High variance, sensitive to scaling
# - BMI:
#   - Near-normal distribution
#   - Higher BMI values more common in diabetic patients
# - DiabetesPedigreeFunction:
#   - Strong right skew
#   - Captures genetic risk rather than absolute diagnosis
# - Age:
#   - Right-skewed
#   - Diabetes likelihood increases with age

#-------------------------------------------------------------------------------#
# 5. Correlation Matrix Insights
#-------------------------------------------------------------------------------#
# Strongest correlations with Outcome:
# - Glucose (~0.47) → strongest predictor
# - BMI (~0.29) → moderate positive relationship
# - Age (~0.24) → diabetes risk increases with age
# - Pregnancies (~0.22) → indirect demographic influence
# Weak correlations:
# - BloodPressure (~0.06)
# - SkinThickness (~0.07)
# - Insulin (~0.13)
# Interpretation:
# - No multicollinearity issues (no correlation > 0.8)
# - ANN can safely learn nonlinear interactions among features

#-------------------------------------------------------------------------------#
# 6. ANN Readiness Conclusion
#-------------------------------------------------------------------------------#
# - Dataset is fully numerical and compatible with ANN
# - Feature scaling is mandatory due to varying ranges
# - Zero-value imputation is critical before training
# - Moderate class imbalance should be addressed
# - ANN is suitable due to nonlinear feature interactions

#-------------------------------------------------------------------------------#
# STEP 2: DATA PREPROCESSING
#-------------------------------------------------------------------------------#

# Define Features and Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ============================================================
# 1. CREATE FEATURE ENGINEERING COLUMNS
# ============================================================

# Example: Create Age_Group
def create_age_group(df):
    df['Age_Group'] = pd.cut(
        df['Age'],
        bins=[0, 18, 35, 50, 100],
        labels=['Child', 'Youth', 'Adult', 'Senior']
    )
    return df

# Example: Create BMI_Category
def create_bmi_category(df):
    df['BMI_Category'] = pd.cut(
        df['BMI'],
        bins=[0, 18.5, 25, 30, 100],
        labels=['Underweight', 'Normal', 'Overweight', 'Obese']
    )
    return df

# Apply feature engineering
df = create_age_group(df)
df = create_bmi_category(df)

# ============================================================
# 2. DEFINE FEATURES AND TARGET
# ============================================================

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

num_features = [
    'Pregnancies', 'Glucose', 'BloodPressure',
    'SkinThickness', 'Insulin', 'BMI',
    'DiabetesPedigreeFunction', 'Age'
]

cat_features = ['Age_Group', 'BMI_Category']

# ============================================================
# 3. PREPROCESSING PIPELINE
# ============================================================

numeric_transformer = Pipeline([
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('onehot', OneHotEncoder(
        drop='first',
        sparse_output=False,
        handle_unknown='ignore'
    ))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_features),
    ('cat', categorical_transformer, cat_features)
])

#-------------------------------------------------------------------------------#
# STEP 3: MODEL TRAINING & EVALUATION
#-------------------------------------------------------------------------------#

# Stratified Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Apply Transformations
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Handle Class Imbalance
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# ANN Architecture
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

# Compile
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy", Recall(name="recall"), AUC(name="auc")]
)

# Early Stopping
early_stop = EarlyStopping(monitor="val_auc", patience=15, mode="max", restore_best_weights=True)

# Train
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=150,
    batch_size=16,
    callbacks=[early_stop],
    class_weight=class_weight_dict,
    verbose=1
)

# Predict and Evaluate
y_prob = model.predict(X_test)
y_pred = (y_prob >= 0.45).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#-------------------------------------------------------------------------------#

