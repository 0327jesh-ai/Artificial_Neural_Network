#-------------------------------------------------------------------------------#
# ARTIFICIAL NEURAL NETWORK (ANN)
# CUSTOMER CHURN PREDICTION
#-------------------------------------------------------------------------------#

#-------------------#
# Business Problem
#-------------------#
# Customer churn is a major problem for banks and subscription-based businesses.
# Losing customers directly impacts revenue and growth. Manual churn analysis
# is inefficient and fails to capture complex customer behavior patterns.
# The objective is to build an ANN-based model that predicts whether a customer
# will leave the bank or not.

#--------------------#
# High Level Solution
#--------------------#
# 1. Load customer data from Churn_Modelling.csv
# 2. Perform data preprocessing and cleaning
# 3. Encode categorical variables (Geography, Gender)
# 4. Scale numerical features
# 5. Train an Artificial Neural Network (ANN)
# 6. Predict customer churn (Exited / Not Exited)

#---------------------#
# Business Objective
#---------------------#
# Binary Classification:
# Exited = 1 -> Customer will churn
# Exited = 0 -> Customer will not churn

#---------------------#
# Business Constraints
#---------------------#
# - Class imbalance (more retained customers)
# - Presence of irrelevant features (CustomerId, Surname)
# - Risk of overfitting
# - Need for high prediction accuracy

#---------------------#
# Success Criteria
#---------------------#
# Business Success:
# - Reduce customer churn rate
# - Improve customer retention strategies
#
# ML Success:
# - Accuracy >= 80%
# - Good recall for churned customers
#
# Economic Success:
# - Reduced customer acquisition cost
# - Improved long-term profitability

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
# - Bank Customer Churn Dataset (Churn_Modelling.csv)

# Data Format:
# - CSV file

#-------------------#
# Dataset Features
#-------------------#
# RowNumber         : Row index (irrelevant)
# CustomerId        : Unique customer ID (irrelevant)
# Surname           : Customer surname (irrelevant)
# CreditScore       : Customer credit score
# Geography         : Country (France, Germany, Spain)
# Gender            : Male / Female
# Age               : Customer age
# Tenure            : Years with bank
# Balance           : Account balance
# NumOfProducts     : Number of bank products used
# HasCrCard         : Credit card availability (0/1)
# IsActiveMember    : Active customer status (0/1)
# EstimatedSalary   : Estimated salary
# Exited            : Target variable (0 = Retained, 1 = Churned)

#------------------------#
# Meta Data Description
#------------------------#
# - Geography and Age strongly influence churn
# - High balance customers tend to churn more
# - Exited is the dependent variable
#-------------------------------------------------------------------------------#

#============================#
# EXPLORATORY DATA ANALYSIS
#============================#

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Recall, AUC
from sklearn.metrics import confusion_matrix, classification_report

#-------------------#
# Load Dataset
#-------------------#
df = pd.read_csv("data/Churn_Modelling.csv")

#-------------------#
# Basic Information
#-------------------#
print("Shape of dataset:", df.shape)
print("\nData Types:\n")
print(df.dtypes)

#-------------------#
# Check Missing Values
#-------------------#
print("\nMissing Values:\n")
print(df.isnull().sum())

#-------------------#
# Statistical Summary
#-------------------#
print("\nStatistical Summary:\n")
print(df.describe())

#-------------------#
# Target Variable Distribution
#-------------------#
print("\nChurn Distribution:\n")
print(df["Exited"].value_counts())

sns.countplot(x="Exited", data=df)
plt.title("Customer Churn Distribution")
plt.show()

#-------------------#
# Categorical Feature Analysis
#-------------------#

# Geography vs Churn
sns.countplot(x="Geography", hue="Exited", data=df)
plt.title("Geography vs Churn")
plt.show()

# Gender vs Churn
sns.countplot(x="Gender", hue="Exited", data=df)
plt.title("Gender vs Churn")
plt.show()

#-------------------#
# Numerical Feature Analysis
#-------------------#

# Age vs Churn
sns.boxplot(x="Exited", y="Age", data=df)
plt.title("Age vs Churn")
plt.show()

# Balance vs Churn
sns.boxplot(x="Exited", y="Balance", data=df)
plt.title("Balance vs Churn")
plt.show()

# Credit Score vs Churn
sns.boxplot(x="Exited", y="CreditScore", data=df)
plt.title("Credit Score vs Churn")
plt.show()

#-------------------#
# Correlation Matrix
#-------------------#
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

#-------------------#
# Irrelevant Columns Check
#-------------------#
df[["RowNumber", "CustomerId"]].head()

# ============================================================
# BANK CHURN FEATURE ENGINEERING + PREPROCESSING PIPELINE
# ============================================================

# ============================================================
# DROP IRRELEVANT COLUMNS
# ============================================================

df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# ============================================================
# FEATURE ENGINEERING
# ============================================================

# Example: Age group
df['Age_Group'] = pd.cut(
    df['Age'],
    bins=[18, 30, 40, 50, 60, 100],
    labels=['18-30','31-40','41-50','51-60','60+']
)

# Example: Balance category
df['Balance_Category'] = pd.cut(
    df['Balance'],
    bins=[-1, 0, 50000, 150000, np.inf],
    labels=['Zero','Low','Medium','High']
)

# ============================================================
# DEFINE FEATURES AND TARGET
# ============================================================

X = df.drop('Exited', axis=1)
y = df['Exited']

# Identify numeric and categorical features
num_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
cat_features = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember', 'Age_Group', 'Balance_Category']

# ============================================================
# PREPROCESSING PIPELINE
# ============================================================

numeric_transformer = Pipeline([
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_features),
    ('cat', categorical_transformer, cat_features)
])

# ============================================================
# TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ============================================================
# APPLY PREPROCESSING
# ============================================================

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

# ============================================================
# HANDLE CLASS IMBALANCE
# ============================================================

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print("Class weights:", class_weight_dict)

# ============================================================
# BUILD AND COMPILE ANN
# ============================================================

input_dim = X_train.shape[1]  # 19 features

model = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', Recall(name='recall'), AUC(name='auc')]
)

# ============================================================
# EARLY STOPPING CALLBACK
# ============================================================

early_stop = EarlyStopping(
    monitor='val_auc',
    patience=15,
    mode='max',
    restore_best_weights=True
)

# ============================================================
# TRAIN THE MODEL
# ============================================================

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=150,
    batch_size=16,
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    verbose=1
)

# ============================================================
# EVALUATE MODEL
# ============================================================

y_prob = model.predict(X_test)
y_pred = (y_prob >= 0.45).astype(int)  # You can tune threshold

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
#-------------------------------------------------------------------------------#
