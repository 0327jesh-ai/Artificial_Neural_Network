#-------------------------------------------------------------------------------#
# ARTIFICIAL NEURAL NETWORK (ANN)
# BREAST CANCER CLASSIFICATION
#-------------------------------------------------------------------------------#

#-------------------#
# Business Problem
#-------------------#
# Breast cancer is one of the leading causes of death among women worldwide.
# Early and accurate diagnosis is critical for effective treatment and improved
# survival rates. Manual diagnosis based on clinical measurements can be time-
# consuming and prone to human error.
#
# The objective is to build an ANN-based classification model that predicts
# whether a breast tumor is malignant or benign based on diagnostic features.

#--------------------#
# High Level Solution
#--------------------#
# 1. Load the Breast Cancer dataset
# 2. Perform data preprocessing and cleaning
# 3. Remove irrelevant features (ID, unnamed columns)
# 4. Encode the target variable (Malignant / Benign)
# 5. Scale numerical features
# 6. Train an Artificial Neural Network (ANN)
# 7. Predict tumor diagnosis (Malignant or Benign)

#---------------------#
# Business Objective
#---------------------#
# Binary Classification:
# Malignant = 1 -> Cancerous tumor
# Benign    = 0 -> Non-cancerous tumor

#---------------------#
# Business Constraints
#---------------------#
# - Class imbalance between malignant and benign cases
# - Presence of irrelevant or redundant features (id, unnamed column)
# - Risk of overfitting due to high-dimensional feature space
# - Need for high recall to minimize false negatives (missing cancer cases)

#---------------------#
# Success Criteria
#---------------------#
# Business Success:
# - Early detection of malignant tumors
# - Support clinical decision-making
#
# ML Success:
# - Accuracy >= 95%
# - High recall for malignant class
# - Low false negative rate
#
# Economic Success:
# - Reduced diagnostic costs
# - Improved treatment outcomes
# - Optimized healthcare resource utilization

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
# - Breast Cancer Wisconsin Diagnostic Dataset

# Data Format:
# - CSV file

#-------------------#
# Dataset Features
#-------------------#
# id                        : Unique patient identifier (irrelevant)
# diagnosis                 : Target variable (M = Malignant, B = Benign)
#
# radius_mean               : Mean radius of tumor
# texture_mean              : Mean texture
# perimeter_mean            : Mean perimeter
# area_mean                 : Mean area
# smoothness_mean           : Mean smoothness
# compactness_mean          : Mean compactness
# concavity_mean            : Mean concavity
# concave points_mean       : Mean concave points
# symmetry_mean             : Mean symmetry
# fractal_dimension_mean    : Mean fractal dimension
#
# radius_se                 : Radius standard error
# texture_se                : Texture standard error
# perimeter_se              : Perimeter standard error
# area_se                   : Area standard error
# smoothness_se             : Smoothness standard error
# compactness_se            : Compactness standard error
# concavity_se              : Concavity standard error
# concave points_se         : Concave points standard error
# symmetry_se               : Symmetry standard error
# fractal_dimension_se      : Fractal dimension standard error
#
# radius_worst              : Worst radius
# texture_worst             : Worst texture
# perimeter_worst           : Worst perimeter
# area_worst                : Worst area
# smoothness_worst          : Worst smoothness
# compactness_worst         : Worst compactness
# concavity_worst           : Worst concavity
# concave points_worst      : Worst concave points
# symmetry_worst            : Worst symmetry
# fractal_dimension_worst   : Worst fractal dimension
#
# Unnamed: 32               : Empty column (irrelevant)

#------------------------#
# Meta Data Description
#------------------------#
# - Radius, perimeter, and area features are strong indicators of malignancy
# - Concavity and concave points play a major role in tumor severity
# - Worst-case measurements provide critical diagnostic information
# - Diagnosis is the dependent variable
#-------------------------------------------------------------------------------#

# ================================================== #
# Code Modularity
# ================================================== #

# Data manipulation & analysis
import pandas as pd                  # for DataFrames, data manipulation
import numpy as np                   # for numerical operations

# Data visualization
import matplotlib.pyplot as plt      # for basic plotting
import seaborn as sns                # for advanced and styled plots

# Machine learning preprocessing
from sklearn.model_selection import train_test_split  # for splitting data
from sklearn.preprocessing import StandardScaler      # for feature scaling

# Deep learning / ANN
import tensorflow as tf
from tensorflow.keras.models import Sequential       # for ANN model
from tensorflow.keras.layers import Dense, Dropout  # layers for ANN
from tensorflow.keras.optimizers import Adam        # optimizer for ANN

# Metrics
from sklearn.metrics import confusion_matrix, classification_report

# Visualization settings
plt.rcParams['figure.figsize'] = (10, 6)
sns.set_style("whitegrid")


# ================================================== #
# LOAD DATASET
# ================================================== #
df = pd.read_csv("data/breast_cancer.csv")   # Update path if needed

print("Dataset Loaded Successfully")
print("-" * 50)


# ================================================== #
# DATASET OVERVIEW
# ================================================== #
print("Shape of Dataset:", df.shape)       # rows x columns
print("\nData Types & Non-Null Values:")
print(df.info())                           # datatype & missing info


# ================================================== #
# MISSING VALUE ANALYSIS
# ================================================== #
print("\nMissing Values:")
print(df.isnull().sum())                   # check for nulls


# ================================================== #
# TARGET VARIABLE ANALYSIS
# ================================================== #
print("\nTarget Variable Distribution:")
print(df['diagnosis'].value_counts())      # B = benign, M = malignant

# Visualize class distribution
sns.countplot(x='diagnosis', data=df)
plt.title("Diagnosis Class Distribution")
plt.show()


# ================================================== #
# ENCODE TARGET VARIABLE
# ================================================== #
# Convert target to numeric: B -> 0, M -> 1
df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})
print("\nTarget Encoding Applied: B -> 0, M -> 1")


# ================================================== #
# STATISTICAL SUMMARY
# ================================================== #
print("\nStatistical Summary:")
print(df.describe().T)                     # summary stats for all features


# ================================================== #
# FEATURE DISTRIBUTION ANALYSIS
# ================================================== #
mean_features = ['radius_mean', 'texture_mean', 'perimeter_mean',
                 'area_mean', 'smoothness_mean']

# Histograms for selected features
df[mean_features].hist(bins=20)
plt.suptitle("Distribution of Mean Features")
plt.show()


# ================================================== #
# OUTLIER DETECTION
# ================================================== #
plt.figure(figsize=(14, 6))
sns.boxplot(data=df[mean_features])
plt.xticks(rotation=45)
plt.title("Outlier Detection in Mean Features")
plt.show()


# ================================================== #
# CORRELATION ANALYSIS
# ================================================== #
plt.figure(figsize=(14, 10))
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


# ================================================== #
# DIAGNOSIS VS KEY FEATURES
# ================================================== #
sns.boxplot(x='diagnosis', y='radius_mean', data=df)
plt.title("Radius Mean vs Diagnosis")
plt.show()

sns.boxplot(x='diagnosis', y='concave points_mean', data=df)
plt.title("Concave Points Mean vs Diagnosis")
plt.show()


# ================================================== #
# DATA PREPROCESSING
# ================================================== #

# 12.1 Drop irrelevant columns
df.drop(columns=['id', 'Unnamed: 32'], inplace=True, errors='ignore')
print("Dropped columns: id, Unnamed: 32")
print("Updated Shape:", df.shape)

# Safety check for target encoding
print("Unique values in diagnosis:", df['diagnosis'].unique())
assert df.shape[0] > 0, "DataFrame is empty! Check your source data."


# 12.2 Separate features and target
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

print("Feature Matrix Shape:", X.shape)
print("Target Vector Shape:", y.shape)


# 12.3 Train-test split (stratified to maintain class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain-Test Split Completed")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# 12.4 Feature scaling (Standardization)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature Scaling Applied (StandardScaler)")
print("Mean (Train):", np.round(X_train_scaled.mean(), 4))
print("Std  (Train):", np.round(X_train_scaled.std(), 4))


# ================================================== #
# BUILD & TRAIN ANN MODEL
# ================================================== #
# Define ANN architecture
model = Sequential([
    Dense(32, input_dim=X_train_scaled.shape[1], activation='relu'),  # input + 1st hidden
    Dense(16, activation='relu'),                                      # 2nd hidden
    Dense(8, activation='relu'),                                       # optional 3rd hidden
    Dense(1, activation='sigmoid')                                     # output layer (binary)
])

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
)

# Model summary
model.summary()

# Train model
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,  # 20% of training data for validation
    epochs=50,
    batch_size=16,
    verbose=1
)


# ================================================== #
# EVALUATION
# ================================================== #
loss, accuracy, recall, auc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test AUC: {auc:.4f}")

# Predict on test set
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)  # convert probabilities to 0/1

# Confusion matrix & classification report
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


#===============================================================================#

