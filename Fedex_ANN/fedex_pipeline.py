#-------------------------------------------------------------------------------#
# ARTIFICIAL NEURAL NETWORK (ANN) FEDEX SHIPMENT DELIVERY PREDICTION
#-------------------------------------------------------------------------------#

#-------------------
# Business Problem
#-------------------

# FedEx handles a massive volume of shipments daily across multiple routes,
# carriers, and time schedules. Delays in delivery lead to customer dissatisfaction,
# financial penalties, and operational inefficiencies.
#
# The challenge is to accurately predict whether a shipment will be delivered
# on time or delayed using historical shipment data.
#
# An ANN-based predictive system can help proactively identify high-risk shipments
# and enable better logistical decision-making.

#--------------------
# High Level Solution
#--------------------

# 1. Collect historical FedEx shipment data containing time, distance,
#    carrier, route, and delay-related features.
# 2. Perform data preprocessing and feature engineering to prepare
#    data suitable for ANN training.
# 3. Build an Artificial Neural Network (ANN) to classify shipments as
#    "On-Time" or "Delayed".
# 4. Use the trained model to predict delivery outcomes for future shipments.
# 5. Integrate predictions into monitoring dashboards for real-time insights.

#---------------------
# Business Objective
#---------------------

# Delivery Status Prediction
# (On-Time vs Delayed)

#---------------------
# Business Constraints
#---------------------

# • Limited historical labeled data
# • High class imbalance (more on-time than delayed shipments)
# • Real-time prediction requirements
# • Memory and compute efficiency for large datasets

#---------------------
# Success Criteria
#---------------------

# Business Success Criteria:
# • Reduction in delayed shipments
# • Improved customer satisfaction
# • Better route and carrier optimization

# Machine Learning Success Criteria:
# • ANN classification accuracy ≥ 90%
# • Stable generalization on unseen data
# • Low false-negative rate for delayed shipments

# Economic Success Criteria:
# • Lower operational costs
# • Efficient resource allocation
# • Improved ROI through predictive logistics

#-------------------------------------------------------------------------------#
# HIGH LEVEL DESIGN
#        |
# DATA ARCHITECTURE
#        |
# MODEL ARCHITECTURE
#-------------------------------------------------------------------------------#

#-------------------------------
# DATA UNDERSTANDING
#-------------------------------#

# DATA SOURCES
# • FedEx internal shipment logs
#
# DATA COLLECTION
# • CSV-based historical shipment records
#
# DATA STORAGE
# • Local filesystem / Cloud storage
#
# DATA ANALYSIS
# • Statistical and visual inspection
# • Feature relationships and trends

#-------------------
# Data Description
#-------------------

# The dataset contains historical shipment records with the following attributes:
#
# • Year                    : Shipment year
# • Month                   : Shipment month
# • DayofMonth              : Day of the month
# • DayOfWeek               : Day of the week
# • Actual_Shipment_Time    : Actual shipment dispatch time
# • Planned_Shipment_Time   : Scheduled shipment time
# • Planned_Delivery_Time   : Expected delivery time
# • Planned_TimeofTravel    : Planned travel duration
# • Shipment_Delay          : Delay duration (minutes)
# • Distance                : Shipping distance
# • Carrier_Name            : Logistics carrier
# • Source                  : Shipment origin
# • Destination             : Shipment destination
# • Delivery_Status         : Target variable (0 = On-Time, 1 = Delayed)

#------------------------
# Meta Data Description
#------------------------

# • Time-based features help capture operational delays
# • Distance and travel time indicate logistical complexity
# • Carrier, source, and destination affect reliability
# • Delivery_Status is the dependent variable used for ANN classification

#-------------------------------------------------------------------------------#
# CODE MODULARITY
#-------------------------------------------------------------------------------#

# 1. Library Imports & Environment Setup
# 2. Data Acquisition & Initial Audit
# 3. Exploratory Data Analysis (EDA)
# 4. Data Preprocessing & Feature Engineering
# 5. Feature Scaling & Encoding
# 6. Clustering (Unsupervised Learning)
# 7. ANN Model Building
# 8. Model Training & Optimization
# 9. Model Evaluation
# 10. Model Persistence & Deployment Readiness

#-------------------------------------------------------------------------------#
# LIBRARY IMPORTS & SETUP
#-------------------------------------------------------------------------------#

# pandas:
# Used for data loading, manipulation, cleaning, and feature engineering.
import pandas as pd

# numpy:
# Provides numerical operations and array-based computations.
import numpy as np

# matplotlib.pyplot:
# Used for basic plotting and visualization.
import matplotlib.pyplot as plt

# seaborn:
# High-level visualization library for statistical plots.
import seaborn as sns

# scikit-learn:
# Used for preprocessing, scaling, train-test split, clustering,
# and evaluation utilities.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix

# TensorFlow / Keras:
# Used for building, training, and optimizing the ANN model.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# joblib:
# Used for saving preprocessing objects like scalers.
import joblib

# Set visualization theme
sns.set_theme(style="whitegrid")

#-------------------------------------------------------------------------------#


#--------------------------------------------------------------------------------
# Exploratory Data Analysis
#--------------------------------------------------------------------------------

# ========================================== 
# DATA ACQUISITION & INITIAL AUDIT 
# ========================================== 
# Load the FedEx dataset into a pandas DataFrame 
df = pd.read_csv("data/fedex.csv")

# Print basic structural information to understand the "scale" of the data
print(f"Dataset Dimensions: {df.shape}") 
print("\nAvailable Features (Columns):\n", df.columns)

# Display the first 5 rows to get a 'feel' for the actual data entries
print("\nSample Data:\n", df.head())

# Check data types (e.g., is 'Distance' an integer or a string?) 
# and identify missing values that could break your analysis later.
print("\nData Audit (Types & Nulls):")
print(df.info()) 

# ==========================================
# UNIVARIATE ANALYSIS (Single Variable)
# ==========================================

# Summarize the 'Delivery_Status' to see if the classes are balanced or skewed
print("\nDelivery Status Breakdown:\n", df["Delivery_Status"].value_counts())

# Bar chart for Delivery Status frequency
plt.figure(figsize=(8, 5))
sns.countplot(x="Delivery_Status", data=df, palette="Set2")
plt.title("Volume of Deliveries by Status")
plt.show()

# Histogram to see the range and frequency of shipping distances
plt.figure(figsize=(8, 5))
sns.histplot(df["Distance"], bins=30, kde=True, color="teal")
plt.title("Distribution of Shipping Distance")
plt.xlabel("Distance")
plt.show()

# ==========================================
# BIVARIATE ANALYSIS (Relationships)
# ==========================================

# Boxplot: Helps identify if "Shipment Delay" is higher for certain delivery statuses
# (e.g., checking if 'Late' deliveries have a specific delay threshold)
plt.figure(figsize=(8, 5))
sns.boxplot(x="Delivery_Status", y="Shipment_Delay", data=df)
plt.title("Shipment Delay vs. Delivery Status")
plt.show()

# Boxplot: Checking if longer distances lead to specific delivery outcomes
plt.figure(figsize=(8, 5))
sns.boxplot(x="Delivery_Status", y="Distance", data=df)
plt.title("Does Distance Influence Delivery Success?")
plt.show()

# Grouped Bar Chart: Comparing carrier reliability side-by-side
plt.figure(figsize=(12, 6))
sns.countplot(x="Carrier_Name", hue="Delivery_Status", data=df)
plt.title("Carrier Performance Comparison")
plt.xticks(rotation=45) # Rotate labels so they don't overlap
plt.show()

# ==========================================
# DATA AGGREGATION & CORRELATION
# ==========================================

# Grouping by Source and Destination to find routes with the lowest performance
# We take the mean of the status (assuming 1 is successful, lower is worse)
top_routes = (
    df.groupby(["Source", "Destination"])["Delivery_Status"]
    .mean()
    .reset_index()
    .sort_values(by="Delivery_Status")
)

print("\nCritical Routes (Lowest Performance):\n", top_routes.head())

# Correlation Heatmap: Shows which numerical features move together.
# 1.0 is a perfect positive relationship; -1.0 is a perfect negative relationship.
plt.figure(figsize=(10, 8))
# 'numeric_only=True' ensures we don't try to correlate text columns
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# =========================================================
# MEMORY-EFFICIENT FEATURE ENGINEERING & PREPROCESSING
# =========================================================

# Avoid memory-heavy drop_duplicates()
# df = df.drop_duplicates()  # Skipped to save memory

# Separate numerical and categorical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

# Fill missing values
# Numerical columns: median (robust to outliers)
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical columns: mode (most frequent)
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# ------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------

# 3.1 Time Normalization
# Convert HHMM format to total minutes from midnight
def convert_hhmm_to_minutes(x):
    try:
        x = int(x)
        hh = x // 100
        mm = x % 100
        return hh*60 + mm
    except:
        return 0

df['Actual_Shipment_Time'] = df['Actual_Shipment_Time'].apply(convert_hhmm_to_minutes)
df['Planned_Shipment_Time'] = df['Planned_Shipment_Time'].apply(convert_hhmm_to_minutes)
df['Planned_Delivery_Time'] = df['Planned_Delivery_Time'].apply(convert_hhmm_to_minutes)

# 3.2 Derived Metrics
# Shipment Duration
df['Shipment_Duration'] = df['Planned_Delivery_Time'] - df['Planned_Shipment_Time']

# Delay Flag: 1 if delayed, 0 otherwise
df['Delay_Flag'] = (df['Shipment_Delay'] > 0).astype(int)

# Speed: Distance / Planned Time of Travel
df['Speed'] = df['Distance'] / (df['Planned_TimeofTravel'] + 1)  # +1 to avoid division by zero

# 3.3 Cyclic Encoding (Month and DayOfWeek)
df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

# ------------------------------------------
# MEMORY-EFFICIENT CATEGORICAL ENCODING
# ------------------------------------------

# Use frequency encoding for high-cardinality columns to save memory
high_card_cols = ['Carrier_Name', 'Source', 'Destination']

for col in high_card_cols:
    freq = df[col].value_counts() / len(df)
    df[col + '_freq'] = df[col].map(freq)

# Drop original categorical columns to save memory
df.drop(high_card_cols, axis=1, inplace=True)

# ------------------------------------------
# FEATURE SCALING
# ------------------------------------------

# Combine numeric + derived + freq-encoded features
num_features = [
    'Year', 'Month', 'DayofMonth', 'Actual_Shipment_Time',
    'Planned_Shipment_Time', 'Planned_Delivery_Time',
    'Carrier_Num', 'Planned_TimeofTravel', 'Shipment_Delay',
    'Distance', 'Shipment_Duration', 'Speed',
    'Month_sin', 'Month_cos', 'DayOfWeek_sin', 'DayOfWeek_cos',
    'Delay_Flag',
    'Carrier_Name_freq', 'Source_freq', 'Destination_freq'
]

# Standard scaling: mean=0, std=1 (important for ANN & clustering)
scaler = StandardScaler()
df[num_features] = scaler.fit_transform(df[num_features])

# ------------------------------------------
# FINAL FEATURE MATRIX & TARGET
# ------------------------------------------

X = df[num_features]  # Features for clustering / ANN
y = df['Delivery_Status']  # Target variable for supervised tasks

print("Feature Engineering & Preprocessing Complete!")
print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")


#-------------------------------------------------------------------------------
# Model Building
#-------------------------------------------------------------------------------

# =========================================================
# MEMORY-EFFICIENT BEST K SELECTION FOR CLUSTERING
# =========================================================

# Take a random sample to save memory (e.g., 100k rows)
sample_size = 100000
X_sample = X.sample(n=sample_size, random_state=42)

# Range of k to try
k_values = range(2, 11)  # Try 2 to 10 clusters

inertia_list = []

for k in k_values:
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=10000, random_state=42)
    kmeans.fit(X_sample)
    inertia_list.append(kmeans.inertia_)
    print(f"k={k}, Inertia={kmeans.inertia_}")

# Plot Elbow curve
plt.figure(figsize=(8,5))
plt.plot(k_values, inertia_list, 'bo-', markersize=8)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Sum of Squared Distances)')
plt.title('Elbow Method for Optimal k')
plt.show()

# =========================================================
# FIT MINI BATCH KMEANS WITH BEST K
# =========================================================

# Suppose the elbow suggests k=4
best_k = 4

kmeans_final = MiniBatchKMeans(n_clusters=best_k, batch_size=10000, random_state=42)
df['Cluster_Label'] = kmeans_final.fit_predict(X)

print(" Clustering with best k complete!")
print(df['Cluster_Label'].value_counts())



# ==========================================
# TRAIN-TEST SPLIT
# ==========================================
# Assuming X and y are already preprocessed feature matrix and target vector
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training shape: {X_train.shape}, Testing shape: {X_test.shape}")

# ==========================================
# HANDLE CLASS IMBALANCE
# ==========================================
# Compute class weights to handle imbalanced dataset
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights))
print(f"Class weights: {class_weights_dict}")

# ==========================================
# BUILD ANN MODEL
# ==========================================
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.3),  # Dropout for regularization
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==========================================
# EARLY STOPPING CALLBACK
# ==========================================
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
)

# ==========================================
# TRAIN MODEL
# ==========================================
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=8192,  # Large batch to save memory
    class_weight=class_weights_dict,
    callbacks=[early_stop],
    verbose=2
)

# ==========================================
# EVALUATION
# ==========================================
# Evaluate on test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

# Predictions
y_pred = (model.predict(X_test, batch_size=8192) > 0.5).astype(int)

# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Save the ANN model to disk
model.save("shipment_delivery_model.h5")
print(" Model saved successfully!")

# Save the StandardScaler
joblib.dump(scaler, "scaler.pkl")
print("Scaler saved successfully!")

#-------------------------------------------------------------------------------#
