import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# Load the dataset
df = pd.read_csv("industrial_aerospace_pdm_dataset_with_maintenance.csv")

# ===================== FEATURE ENGINEERING IMPROVEMENTS =====================

# Convert timestamp to datetime if it's not already
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Extract time-based features
df['hour_of_day'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month

# Group by machine_id and create features based on rolling windows
df = df.sort_values(by=['machine_id', 'timestamp'])

# Time since last maintenance binning (create categories)
df['maintenance_bin'] = pd.cut(df['time_since_last_maintenance'], 
                              bins=[0, 100, 200, 300, 400, 500, float('inf')],
                              labels=[0, 1, 2, 3, 4, 5])

# Group by machine ID to create machine-specific features
df_grouped = df.groupby('machine_id')

# Rolling window statistics for each machine
rolling_windows = [5, 10, 20]
for window in rolling_windows:
    # Temperature rolling stats
    df[f'temp_rolling_mean_{window}'] = df_grouped['temperature_C'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean())
    df[f'temp_rolling_std_{window}'] = df_grouped['temperature_C'].transform(
        lambda x: x.rolling(window=window, min_periods=1).std().fillna(0))
    
    # Vibration rolling stats
    df[f'vibration_rolling_mean_{window}'] = df_grouped['vibration_g'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean())
    df[f'vibration_rolling_std_{window}'] = df_grouped['vibration_g'].transform(
        lambda x: x.rolling(window=window, min_periods=1).std().fillna(0))
    
    # Feature 1 rolling stats
    df[f'feature1_rolling_mean_{window}'] = df_grouped['feature_1'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean())
    df[f'feature1_rolling_std_{window}'] = df_grouped['feature_1'].transform(
        lambda x: x.rolling(window=window, min_periods=1).std().fillna(0))

# Create interaction features
df['temp_vibration_interaction'] = df['temperature_C'] * df['vibration_g']
df['feature1_temp_interaction'] = df['feature_1'] * df['temperature_C']
df['feature1_vibration_interaction'] = df['feature_1'] * df['vibration_g']

# One-hot encode machine types
df = pd.get_dummies(df, columns=['machine_type', 'maintenance_bin'])

# ===================== DATA PREPARATION IMPROVEMENTS =====================

# Define features and target
base_features = [
    'temperature_C', 'vibration_g', 'feature_1', 'time_since_last_maintenance',
    'hour_of_day', 'day_of_week', 'month',
    'temp_vibration_interaction', 'feature1_temp_interaction', 'feature1_vibration_interaction'
]

# Add rolling window features to the feature list
for window in rolling_windows:
    base_features.extend([
        f'temp_rolling_mean_{window}', f'temp_rolling_std_{window}',
        f'vibration_rolling_mean_{window}', f'vibration_rolling_std_{window}',
        f'feature1_rolling_mean_{window}', f'feature1_rolling_std_{window}'
    ])

# Add machine type and maintenance bin columns
machine_cols = [col for col in df.columns if col.startswith('machine_type_') or col.startswith('maintenance_bin_')]
features = base_features + machine_cols
target = 'RUL'

# Handle NaN values
df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

# Better data stratification - split by machine_id to prevent data leakage
unique_machines = df['machine_id'].unique()
train_machines, test_machines = train_test_split(unique_machines, test_size=0.2, random_state=42)

train_df = df[df['machine_id'].isin(train_machines)]
test_df = df[df['machine_id'].isin(test_machines)]

# Use robust scalers for better outlier handling
feature_scaler = StandardScaler()  # Better than MinMaxScaler for many ML problems
X_train = feature_scaler.fit_transform(train_df[features])
X_test = feature_scaler.transform(test_df[features])

# Scale target separately
target_scaler = MinMaxScaler()  # Keep MinMaxScaler for target to have bounded values
y_train = target_scaler.fit_transform(train_df[[target]])
y_test = target_scaler.transform(test_df[[target]])

# Clean data
X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)
y_train = np.nan_to_num(y_train)
y_test = np.nan_to_num(y_test)

# Ensure float32 type for TensorFlow efficiency
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

# ===================== IMPROVED MODEL ARCHITECTURE =====================

def build_advanced_model(input_shape):
    # Input layer
    inputs = Input(shape=(input_shape,))
    
    # First branch - Deep network for complex patterns
    x1 = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.3)(x1)
    
    x1 = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.3)(x1)
    
    x1 = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.3)(x1)
    
    # Second branch - Shallower network for direct patterns
    x2 = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(inputs)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.2)(x2)
    
    # Combine branches
    combined = Concatenate()([x1, x2])
    
    # Final processing
    combined = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.2)(combined)
    
    combined = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.2)(combined)
    
    # Output layer - Using sigmoid for bounded prediction as we normalized the target
    output = Dense(1, activation='sigmoid')(combined)
    
    # Create and compile model
    model = Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    
    return model

# ===================== TRAINING WITH CROSS-VALIDATION =====================

# Define model saving path
model_path = 'advanced_rul_prediction_model.h5'

# Use cross-validation for more robust training
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Lists to store metrics across folds
val_rmses = []
val_maes = []
val_r2s = []
histories = []

# Check if model exists
if os.path.exists(model_path):
    # Load the saved model
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Model loaded from '{model_path}'")
else:
    # Cross-validation training
    print("Starting cross-validation training...")
    
    fold = 0
    for train_idx, val_idx in kf.split(X_train):
        fold += 1
        print(f"\n--- Training fold {fold}/{n_splits} ---")
        
        # Split data for this fold
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        
        # Build model for this fold
        model = build_advanced_model(X_train.shape[1])
        
        if fold == 1:
            model.summary()
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=0.00001,
            verbose=1
        )
        
        # Checkpoint to save best model
        checkpoint = ModelCheckpoint(
            f'temp_model_fold_{fold}.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        # Train the model
        history = model.fit(
            X_train_fold, y_train_fold,
            epochs=200,
            batch_size=32,
            validation_data=(X_val_fold, y_val_fold),
            callbacks=[early_stopping, reduce_lr, checkpoint],
            verbose=1
        )
        
        # Store history for plotting
        histories.append(history)
        
        # Evaluate on validation set
        y_val_pred = model.predict(X_val_fold)
        
        # Scale back predictions and actual values
        y_val_orig = target_scaler.inverse_transform(y_val_fold)
        y_val_pred_orig = target_scaler.inverse_transform(y_val_pred)
        
        # Calculate metrics
        val_rmse = np.sqrt(mean_squared_error(y_val_orig, y_val_pred_orig))
        val_mae = mean_absolute_error(y_val_orig, y_val_pred_orig)
        val_r2 = r2_score(y_val_orig, y_val_pred_orig)
        
        val_rmses.append(val_rmse)
        val_maes.append(val_mae)
        val_r2s.append(val_r2)
        
        print(f"Fold {fold} - RMSE: {val_rmse:.2f}, MAE: {val_mae:.2f}, R²: {val_r2:.4f}")
    
    # Print average metrics
    print(f"\nAverage Cross-validation - RMSE: {np.mean(val_rmses):.2f}, MAE: {np.mean(val_maes):.2f}, R²: {np.mean(val_r2s):.4f}")
    
    # Train final model on all training data
    print("\nTraining final model on all training data...")
    final_model = build_advanced_model(X_train.shape[1])
    
    # Callbacks for final model
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=0.00001,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Train final model
    final_history = final_model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1
    )
    
    # Load the best model (saved by checkpoint callback)
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Final model saved to '{model_path}'")
    
    # Plot training metrics from the final model
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(final_history.history['loss'], label='Train')
    plt.plot(final_history.history['val_loss'], label='Validation')
    plt.title('Final Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(final_history.history['mean_absolute_error'], label='Train')
    plt.plot(final_history.history['val_mean_absolute_error'], label='Validation')
    plt.title('Final Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(final_history.history['lr'])
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# ===================== IMPROVED EVALUATION =====================

# Evaluate on test set
y_pred = model.predict(X_test)

# Scale back predictions and actual values
y_test_orig = target_scaler.inverse_transform(y_test)
y_pred_orig = target_scaler.inverse_transform(y_pred)

# Calculate error metrics
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
mae = mean_absolute_error(y_test_orig, y_pred_orig)
r2 = r2_score(y_test_orig, y_pred_orig)

print(f"\nTest Set Evaluation:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.4f}")

# Calculate machine type specific metrics
test_df['predictions'] = y_pred_orig

# Aggregate machine types (get back from one-hot encoding)
machine_types = ['CNC', 'Machining', 'PickAndPlace']
for machine_type in machine_types:
    mask = test_df[f'machine_type_{machine_type}'] == 1
    if mask.sum() > 0:
        machine_rmse = np.sqrt(mean_squared_error(test_df.loc[mask, 'RUL'], test_df.loc[mask, 'predictions']))
        machine_mae = mean_absolute_error(test_df.loc[mask, 'RUL'], test_df.loc[mask, 'predictions'])
        machine_r2 = r2_score(test_df.loc[mask, 'RUL'], test_df.loc[mask, 'predictions'])
        print(f"\n{machine_type} Machine Type:")
        print(f"RMSE: {machine_rmse:.2f}")
        print(f"MAE: {machine_mae:.2f}")
        print(f"R²: {machine_r2:.4f}")

# Create enhanced visualization
plt.figure(figsize=(18, 6))

# True vs Predicted scatter plot with better styling
plt.subplot(1, 3, 1)
plt.scatter(y_test_orig, y_pred_orig, alpha=0.5, edgecolors='w', linewidths=0.5)
max_val = max(np.max(y_test_orig), np.max(y_pred_orig))
plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label="Ideal")
plt.title("True vs Predicted RUL")
plt.xlabel("True RUL")
plt.ylabel("Predicted RUL")
plt.legend()
plt.grid(True, alpha=0.3)

# Prediction error histogram
plt.subplot(1, 3, 2)
errors = y_test_orig.flatten() - y_pred_orig.flatten()
plt.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
plt.title("Prediction Error Distribution")
plt.xlabel("Error (True - Predicted)")
plt.ylabel("Frequency")
plt.grid(True, alpha=0.3)

# Feature importance (compute correlation of features with RUL)
plt.subplot(1, 3, 3)
# Combine X_test with y_test for correlation analysis
test_data_for_corr = np.hstack([X_test, y_test])
feature_names = features + ['RUL']
test_df_for_corr = pd.DataFrame(test_data_for_corr, columns=feature_names)

# Calculate correlation with target (RUL)
correlations = test_df_for_corr.corr()['RUL'].drop('RUL')
top_corr = correlations.abs().sort_values(ascending=False).head(10)
top_features = top_corr.index

# Plot top correlations
colors = ['red' if c < 0 else 'green' for c in correlations[top_features]]
plt.barh(top_features, correlations[top_features], color=colors)
plt.title("Top Feature Correlations with RUL")
plt.xlabel("Correlation Coefficient")
plt.tight_layout()
plt.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('evaluation_results.png')
plt.show()

# ===================== FEATURE IMPORTANCE ANALYSIS =====================

# Calculate feature importance using permutation importance
from sklearn.inspection import permutation_importance

# Compute permutation importance (this might take some time)
perm_importance = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)

# Get feature importance scores
feature_importance = perm_importance.importances_mean
sorted_idx = np.argsort(feature_importance)[-15:]  # Top 15 features

# Plot feature importance
plt.figure(figsize=(10, 8))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
plt.title('Feature Importance (Permutation Method)')
plt.xlabel('Mean Decrease in MSE')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

print("\n✅ Model evaluation complete!")