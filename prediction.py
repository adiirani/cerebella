import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, recall_score, precision_score

# Download and load the dataset
# You can use these URLs to download the data
train_url = "http://azuremlsamples.azureml.net/templatedata/PM_train.txt"
test_url = "http://azuremlsamples.azureml.net/templatedata/PM_test.txt"
truth_url = "http://azuremlsamples.azureml.net/templatedata/PM_truth.txt"

# Load data
train_df = pd.read_csv(train_url, sep=" ", header=None)
test_df = pd.read_csv(test_url, sep=" ", header=None)
truth_df = pd.read_csv(truth_url, sep=" ", header=None)

# Clean the data
train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)

# Rename columns
column_names = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 
                's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 
                's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
train_df.columns = column_names
test_df.columns = column_names

# Sort data by ID and cycle
train_df = train_df.sort_values(['id', 'cycle'])
test_df = test_df.sort_values(['id', 'cycle'])

# Calculate remaining useful life (RUL) for training data
rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
train_df = train_df.merge(rul, on=['id'], how='left')
train_df['RUL'] = train_df['max'] - train_df['cycle']
train_df.drop('max', axis=1, inplace=True)

# Same for test data
rul = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
test_df = test_df.merge(rul, on=['id'], how='left')
test_df['RUL'] = test_df['max'] - test_df['cycle']
test_df.drop('max', axis=1, inplace=True)

# Create a binary classification label (1 if RUL <= 30, else 0)
w1 = 30
train_df['label'] = np.where(train_df['RUL'] <= w1, 1, 0)
test_df['label'] = np.where(test_df['RUL'] <= w1, 1, 0)

# Select relevant columns
sequence_cols = ['setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 
                's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']

# Normalize the data
scaler = MinMaxScaler()
train_df[sequence_cols] = scaler.fit_transform(train_df[sequence_cols])
test_df[sequence_cols] = scaler.transform(test_df[sequence_cols])

# Function to create sequences
def create_sequences(df, seq_length, sequence_cols):
    data = []
    labels = []
    
    for engine_id in df['id'].unique():
        engine_data = df[df['id'] == engine_id]
        
        if len(engine_data) >= seq_length:
            for i in range(len(engine_data) - seq_length + 1):
                seq = engine_data[sequence_cols].iloc[i:i+seq_length].values
                label = engine_data['label'].iloc[i+seq_length-1]
                data.append(seq)
                labels.append(label)
    
    return np.array(data), np.array(labels)

# Create sequences with a lookback of 50 cycles
seq_length = 50
X_train, y_train = create_sequences(train_df, seq_length, sequence_cols)
X_test, y_test = create_sequences(test_df, seq_length, sequence_cols)

# Build the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

# Define early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                 patience=10, 
                                                 mode='min')

# Train the model
history = model.fit(X_train, y_train, 
                    epochs=100, 
                    batch_size=32, 
                    validation_split=0.1,
                    callbacks=[early_stopping],
                    verbose=1)


# Evaluate on test data
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Calculate metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = 2 * (precision * recall) / (precision + recall)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# Save the model
model.save('turbofan_predictive_maintenance.h5')

# Function to make predictions on new data
def predict_maintenance(new_data, model, scaler, sequence_cols, seq_length):
    # Preprocess new data
    new_data[sequence_cols] = scaler.transform(new_data[sequence_cols])
    
    # Create sequences
    X_new, _ = create_sequences(new_data, seq_length, sequence_cols)
    
    # Make predictions
    predictions = model.predict(X_new)
    
    return predictions

# Example of loading the model for future use
# loaded_model = tf.keras.models.load_model('turbofan_predictive_maintenance.h5')

