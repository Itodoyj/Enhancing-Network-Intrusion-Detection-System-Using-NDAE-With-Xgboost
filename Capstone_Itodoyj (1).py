#!/usr/bin/env python
# coding: utf-8

# **Installing Needed Packages**

# In[1]:


#Needed Packages
import tensorflow as tf
import tensorflow_datasets as tfds
from google.cloud import storage
import pandas as pd
import io
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten, Concatenate
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import google.auth
from tensorflow.keras.optimizers import Adam
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# **Reading the KDD Cup 99 Data from Tensorflow into the notebook**

# In[ ]:


ds, info = tfds.load('kddcup99', with_info=True)


# In[ ]:


ds_split=ds['train']
df_train = tfds.as_dataframe(ds_split, info)


# In[ ]:


ds_split2=ds['test']
df_test = tfds.as_dataframe(ds_split2, info)


# **Connecting Notebook to Google Cloud Storage**

# In[2]:


get_ipython().system('gcloud auth application-default login')


# **Uploading Training Data to Cloud Storage**

# In[ ]:


# Define the bucket name and file name
bucket_name = "itodoyjcapstone"
file_name = "capstonetrain.csv"

# Convert the DataFrame to a CSV string
csv_string = df_train.to_csv(index=False)

# Create a storage client
storage_client = storage.Client()

# Upload the CSV string to the bucket
bucket = storage_client.get_bucket(bucket_name)
blob = bucket.blob(file_name)
blob.upload_from_string(csv_string, content_type="text/csv")

# Print the URL to the uploaded file
print(f"File uploaded to: gs://{bucket_name}/{file_name}")


# **Uploading Test Data to Cloud Storage**

# In[ ]:


# Define the bucket name and file name
bucket_name = "itodoyjcapstone"
file_name = "capstonetest.csv"

# Convert the DataFrame to a CSV string
csv_string = df_test.to_csv(index=False)

# Create a storage client
storage_client = storage.Client()

# Upload the CSV string to the bucket
bucket = storage_client.get_bucket(bucket_name)
blob = bucket.blob(file_name)
blob.upload_from_string(csv_string, content_type="text/csv")

# Print the URL to the uploaded file
print(f"File uploaded to: gs://{bucket_name}/{file_name}")


# **Downloading Training Data from Google Cloud Storage**

# In[3]:


# Explicitly specify the project ID
project_id = "hidden-circlet-414923"

# Create a storage client object with the specified project
storage_client = storage.Client(project=project_id)

# Create a bucket object for the desired bucket.
bucket = storage_client.get_bucket('itodoyjcapstone')

# Download the desired file as bytes.
blob1 = bucket.get_blob('capstonetrain.csv')
train_dat = blob1.download_as_string()

blob2 = bucket.get_blob('label.labels.txt')
label_dat = blob2.download_as_string()

blob3 = bucket.get_blob('protocol_type.labels.txt')
protocol_type = blob3.download_as_string()

blob4 = bucket.get_blob('service.labels.txt')
service_labels = blob4.download_as_string()

blob4 = bucket.get_blob('flag.labels.txt')
flag_labels = blob4.download_as_string()

# Convert the bytes to a DataFrame.
train_data = pd.read_csv(io.BytesIO(train_dat))
label_data = pd.read_csv(io.BytesIO(label_dat))
protocol_type_data = pd.read_csv(io.BytesIO(protocol_type))
service_labels_data = pd.read_csv(io.BytesIO(service_labels))
flag_labels_data = pd.read_csv(io.BytesIO(flag_labels))


# **Data** **Preprocessing**

# In[4]:


# Create a mapping dictionary with numbers starting from 1
mapping_dict_label = {i + 1: label for i, label in enumerate(label_data['label'])}
mapping_dict_protocol_type = {i: label for i, label in enumerate(protocol_type_data['label'])}
mapping_dict_service = {i: label for i, label in enumerate(service_labels_data['label'])}
mapping_dict_flag = {i: label for i, label in enumerate(flag_labels_data['label'])}

# Apply the mapping to the traning dataFrame
train_data['mapped_label'] = train_data['label'].map(mapping_dict_label)
train_data['mapped_protocol'] = train_data['protocol_type'].map(mapping_dict_protocol_type)
train_data['mapped_service'] = train_data['service'].map(mapping_dict_service)
train_data['mapped_flag'] = train_data['flag'].map(mapping_dict_flag)



# In[ ]:


print(train_data.isna().sum())


# In[ ]:


#Box Plot of the Data
# Calculate number of rows and columns for subplots
num_columns = len(train_data.columns)
rows = int(num_columns ** 0.5)
cols = rows if rows ** 2 >= num_columns else rows + 1

# Create a large figure to hold the subplots
plt.figure(figsize=(50, 50))

# Create a boxplot for each feature
for i, column in enumerate(train_data.columns):
    plt.subplot(rows, cols, i + 1)
    train_data.boxplot([column])  # Make sure to pass a list to boxplot
    plt.title(column)

# Add a main title and adjust the layout
plt.suptitle('Box Plots of Features', fontsize=35)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# Show the plot
plt.show()


# In[ ]:


# Count the occurrences of each label
label_counts = train_data['mapped_flag'].value_counts()

# Create a bar chart
label_counts.plot(kind='bar')

# Set the title and labels
plt.title('Count of Flag Labels')
plt.xlabel('Flag Label')
plt.ylabel('Count')

# Show the plot
plt.show()


# **Split Data into training and test sets**

# In[5]:


#Split the training data into X and Y
X = train_data.drop(['flag', 'mapped_flag','mapped_label','mapped_protocol','mapped_service' ], axis=1)
y = train_data['flag']

# Split the train data into validation and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Convert the NumPy arrays to pandas DataFrames
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')



# Print the shapes of the resulting datasets
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)
print('y_train shape:', y_train.shape)


# **Run NDAE on the Data**

# In[7]:


#The Input should be the number of features in X_train
input_dim = X_train.shape[1]
# Define the NDAE architecture
input_layer = Input(shape=(input_dim,))
encoded = Dense(32, activation='relu')(input_layer)  # Pass the output of the input layer
encoded = Dense(16, activation='relu')(encoded)      # Continue chaining outputs to inputs
decoded = Dense(32, activation='relu')(encoded)      # The output of the second Dense as input
output_layer = Dense(input_dim, activation='sigmoid')(decoded)  # Output should reconstruct the input

# Create the model using the input layer and the last output layer
autoencoder = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mse')

# Display the model architecture to ensure correctness
autoencoder.summary()

# Train the autoencoder
autoencoder.fit(X_train, X_train, epochs=50, batch_size=1000)

# Use the autoencoder to encode the input data
encoded_X_train = autoencoder.predict(X_train)
encoded_X_test = autoencoder.predict(X_test)


# In[8]:


# Convert the NumPy array to a pandas DataFrame
encoded_X_train_df= pd.DataFrame(encoded_X_train)
encoded_X_test_df= pd.DataFrame(encoded_X_test)
# Define the bucket name and file name
bucket_name = "itodoyjcapstone"
file_name = "encoded_X_train.csv"

# Convert the DataFrame to a CSV string
csv_string_encoded = encoded_X_train_df.to_csv(index=False)

# Create a storage client
storage_client = storage.Client(project=project_id)

# Upload the CSV string to the bucket
bucket = storage_client.get_bucket(bucket_name)
blob = bucket.blob(file_name)
blob.upload_from_string(csv_string_encoded, content_type="text/csv")

# Print the URL to the uploaded file
print(f"File uploaded to: gs://{bucket_name}/{file_name}")


# In[9]:


# Define the bucket name and file name
bucket_name = "itodoyjcapstone"
file_name = "encoded_X_test.csv"

# Convert the DataFrame to a CSV string
csv_string_encoded_test = encoded_X_test_df.to_csv(index=False)

# Create a storage client
storage_client = storage.Client(project=project_id)

# Upload the CSV string to the bucket
bucket = storage_client.get_bucket(bucket_name)
blob = bucket.blob(file_name)
blob.upload_from_string(csv_string_encoded_test, content_type="text/csv")

# Print the URL to the uploaded file
print(f"File uploaded to: gs://{bucket_name}/{file_name}")


# In[10]:


# Explicitly specify the project ID
project_id = "hidden-circlet-414923"

# Create a storage client object with the specified project
storage_client = storage.Client(project=project_id)

# Create a bucket object for the desired bucket.
bucket = storage_client.get_bucket('itodoyjcapstone')

# Download the desired file as bytes.
blob5 = bucket.get_blob('encoded_X_train.csv')
encoded_X_train_new1 = blob5.download_as_string()
blob6 = bucket.get_blob('encoded_X_test.csv')
encoded_X_test_new1 = blob6.download_as_string()

# Convert the bytes to a DataFrame.
encoded_X_train_new = pd.read_csv(io.BytesIO(encoded_X_train_new1))
encoded_X_test_new = pd.read_csv(io.BytesIO(encoded_X_test_new1))



# In[13]:


# Calculate weights manually for each class if necessary
weights = np.zeros(len(y_train))
unique_classes = np.unique(y_train)
for cls in unique_classes:
    class_weight = (y_train.size - np.sum(y_train == cls)) / np.sum(y_train == cls)
    weights[y_train == cls] = class_weight
print(weights)

# Define the XGBoost model
xgb_model = XGBClassifier(objective='multi:softmax', max_depth= 3, min_child_weight=1, gamma=0.5,
                              subsample=0.6, colsample_bytree= 0.6, learning_rate=0.01)

# Fit the grid search to the data
xgb_model.fit(encoded_X_train_new, y_train,sample_weight=weights )

# Make predictions on the test set
y_pred = xgb_model.predict(encoded_X_test_new)

# Evaluate the best model
accuracy = xgb_model.score(encoded_X_test_new, y_test)
print("Accuracy of the best model: {:.2f}% ", accuracy)

# Convert numpy arrays to pandas Series
y_pred_series = pd.Series(y_pred)
y_test_series = pd.Series(y_test)

# Use the mapping dictionary to map back to labels
y_pred_labels = y_pred_series.map(mapping_dict_flag)
y_test_labels = y_test_series.map(mapping_dict_flag)

# Print the classification report
print(classification_report(y_test_labels, y_pred_labels, zero_division=1))
conf_matrix = metrics.confusion_matrix(y_test_labels, y_pred_labels)
print('Confusion Matrix: ', conf_matrix)
# Initialize dictionaries to store FPR and FNR
FPR = {}
FNR = {}

for i in range(len(conf_matrix)):
    FN = conf_matrix[i, :].sum() - conf_matrix[i, i]
    FP = conf_matrix[:, i].sum() - conf_matrix[i, i]
    TN = conf_matrix.sum() - (conf_matrix[i, i] + FP + FN)
    FPR[i] = FP / (FP + TN)
    FNR[i] = FN / (FN + conf_matrix[i, i])

# Print the results
print("False Positive Rates by class:")
for class_label, rate in FPR.items():
    print(f"Class {class_label}: {rate}")

print("\nFalse Negative Rates by class:")
for class_label, rate in FNR.items():
    print(f"Class {class_label}: {rate}")


# In[14]:


# Calculate weights manually for each class if necessary
weights = np.zeros(len(y_train))
unique_classes = np.unique(y_train)
for cls in unique_classes:
    class_weight = (y_train.size - np.sum(y_train == cls)) / np.sum(y_train == cls)
    weights[y_train == cls] = class_weight
print(weights)
# Define the XGBoost model
xgb_model2 = XGBClassifier(objective='multi:softmax', max_depth= 6, min_child_weight=5, gamma=1,
                              subsample=0.8, colsample_bytree= 0.8, learning_rate=0.1)

# Fit the grid search to the data
xgb_model2.fit(encoded_X_train_new, y_train, sample_weight=weights)

# Make predictions on the test set
y_pred2 = xgb_model2.predict(encoded_X_test_new)

# Evaluate the best model
accuracy = xgb_model2.score(encoded_X_test_new, y_test)
print("Accuracy of the model: {:.2f}% ", accuracy)
# Convert numpy arrays to pandas Series
y_pred_series = pd.Series(y_pred2)
y_test_series = pd.Series(y_test)

# Use the mapping dictionary to map back to labels
y_pred_labels = y_pred_series.map(mapping_dict_flag)
y_test_labels = y_test_series.map(mapping_dict_flag)

# Print the classification report
print(classification_report(y_test_labels, y_pred_labels, zero_division=1))
conf_matrix = metrics.confusion_matrix(y_test, y_pred2)
print('Confusion Matrix: ', conf_matrix)
# Calculate FPR and FNR for each class
# Initialize dictionaries to store FPR and FNR
FPR = {}
FNR = {}

for i in range(len(conf_matrix)):
    FN = conf_matrix[i, :].sum() - conf_matrix[i, i]
    FP = conf_matrix[:, i].sum() - conf_matrix[i, i]
    TN = conf_matrix.sum() - (conf_matrix[i, i] + FP + FN)
    FPR[i] = FP / (FP + TN)
    FNR[i] = FN / (FN + conf_matrix[i, i])

# Print the results
print("False Positive Rates by class:")
for class_label, rate in FPR.items():
    print(f"Class {class_label}: {rate}")

print("\nFalse Negative Rates by class:")
for class_label, rate in FNR.items():
    print(f"Class {class_label}: {rate}")


# In[15]:


# Define the XGBoost model
xgb_model3 = XGBClassifier(objective='multi:softmax', max_depth= 9, min_child_weight=10, gamma=1,
                              subsample=0.8, colsample_bytree= 0.8, learning_rate=0.1)

# Fit the grid search to the data
xgb_model3.fit(encoded_X_train_new, y_train,sample_weight=weights)

# Make predictions on the test set
y_pred3 = xgb_model3.predict(encoded_X_test_new)

# Evaluate the best model
accuracy = xgb_model3.score(encoded_X_test_new, y_test)
print("Accuracy of the best model: {:.2f}% ", accuracy)
# Convert numpy arrays to pandas Series
y_pred_series = pd.Series(y_pred3)
y_test_series = pd.Series(y_test)

# Use the mapping dictionary to map back to labels
y_pred_labels = y_pred_series.map(mapping_dict_flag)
y_test_labels = y_test_series.map(mapping_dict_flag)

# Print the classification report
print(classification_report(y_test_labels, y_pred_labels, zero_division=1))
conf_matrix = metrics.confusion_matrix(y_test, y_pred3)
print('Confusion Matrix: ', conf_matrix)

# Initialize dictionaries to store FPR and FNR
FPR = {}
FNR = {}
# Calculate FPR and FNR for each class
for i in range(len(conf_matrix)):
    FN = conf_matrix[i, :].sum() - conf_matrix[i, i]
    FP = conf_matrix[:, i].sum() - conf_matrix[i, i]
    TN = conf_matrix.sum() - (conf_matrix[i, i] + FP + FN)
    FPR[i] = FP / (FP + TN)
    FNR[i] = FN / (FN + conf_matrix[i, i])

# Print the results
print("False Positive Rates by class:")
for class_label, rate in FPR.items():
    print(f"Class {class_label}: {rate}")

print("\nFalse Negative Rates by class:")
for class_label, rate in FNR.items():
    print(f"Class {class_label}: {rate}")


# In[17]:


# Define the XGBoost model
xgb_model4 = XGBClassifier(objective='multi:softmax', max_depth= 9, min_child_weight=10, gamma=1,
                              subsample=0.8, colsample_bytree= 0.8, learning_rate=0.1, alpha=1)
weights = {6: 3, 7: 5, 8: 10}  # Increase weights for classes 6, 7, and 8
sample_weights = np.ones(y_train.shape[0])
for class_index, weight in weights.items():
    sample_weights[y_train == class_index] = weight


# Fit the grid search to the data
xgb_model4.fit(encoded_X_train_new, y_train,sample_weight=sample_weights)

# Make predictions on the test set
y_pred4 = xgb_model4.predict(encoded_X_test_new)

# Evaluate the best model
accuracy = xgb_model4.score(encoded_X_test_new, y_test)
print("Accuracy of the best model: {:.2f}% ", accuracy)
# Convert numpy arrays to pandas Series
y_pred_series = pd.Series(y_pred4)
y_test_series = pd.Series(y_test)

# Use the mapping dictionary to map back to labels
y_pred_labels = y_pred_series.map(mapping_dict_flag)
y_test_labels = y_test_series.map(mapping_dict_flag)

# Print the classification report
print(classification_report(y_test_labels, y_pred_labels, zero_division=1))
conf_matrix = metrics.confusion_matrix(y_test, y_pred4)
print('Confusion Matrix: ', conf_matrix)
# Initialize dictionaries to store FPR and FNR
FPR = {}
FNR = {}
# Calculate FPR and FNR for each class
for i in range(len(conf_matrix)):
    FN = conf_matrix[i, :].sum() - conf_matrix[i, i]
    FP = conf_matrix[:, i].sum() - conf_matrix[i, i]
    TN = conf_matrix.sum() - (conf_matrix[i, i] + FP + FN)
    FPR[i] = FP / (FP + TN)
    FNR[i] = FN / (FN + conf_matrix[i, i])

# Print the results
print("False Positive Rates by class:")
for class_label, rate in FPR.items():
    print(f"Class {class_label}: {rate}")

print("\nFalse Negative Rates by class:")
for class_label, rate in FNR.items():
    print(f"Class {class_label}: {rate}")

    print(f"{class_label}: {rate}")


# In[18]:


# Calculate weights manually for each class if necessary
weights = np.zeros(len(y_train))
unique_classes = np.unique(y_train)
for cls in unique_classes:
    class_weight = (y_train.size - np.sum(y_train == cls)) / np.sum(y_train == cls)
    weights[y_train == cls] = class_weight
print(weights)
# Define the XGBoost model
xgb_model5 = XGBClassifier(objective='multi:softmax', max_depth= 9, min_child_weight=10, gamma=1,
                              subsample=0.8, colsample_bytree= 0.8, learning_rate=0.2, alpha=100)

# Fit the grid search to the data
xgb_model5.fit(encoded_X_train_new, y_train,sample_weight=weights)

# Make predictions on the test set
y_pred5 = xgb_model5.predict(encoded_X_test_new)

# Evaluate the best model
accuracy = xgb_model5.score(encoded_X_test_new, y_test)
print("Accuracy of the best model: {:.2f}% ", accuracy)
# Convert numpy arrays to pandas Series
y_pred_series = pd.Series(y_pred5)
y_test_series = pd.Series(y_test)

# Use the mapping dictionary to map back to labels
y_pred_labels = y_pred_series.map(mapping_dict_flag)
y_test_labels = y_test_series.map(mapping_dict_flag)

# Print the classification report
print(classification_report(y_test_labels, y_pred_labels, zero_division=1))
conf_matrix = metrics.confusion_matrix(y_test_labels, y_pred_labels)
print('Confusion Matrix: ', conf_matrix)

# Initialize dictionaries to store FPR and FNR
FPR = {}
FNR = {}
# Calculate FPR and FNR for each class
for i in range(len(conf_matrix)):
    FN = conf_matrix[i, :].sum() - conf_matrix[i, i]
    FP = conf_matrix[:, i].sum() - conf_matrix[i, i]
    TN = conf_matrix.sum() - (conf_matrix[i, i] + FP + FN)
    FPR[i] = FP / (FP + TN)
    FNR[i] = FN / (FN + conf_matrix[i, i])

# Print the results
print("False Positive Rates by class:")
for class_label, rate in FPR.items():
    print(f"Class {class_label}: {rate}")

print("\nFalse Negative Rates by class:")
for class_label, rate in FNR.items():
    print(f"Class {class_label}: {rate}")


# **Download Validation data from google cloud storage**

# In[19]:


# Explicitly specify the project ID
project_id = "hidden-circlet-414923"

# Create a storage client object with the specified project
storage_client = storage.Client(project=project_id)

# Create a bucket object for the desired bucket.
bucket = storage_client.get_bucket('itodoyjcapstone')

# Download the desired file as bytes.
blob7 = bucket.get_blob('capstonetest.csv')
validation_dat = blob7.download_as_string()

# Convert the bytes to a DataFrame.
val_dat = pd.read_csv(io.BytesIO(validation_dat))


# **Split the validation data into X and Y variables**

# In[20]:


X_val = val_dat.drop('flag', axis=1)
y_val = val_dat['flag']

# Convert numpy arrays to pandas Series
y_val_series = pd.Series(y_val)

# Use the mapping dictionary to map back to labels
y_val_labels = y_val_series.map(mapping_dict_flag)
X_val = X_val.astype('float32')


# In[21]:


#The Input should be the number of features in X_train
input_dim = X_val.shape[1]
# Define the NDAE architecture
input_layer = Input(shape=(input_dim,))
encoded = Dense(32, activation='relu')(input_layer)  # Pass the output of the input layer
encoded = Dense(16, activation='relu')(encoded)      # Continue chaining outputs to inputs
decoded = Dense(32, activation='relu')(encoded)      # The output of the second Dense as input
output_layer = Dense(input_dim, activation='sigmoid')(decoded)  # Output should reconstruct the input

# Create the model using the input layer and the last output layer
autoencoder = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mse')

# Display the model architecture to ensure correctness
autoencoder.summary()

# Train the autoencoder
autoencoder.fit(X_val, X_val, epochs=50, batch_size=100)

# Use the autoencoder to encode the input data
encoded_X_val = autoencoder.predict(X_val)


# In[22]:


# Make predictions on the Validation set
y_pred6 = xgb_model4.predict(encoded_X_val)

# Evaluate the best model
accuracy = xgb_model4.score(encoded_X_val, y_val)
print("Accuracy of the best model: {:.2f}% ", accuracy)
# Convert numpy arrays to pandas Series
y_pred_series2 = pd.Series(y_pred6)

# Use the mapping dictionary to map back to labels
y_pred_labels2 = y_pred_series2.map(mapping_dict_flag)

# Print the classification report
print(classification_report(y_val_labels, y_pred_labels2, zero_division=1))
conf_matrix = metrics.confusion_matrix(y_val_labels, y_pred_labels2)
print('Confusion Matrix: ', conf_matrix)


# In[19]:


# Make predictions on the Validation set
y_pred7 = xgb_model.predict(encoded_X_val)

# Evaluate the best model
accuracy = xgb_model.score(encoded_X_val, y_val)
print("Accuracy of the best model: {:.2f}% ", accuracy)
# Convert numpy arrays to pandas Series
y_pred_series3 = pd.Series(y_pred7)

# Use the mapping dictionary to map back to labels
y_pred_labels3 = y_pred_series3.map(mapping_dict_flag)

# Print the classification report
print(classification_report(y_val_labels, y_pred_labels3, zero_division=1))
conf_matrix = metrics.confusion_matrix(y_val_labels, y_pred_labels3)
print('Confusion Matrix: ', conf_matrix)


# In[ ]:




