import os
import logging
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

# Setup logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "feature_engineering.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def feature_engineering(fraud_data, output_future_engineered):
    try:
        print("Starting feature engineering...")
        logging.info("Starting feature engineering...")
        
        # Calculate transaction frequency per user
        print("Calculating transaction frequency per user...")
        transaction_frequency = fraud_data.groupby('user_id').size().reset_index(name='transaction_frequency')
        fraud_data = fraud_data.merge(transaction_frequency, on='user_id', how='left')
        logging.info("Transaction frequency calculated and merged.")
        
        # Convert purchase_time to datetime
        print("Converting 'purchase_time' to datetime format...")
        fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])
        
        # Calculate transaction velocity
        print("Calculating transaction velocity...")
        user_transaction_times = fraud_data.groupby('user_id')['purchase_time'].agg(['min', 'max'])
        user_transaction_times['transaction_velocity'] = (
            (user_transaction_times['max'] - user_transaction_times['min']).dt.total_seconds() /
            fraud_data.groupby('user_id').size()
        )
        fraud_data = fraud_data.merge(user_transaction_times[['transaction_velocity']], on='user_id', how='left')
        logging.info("Transaction velocity calculated and merged.")
        
        # Extract time-based features
        print("Extracting time-based features (hour_of_day, day_of_week)...")
        fraud_data['hour_of_day'] = fraud_data['purchase_time'].dt.hour
        fraud_data['day_of_week'] = fraud_data['purchase_time'].dt.dayofweek
        logging.info("Time-based features extracted.")
        
        # Normalize purchase_value, transaction_frequency, transaction_velocity
        print("Applying MinMax scaling...")
        scaler = MinMaxScaler()
        fraud_data[['purchase_value', 'transaction_frequency', 'transaction_velocity']] = scaler.fit_transform(
            fraud_data[['purchase_value', 'transaction_frequency', 'transaction_velocity']]
        )
        logging.info("MinMax scaling applied.")
        
        # Standardize the 'age' column
        print("Applying Standard scaling to 'age' column...")
        scaler = StandardScaler()
        fraud_data['age'] = scaler.fit_transform(fraud_data[['age']])
        logging.info("Standard scaling applied to 'age'.")
        
        # Convert datetime columns to timestamps
        print("Converting datetime columns to timestamps...")
        fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time']).astype('int64') // 10**9
        fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time']).astype('int64') // 10**9
        logging.info("Datetime columns converted to timestamps.")
        
        # One-hot encode categorical columns
        print("Performing one-hot encoding on categorical columns...")
        fraud_data = pd.get_dummies(fraud_data, columns=['source', 'browser', 'country'], drop_first=True)
        logging.info("One-hot encoding applied.")
        
        # Label encode 'sex' column
        print("Applying label encoding to 'sex' column...")
        label_encoder = LabelEncoder()
        fraud_data['sex'] = label_encoder.fit_transform(fraud_data['sex'])
        logging.info("Label encoding applied to 'sex'.")
        
        # Convert boolean columns to integers
        print("Converting boolean columns to integers...")
        boolean_columns = fraud_data.select_dtypes(include=['bool']).columns
        fraud_data[boolean_columns] = fraud_data[boolean_columns].astype(int)
        logging.info("Boolean columns converted to integers.")
        
        # Save processed data
        print(f"Saving processed data to {output_future_engineered}...")
        fraud_data.to_csv(output_future_engineered, index=False)
        logging.info(f"Data successfully saved to {output_future_engineered}.")
        print("Feature engineering complete!")
    
    except Exception as e:
        logging.error(f"Error during feature engineering: {str(e)}")
        print(f"Error: {str(e)}")

