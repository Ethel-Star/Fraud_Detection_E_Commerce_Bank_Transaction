import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import logging

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename="logs/data_processing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w"  # Overwrites the log file each run; use "a" to append
)

def display_data_shapes(fraud_data, creditcard_data, ip_to_country):
    """
    Display the shapes of the datasets and log the details.
    """
    logging.info("Displaying dataset shapes...")
    print("\nFraud Data Shape:", fraud_data.shape)
    print("Credit Card Data Shape:", creditcard_data.shape)
    print("IP to Country Data Shape:", ip_to_country.shape)
    
    logging.info(f"Fraud Data Shape: {fraud_data.shape}")
    logging.info(f"Credit Card Data Shape: {creditcard_data.shape}")
    logging.info(f"IP to Country Data Shape: {ip_to_country.shape}")

def display_data_info(fraud_data, creditcard_data, ip_to_country):
    """
    Display information about the datasets (column types, non-null counts, etc.) and log the details.
    """
    logging.info("Displaying dataset info...")
    print("\nFraud Data Info:")
    print(fraud_data.info())
    print("\nCredit Card Data Info:")
    print(creditcard_data.info())
    print("\nIP to Country Data Info:")
    print(ip_to_country.info())

    logging.info("Fraud Data Info:")
    logging.info(fraud_data.info())
    logging.info("Credit Card Data Info:")
    logging.info(creditcard_data.info())
    logging.info("IP to Country Data Info:")
    logging.info(ip_to_country.info())

def check_missing_values(fraud_data, creditcard_data, ip_to_country):
    """
    Check for missing values in each dataset and log the missing value counts.
    """
    logging.info("Checking for missing values...")
    fraud_data_missing = fraud_data.isnull().sum()
    creditcard_data_missing = creditcard_data.isnull().sum()
    ip_to_country_missing = ip_to_country.isnull().sum()

    print("\nMissing Values in Fraud Data:")
    print(fraud_data_missing)
    print("\nMissing Values in Credit Card Data:")
    print(creditcard_data_missing)
    print("\nMissing Values in IP to Country Data:")
    print(ip_to_country_missing)

    logging.info(f"Missing Values in Fraud Data:\n{fraud_data_missing}")
    logging.info(f"Missing Values in Credit Card Data:\n{creditcard_data_missing}")
    logging.info(f"Missing Values in IP to Country Data:\n{ip_to_country_missing}")

    return fraud_data_missing, creditcard_data_missing, ip_to_country_missing

def check_duplicates(fraud_data, creditcard_data, ip_to_country):
    """
    Check for duplicate rows in each dataset and remove them, logging the changes.
    """
    datasets = {
        "Fraud Data": fraud_data,
        "Credit Card Data": creditcard_data,
        "IP to Country Data": ip_to_country
    }

    for name, data in datasets.items():
        duplicates = data.duplicated().sum()
        logging.info(f"Checking for duplicates in {name}...")
        if duplicates > 0:
            print(f"\n{name}:")
            print(f"⚠️ Found {duplicates} duplicate rows. Removing them...")
            datasets[name] = data.drop_duplicates()
            logging.info(f"Found {duplicates} duplicates in {name}. Removing them.")
            print("✅ Duplicates removed successfully.")
        else:
            logging.info(f"No duplicates found in {name}.")
            print("✅ No duplicates found.")

    return datasets["Fraud Data"], datasets["Credit Card Data"], datasets["IP to Country Data"]

def plot_target_class_distribution(fraud_data, creditcard_data):
    """
    Plot the distribution of the target variable ('class' or 'Class') for Fraud and Credit Card datasets and log the outputs.
    """
    logging.info("Plotting target class distribution...")
    
    # Group by target variable 'class' in Fraud_Data
    fraud_class_counts = fraud_data['class'].value_counts()
    fraud_class_percentage = fraud_data['class'].value_counts(normalize=True) * 100

    # Group by target variable 'Class' in creditcard dataset
    creditcard_class_counts = creditcard_data['Class'].value_counts()
    creditcard_class_percentage = creditcard_data['Class'].value_counts(normalize=True) * 100

    # Displaying counts and percentages for both datasets
    print("\nFraud_Data Class Distribution Count:")
    print(fraud_class_counts)
    print("\nFraud_Data Class Distribution Percentage:")
    print(fraud_class_percentage)

    print("\nCreditCard Class Distribution Count:")
    print(creditcard_class_counts)
    print("\nCreditCard Class Distribution Percentage:")
    print(creditcard_class_percentage)

    logging.info(f"Fraud_Data Class Distribution Count:\n{fraud_class_counts}")
    logging.info(f"Fraud_Data Class Distribution Percentage:\n{fraud_class_percentage}")
    logging.info(f"CreditCard Class Distribution Count:\n{creditcard_class_counts}")
    logging.info(f"CreditCard Class Distribution Percentage:\n{creditcard_class_percentage}")

    # Create 3D Pie Charts using Plotly
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]], 
                        subplot_titles=("Fraud_Data Target Class Distribution", "CreditCard Target Class Distribution"))

    # Fraud Data 3D Pie Chart
    fig.add_trace(go.Pie(
        labels=['Non-Fraudulent (0)', 'Fraudulent (1)'],
        values=fraud_class_counts,
        hole=0.4,
        pull=[0, 0.1],  # Pull out the "Fraudulent" slice
        marker=dict(colors=['#66b3ff', '#ff6666']),
        textinfo='percent+label',
        hoverinfo='label+percent',
        name="Fraud_Data"
    ), 1, 1)

    # Credit Card 3D Pie Chart
    fig.add_trace(go.Pie(
        labels=['Non-Fraudulent (0)', 'Fraudulent (1)'],
        values=creditcard_class_counts,
        hole=0.4,
        pull=[0, 0.1],  # Pull out the "Fraudulent" slice
        marker=dict(colors=['#66b3ff', '#ff6666']),
        textinfo='percent+label',
        hoverinfo='label+percent',
        name="CreditCard"
    ), 1, 2)

    # Update layout for 3D effect and annotations
    fig.update_layout(
        title_text="Target Class Distribution in Fraud_Data and CreditCard Datasets",
        title_font_size=20,
        title_x=0.5,
        annotations=[dict(text='Fraud_Data', x=0.18, y=0.5, font_size=14, showarrow=False),
                     dict(text='CreditCard', x=0.82, y=0.5, font_size=14, showarrow=False)],
        showlegend=False
    )

    # Add 3D effect
    fig.update_traces(hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
                      textfont=dict(size=14, family="Arial"))

    # Show the interactive 3D pie charts
    fig.show()

def process_data(fraud_data, creditcard_data, ip_to_country):
    """
    Main function to process and analyze the datasets, with logging.
    """
    logging.info("Starting data processing...")
    display_data_shapes(fraud_data, creditcard_data, ip_to_country)
    display_data_info(fraud_data, creditcard_data, ip_to_country)
    check_missing_values(fraud_data, creditcard_data, ip_to_country)
    check_duplicates(fraud_data, creditcard_data, ip_to_country)
    plot_target_class_distribution(fraud_data, creditcard_data)
    logging.info("Data processing completed.")
