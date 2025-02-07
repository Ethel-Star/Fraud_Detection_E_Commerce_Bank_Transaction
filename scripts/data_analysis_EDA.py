import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename="logs/data_analysis_EDA.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w"  # Overwrites the log file each run; use "a" to append
)

def plot_fraud_data_distributions(fraud_data):
    """
    Plot distributions for 'purchase_value', 'age', 'source', and 'browser' in fraud_data.
    """
    print("\nPlotting distributions for fraud_data...")
    logging.info("Plotting distributions for fraud_data.")
    try:
        # Set up the figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Plot 'purchase_value' - Histogram
        print("Plotting 'purchase_value' histogram...")
        sns.histplot(fraud_data['purchase_value'], bins=50, kde=True, color='skyblue', ax=axes[0, 0])
        axes[0, 0].set_title('Distribution of Purchase Value')
        axes[0, 0].set_xlabel('Purchase Value ($)')
        axes[0, 0].set_ylabel('Frequency')

        # Plot 'age' - Histogram
        print("Plotting 'age' histogram...")
        sns.histplot(fraud_data['age'], bins=50, kde=True, color='lightgreen', ax=axes[0, 1])
        axes[0, 1].set_title('Distribution of Age')
        axes[0, 1].set_xlabel('Age')
        axes[0, 1].set_ylabel('Frequency')

        # Plot 'source' - Bar Chart
        print("Plotting 'source' bar chart...")
        sns.countplot(x='source', data=fraud_data, ax=axes[1, 0])
        axes[1, 0].set_title('Distribution of Source')
        axes[1, 0].set_xlabel('Source')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Plot 'browser' - Bar Chart
        print("Plotting 'browser' bar chart...")
        sns.countplot(x='browser', data=fraud_data, ax=axes[1, 1])
        axes[1, 1].set_title('Distribution of Browser')
        axes[1, 1].set_xlabel('Browser')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()
        print("Successfully plotted fraud_data distributions.")
        logging.info("Successfully plotted fraud_data distributions.")
    except Exception as e:
        print(f"Error plotting fraud_data distributions: {e}")
        logging.error(f"Error plotting fraud_data distributions: {e}")

def plot_creditcard_data_distributions(creditcard_data):
    """
    Plot distributions for 'Time', 'Amount', and 'Class' in creditcard_data.
    """
    print("\nPlotting distributions for creditcard_data...")
    logging.info("Plotting distributions for creditcard_data.")
    try:
        # Set up the figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot 'Time' - Histogram
        print("Plotting 'Time' histogram...")
        sns.histplot(creditcard_data['Time'], bins=50, kde=True, color='lightblue', ax=axes[0])
        axes[0].set_title('Distribution of Time', fontsize=14)
        axes[0].set_xlabel('Time (seconds)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)

        # Plot 'Amount' - Histogram
        print("Plotting 'Amount' histogram...")
        sns.histplot(creditcard_data['Amount'], bins=50, kde=True, color='lightgreen', ax=axes[1])
        axes[1].set_title('Distribution of Transaction Amount', fontsize=14)
        axes[1].set_xlabel('Amount ($)', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)

        # Plot 'Class' - Bar Chart (Fraudulent vs Non-Fraudulent)
        print("Plotting 'Class' bar chart...")
        sns.countplot(x='Class', data=creditcard_data, ax=axes[2])
        axes[2].set_title('Fraudulent vs Non-Fraudulent Transactions', fontsize=14)
        axes[2].set_xlabel('Class', fontsize=12)
        axes[2].set_ylabel('Count', fontsize=12)

        # Fix the x-tick labels for the 'Class' bar chart
        axes[2].set_xticks([0, 1])  # Explicitly set tick positions
        axes[2].set_xticklabels(['Non-Fraudulent', 'Fraudulent'], fontsize=12)  # Now safe to use

        # Add annotations to the bar chart
        for p in axes[2].patches:
            axes[2].annotate(f'{int(p.get_height())}', 
                             (p.get_x() + p.get_width() / 2., p.get_height()), 
                             ha='center', va='center', 
                             xytext=(0, 10), 
                             textcoords='offset points',
                             fontsize=12)

        # Adjust layout and display
        plt.tight_layout()
        plt.show()
        print("Successfully plotted creditcard_data distributions.")
        logging.info("Successfully plotted creditcard_data distributions.")
    except Exception as e:
        print(f"Error plotting creditcard_data distributions: {e}")
        logging.error(f"Error plotting creditcard_data distributions: {e}")

def plot_fraud_data_relationships(fraud_data):
    """
    Plot relationships between features in fraud_data.
    """
    print("\nPlotting relationships in fraud_data...")
    logging.info("Plotting relationships in fraud_data.")
    try:
        # Set style
        sns.set_style("whitegrid")

        # Set up figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))

        # 1. Purchase Value vs Age (Fraud & Non-Fraud)
        print("Plotting Purchase Value vs Age...")
        sns.scatterplot(x='age', y='purchase_value', hue='class', data=fraud_data, alpha=0.5, palette='coolwarm', ax=axes[0, 0])
        axes[0, 0].set_title('Purchase Value vs Age (Fraud & Non-Fraud)')
        axes[0, 0].set_xlabel('Age')
        axes[0, 0].set_ylabel('Purchase Value ($)')

        # 2. Purchase Value vs Source (Fraud & Non-Fraud)
        print("Plotting Purchase Value vs Source...")
        sns.boxplot(x='source', y='purchase_value', hue='class', data=fraud_data, palette='viridis', ax=axes[0, 1])
        axes[0, 1].set_title('Purchase Value vs Source (Fraud & Non-Fraud)')
        axes[0, 1].set_xlabel('Source')
        axes[0, 1].set_ylabel('Purchase Value ($)')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # 3. Age vs Source (Fraud & Non-Fraud)
        print("Plotting Age vs Source...")
        sns.boxplot(x='source', y='age', hue='class', data=fraud_data, palette='coolwarm', ax=axes[1, 0])
        axes[1, 0].set_title('Age vs Source (Fraud & Non-Fraud)')
        axes[1, 0].set_xlabel('Source')
        axes[1, 0].set_ylabel('Age')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # 4. Age vs Browser (Fraud & Non-Fraud)
        print("Plotting Age vs Browser...")
        sns.boxplot(x='browser', y='age', hue='class', data=fraud_data, palette='magma', ax=axes[1, 1])
        axes[1, 1].set_title('Age vs Browser (Fraud & Non-Fraud)')
        axes[1, 1].set_xlabel('Browser')
        axes[1, 1].set_ylabel('Age')
        axes[1, 1].tick_params(axis='x', rotation=90)

        # 5. Fraud Distribution across Age
        print("Plotting Fraud Distribution across Age...")
        sns.histplot(fraud_data, x="age", hue="class", multiple="stack", palette="coolwarm", kde=True, ax=axes[2, 0])
        axes[2, 0].set_title('Fraud Distribution across Age')
        axes[2, 0].set_xlabel('Age')
        axes[2, 0].set_ylabel('Count')

        # 6. Purchase Time vs Class (Fraud & Non-Fraud)
        print("Plotting Purchase Time vs Class...")
        fraud_data['purchase_hour'] = pd.to_datetime(fraud_data['purchase_time']).dt.hour
        sns.histplot(fraud_data, x="purchase_hour", hue="class", multiple="stack", palette="viridis", kde=True, ax=axes[2, 1])
        axes[2, 1].set_title('Fraud vs Purchase Time')
        axes[2, 1].set_xlabel('Purchase Hour')
        axes[2, 1].set_ylabel('Count')

        plt.tight_layout()
        plt.show()
        print("Successfully plotted fraud_data relationships.")
        logging.info("Successfully plotted fraud_data relationships.")
    except Exception as e:
        print(f"Error plotting fraud_data relationships: {e}")
        logging.error(f"Error plotting fraud_data relationships: {e}")

def plot_correlation_analysis(fraud_data, creditcard_data):
    """
    Perform correlation analysis and plot heatmaps for fraud_data and creditcard_data.
    """
    print("\nPerforming correlation analysis...")
    logging.info("Performing correlation analysis.")
    try:
        # Ensure only numeric columns for correlation
        numeric_fraud_data = fraud_data.select_dtypes(include=['number'])
        numeric_credit_data = creditcard_data.select_dtypes(include=['number'])

        # ----- First Figure: Fraud Data (1 row, 2 columns) -----
        print("Plotting Fraud Data heatmap and scatter plot...")
        fig1, axs1 = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns

        # Heatmap for Fraud Data
        corr_fraud = numeric_fraud_data.corr()
        sns.heatmap(corr_fraud, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=axs1[0])
        axs1[0].set_title("Fraud Data Correlation Heatmap")

        # Scatter Plot for Fraud Data (Colorful)
        sns.scatterplot(x=numeric_fraud_data['purchase_value'], y=numeric_fraud_data['age'], hue=numeric_fraud_data['class'], palette="viridis", alpha=0.7, ax=axs1[1])
        axs1[1].set_title("Fraud Data: Purchase Value vs Age (Color by Class)")

        plt.tight_layout()
        plt.show()

        # ----- Second Figure: Credit Card Data (Only Heatmap) -----
        print("Plotting Credit Card Data heatmap...")
        fig2, ax2 = plt.subplots(figsize=(14, 8))  # Single heatmap

        # Heatmap for Credit Card Data
        corr_credit = numeric_credit_data.corr()
        sns.heatmap(corr_credit, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax2)
        ax2.set_title("Credit Card Data Correlation Heatmap")

        plt.tight_layout()
        plt.show()
        print("Successfully performed correlation analysis.")
        logging.info("Successfully performed correlation analysis.")
    except Exception as e:
        print(f"Error performing correlation analysis: {e}")
        logging.error(f"Error performing correlation analysis: {e}")

def map_ip_to_country(fraud_data, ip_to_country):
    """
    Map IP addresses in fraud_data to countries using ip_to_country.
    """
    print("\nMapping IP addresses to countries...")
    logging.info("Mapping IP addresses to countries.")
    try:
        # Convert fraud_data IP address to integer format
        fraud_data['ip_address'] = fraud_data['ip_address'].astype(int)

        # Create a function to check if IP address falls within the range
        def find_country_by_ip(ip):
            # Find the row where IP falls within the range
            matched_row = ip_to_country[(ip >= ip_to_country['lower_bound_ip_address']) & 
                                       (ip <= ip_to_country['upper_bound_ip_address'])]
            if not matched_row.empty:
                return matched_row['country'].values[0]
            else:
                return 'Unknown'  # Handle cases where no match is found

        # Apply the function to fraud_data
        fraud_data['country'] = fraud_data['ip_address'].apply(find_country_by_ip)
        print("Successfully mapped IP addresses to countries.")
        logging.info("Successfully mapped IP addresses to countries.")
        return fraud_data
    except Exception as e:
        print(f"Error mapping IP addresses to countries: {e}")
        logging.error(f"Error mapping IP addresses to countries: {e}")
        return fraud_data

def save_data(fraud_data, output_path):
    """
    Save the processed fraud_data to a CSV file.
    """
    print(f"\nSaving processed data to {output_path}...")
    logging.info(f"Saving processed data to {output_path}.")
    try:
        fraud_data.to_csv(output_path, index=False)
        print(f"Data successfully saved to {output_path}.")
        logging.info(f"Data successfully saved to {output_path}.")
    except Exception as e:
        print(f"Error saving data to {output_path}: {e}")
        logging.error(f"Error saving data to {output_path}: {e}")

def perform_eda(fraud_data, creditcard_data, ip_to_country, output_path):
    """
    Perform exploratory data analysis (EDA) on the datasets.
    """
    print("\nStarting exploratory data analysis (EDA)...")
    logging.info("Starting exploratory data analysis (EDA).")
    plot_fraud_data_distributions(fraud_data)
    plot_creditcard_data_distributions(creditcard_data)
    plot_fraud_data_relationships(fraud_data)
    plot_correlation_analysis(fraud_data, creditcard_data)
    fraud_data = map_ip_to_country(fraud_data, ip_to_country)
    save_data(fraud_data, output_path)
    print("\nExploratory data analysis (EDA) completed.")
    logging.info("Exploratory data analysis (EDA) completed.")