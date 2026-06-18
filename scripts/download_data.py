import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

def download_data():
    print("Fetching Heart Disease Dataset from UCI ML Repository...")
    # fetch dataset 
    heart_disease = fetch_ucirepo(id=45) 
    
    # data (as pandas dataframes) 
    X = heart_disease.data.features 
    y = heart_disease.data.targets 
    
    # Combine features and targets
    df = pd.concat([X, y], axis=1)
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    output_path = os.path.join('data', 'heart_disease.csv')
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully saved to {output_path}")
    print(f"Dataset shape: {df.shape}")
    print("First 5 rows:")
    print(df.head())

if __name__ == "__main__":
    download_data()
