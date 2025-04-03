# clean_styles_csv.py
import pandas as pd
import csv

# Read the CSV with proper quoting and skip bad lines
df = pd.read_csv('data/styles.csv', quoting=csv.QUOTE_ALL, on_bad_lines='skip')

# Ensure only the required columns are kept
required_columns = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'year']
df = df[required_columns]

# Add a meaningful 'usage' column based on articleType or subCategory
# Example mapping (you can adjust based on your domain knowledge)
usage_mapping = {
    'Tshirts': 'Casual',
    'Shirts': 'Formal',
    'Dresses': 'Casual',
    'Jeans': 'Casual',
    'Topwear': 'Casual',
    'Bottomwear': 'Casual',
    'Footwear': 'Sportswear',
    'Accessories': 'Fashionable'
}

# Assign 'usage' based on articleType or subCategory
df['usage'] = df['articleType'].map(usage_mapping).fillna(
    df['subCategory'].map(usage_mapping)
).fillna('Casual')  # Default to 'Casual' if no mapping

# Debug: Print the class distribution of 'usage'
print("Class distribution in 'usage':")
print(df['usage'].value_counts())

# Save the cleaned CSV
df.to_csv('data/styles_cleaned.csv', index=False, quoting=csv.QUOTE_ALL)
print("Cleaned CSV saved as data/styles_cleaned.csv")