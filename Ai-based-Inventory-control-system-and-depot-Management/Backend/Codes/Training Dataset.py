import pandas as pd
import numpy as np
import random

# -----------------------------
# Parameters
# -----------------------------
num_rows_per_class = 1000  # Rows per stock level class → total 5000 rows
stock_levels = ['Very Low', 'Low', 'Perfect', 'High', 'Very High']

# -----------------------------
# Feature ranges per stock level (example, adjust as needed)
# -----------------------------
feature_ranges = {
    'current_stock': {
        'Very Low': (0, 10),
        'Low': (11, 25),
        'Perfect': (26, 50),
        'High': (51, 75),
        'Very High': (76, 100)
    },
    'running_stock': {
        'Very Low': (0, 10),
        'Low': (11, 25),
        'Perfect': (26, 50),
        'High': (51, 75),
        'Very High': (76, 100)
    },
    'quantity': {
        'Very Low': (0, 5),
        'Low': (6, 15),
        'Perfect': (16, 30),
        'High': (31, 50),
        'Very High': (51, 80)
    },
    'unit_price': (10, 1000),  # uniform across classes
    'transaction_cost': (5, 500),  # uniform
    'section_id': (1, 20)  # example
}

# Example categorical values
brands = ['Brand A', 'Brand B', 'Brand C', 'Brand D']
suppliers = ['Supplier X', 'Supplier Y', 'Supplier Z']
locations = ['Depot 1', 'Depot 2', 'Depot 3']
managers = ['Manager 1', 'Manager 2', 'Manager 3']
sections = ['Section A', 'Section B', 'Section C']
transaction_types = ['Purchase', 'Sale', 'Return']
unit_measures = ['Pieces', 'Boxes', 'Kg']
performed_by = ['User1', 'User2', 'User3']

# -----------------------------
# Generate dataset
# -----------------------------
data_rows = []

for level in stock_levels:
    for _ in range(num_rows_per_class):
        row = {
            'current_stock': random.randint(*feature_ranges['current_stock'][level]),
            'running_stock': random.randint(*feature_ranges['running_stock'][level]),
            'quantity': random.randint(*feature_ranges['quantity'][level]),
            'unit_price': round(random.uniform(*feature_ranges['unit_price']), 2),
            'transaction_cost': round(random.uniform(*feature_ranges['transaction_cost']), 2),
            'section_id': random.randint(*feature_ranges['section_id']),
            'brand': random.choice(brands),
            'supplier': random.choice(suppliers),
            'location': random.choice(locations),
            'manager': random.choice(managers),
            'section_name': random.choice(sections),
            'transaction_type': random.choice(transaction_types),
            'unit_of_measure': random.choice(unit_measures),
            'performed_by': random.choice(performed_by),
            'transaction_date': pd.to_datetime('2024-01-01') + pd.to_timedelta(random.randint(0, 365), unit='days'),
            'stock_level': level
        }
        data_rows.append(row)

# Convert to DataFrame
training_df = pd.DataFrame(data_rows)

# Shuffle dataset
training_df = training_df.sample(frac=1).reset_index(drop=True)

# Save to CSV
training_df.to_csv('inventory_training_5000.csv', index=False)
print("✅ Balanced training dataset with 5000 rows generated as 'inventory_training_5000.csv'")
print(training_df.head(10))  # Preview first 10 rows
