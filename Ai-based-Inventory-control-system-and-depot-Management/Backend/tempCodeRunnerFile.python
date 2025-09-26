import pandas as pd
import numpy as np
import random

# -----------------------------
# Parameters
# -----------------------------
num_rows = 50  # Number of new rows for prediction
# Example feature ranges
feature_ranges = {
    'current_stock': (0, 100),
    'running_stock': (0, 100),
    'quantity': (0, 80),
    'unit_price': (10, 1000),
    'transaction_cost': (5, 500),
    'section_id': (1, 20)
}

# Categorical columns
brands = ['Brand A', 'Brand B', 'Brand C', 'Brand D']
suppliers = ['Supplier X', 'Supplier Y', 'Supplier Z']
locations = ['Depot 1', 'Depot 2', 'Depot 3']
managers = ['Manager 1', 'Manager 2', 'Manager 3']
sections = ['Section A', 'Section B', 'Section C']
transaction_types = ['Purchase', 'Sale', 'Return']
unit_measures = ['Pieces', 'Boxes', 'Kg']
performed_by = ['User1', 'User2', 'User3']

# -----------------------------
# Generate new inventory data
# -----------------------------
data_rows = []

for _ in range(num_rows):
    row = {
        'current_stock': random.randint(*feature_ranges['current_stock']),
        'running_stock': random.randint(*feature_ranges['running_stock']),
        'quantity': random.randint(*feature_ranges['quantity']),
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
        'transaction_date': pd.to_datetime('2024-01-01') + pd.to_timedelta(random.randint(0, 365), unit='days')
    }
    data_rows.append(row)

# Convert to DataFrame
new_inventory_df = pd.DataFrame(data_rows)

# Preview 5 random rows
print(new_inventory_df.sample(5))

# -----------------------------
# Save to CSV
# -----------------------------
new_inventory_df.to_csv('new_inventory_for_prediction.csv', index=False)
print("✅ New inventory dataset saved as 'new_inventory_for_prediction.csv'")
