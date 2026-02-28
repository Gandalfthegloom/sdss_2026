import pandas as pd

def generate_dataset_summary(file_paths, output_filename="datasets_summary.txt"):
    """Reads multiple Excel files and writes a structural summary to a text file."""
    
    with open(output_filename, 'w', encoding='utf-8') as file:
        file.write("=== DATASET INFERENCE SUMMARY ===\n\n")
        
        for name, path in file_paths.items():
            try:
                # Load the dataset
                df = pd.read_excel(path)
                
                # Write basic size information
                file.write(f"--- {name} ---\n")
                file.write(f"Size: {df.shape[0]} rows, {df.shape[1]} columns\n\n")
                file.write("Column Summaries:\n")
                
                # Iterate through each column to extract variable information
                for col in df.columns:
                    col_data = df[col]
                    file.write(f"- Column: '{col}' (Type: {col_data.dtype})\n")
                    
                    # 1. Numerical Variables (Calculate Mean, Median, Max)
                    if pd.api.types.is_numeric_dtype(col_data):
                        clean_data = col_data.dropna()
                        if not clean_data.empty:
                            mean_val = clean_data.mean()
                            median_val = clean_data.median()
                            max_val = clean_data.max()
                            file.write(f"  -> Numerical Summary: Mean = {mean_val:.2f}, Median = {median_val:.2f}, Max = {max_val:.2f}\n")
                        else:
                            file.write("  -> Numerical Summary: Column is completely empty (NaNs).\n")
                            
                    # 2. Categorical / Text Variables (Calculate Counts)
                    # UPDATED: Added is_string_dtype to catch 'str' columns
                    elif pd.api.types.is_object_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data) or pd.api.types.is_string_dtype(col_data):
                        value_counts = col_data.value_counts()
                        
                        file.write(f"  -> Categorical Summary (Top 10 Values):\n")
                        for val, count in value_counts.head(10).items():
                            file.write(f"      '{val}': {count} occurrences\n")
                            
                        if len(value_counts) > 10:
                            file.write(f"      ... and {len(value_counts) - 10} more unique categories.\n")
                    
                    # 3. Date/Time Variables
                    elif pd.api.types.is_datetime64_any_dtype(col_data):
                        file.write(f"  -> Datetime Summary: Ranges from {col_data.min()} to {col_data.max()}\n")
                        
                    else:
                        file.write("  -> (No specific summary generated for this data type)\n")
                        
                file.write("\n" + "="*50 + "\n\n")
                print(f"Successfully processed {name}")
                
            except Exception as e:
                error_msg = f"Error processing {name}: {e}\n\n"
                file.write(error_msg)
                print(error_msg)

    print(f"\nAll done! You can find the results in: {output_filename}")

# --- EXECUTION ---
# Map your dataset names to their actual file paths on your computer
my_files = {
    "Dataset 1": "Data/airline_ticket_dataset.xlsx",
    "Dataset 2": "Data/personal_finance_dataset.xlsx",
    "Dataset 3": "Data/public_services_dataset.xlsx"
}

# Run the function
generate_dataset_summary(my_files)