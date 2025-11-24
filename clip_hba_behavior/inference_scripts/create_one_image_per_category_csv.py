"""
Create a CSV file with one image per category from stimuli_metadata.
"""

import pandas as pd
from pathlib import Path
import sys


def create_one_image_per_category_csv(input_csv_path, output_csv_path=None):
    """
    Create a CSV file with one image per category from the input stimuli_metadata CSV.
    
    Args:
        input_csv_path (str or Path): Path to the input stimuli_metadata CSV file
        output_csv_path (str or Path, optional): Path to save the output CSV. 
                                                 If None, appends '_one_per_category' to input filename.
    """
    input_csv_path = Path(input_csv_path)
    
    if not input_csv_path.exists():
        raise FileNotFoundError(f"Input CSV file not found: {input_csv_path}")
    
    # Read the CSV file
    # First read without index_col to see the structure
    df_full = pd.read_csv(input_csv_path)
    
    print(f"Original CSV shape: {df_full.shape}")
    print(f"Number of unique concepts: {df_full['concept'].nunique()}")
    print(f"Columns: {list(df_full.columns)}")
    
    # Check if first column should be used as index (like 'trial_type')
    # If the first column is not 'concept' or 'stimulus', it might be an index column
    first_col = df_full.columns[0]
    use_index = first_col not in ['concept', 'stimulus', 'session', 'run', 'subject_id', 'trial_id']
    
    if use_index:
        # Re-read with index_col=0 to match pipeline behavior
        df = pd.read_csv(input_csv_path, index_col=0)
        # Group by concept and take the first row from each group
        df_filtered = df.groupby('concept').first().reset_index()
        # The index from groupby becomes the new index, which is fine
    else:
        # No special index column, just group normally
        df = df_full
        df_filtered = df.groupby('concept').first().reset_index()
    
    print(f"\nFiltered CSV shape: {df_filtered.shape}")
    print(f"Number of concepts in filtered CSV: {df_filtered['concept'].nunique()}")
    
    # Determine output path
    if output_csv_path is None:
        output_csv_path = input_csv_path.parent / f"{input_csv_path.stem}_one_per_category{input_csv_path.suffix}"
    else:
        output_csv_path = Path(output_csv_path)
    
    # Save the filtered CSV
    # Save without index to match the original format, or with index if needed
    df_filtered.to_csv(output_csv_path, index=True)
    
    print(f"\nFiltered CSV saved to: {output_csv_path}")
    print(f"First few rows:")
    print(df_filtered.head())
    
    return output_csv_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python create_one_image_per_category_csv.py <input_csv_path> [output_csv_path]")
        print("\nExample:")
        print("  python create_one_image_per_category_csv.py sub-01_StimulusMetadata_train_only.csv")
        print("  python create_one_image_per_category_csv.py input.csv output.csv")
        sys.exit(1)
    
    input_csv_path = sys.argv[1]
    output_csv_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        output_path = create_one_image_per_category_csv(input_csv_path, output_csv_path)
        print(f"\n✓ Successfully created filtered CSV: {output_path}")
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

