import pandas as pd
import os

# Verzeichnis in dem die .pkl-Dateien liegen
input_dir = "data_deutsches_zeitungsportal_misc" 
output_dir = os.path.join(input_dir, "gefiltert")
os.makedirs(output_dir, exist_ok=True)

# Spalten die behalten werden sollen
columns_to_keep = ['page_id', 'pagenumber', 'paper_title', 'publication_date']

# Alle .pkl-Dateien durchgehen
for filename in os.listdir(input_dir):
    if filename.endswith(".pkl"):
        filepath = os.path.join(input_dir, filename)
        print(f"Lade: {filename}")
        
        try:
            df = pd.read_pickle(filepath)
            df_filtered = df[columns_to_keep]
        except Exception as e:
            print(f"Fehler beim Verarbeiten von {filename}: {e}")
            continue
        
        output_filename = filename.replace(".pkl", "_filtered.pkl")
        output_path = os.path.join(output_dir, output_filename)
        
        df_filtered.to_pickle(output_path)
        print(f"Gespeichert: {output_filename}")
