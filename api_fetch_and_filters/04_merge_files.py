import pandas as pd
import os

# Ordner mit den .pkl-Dateien
input_dir = "data_deutsches_zeitungsportal_misc/nur_gueltige_artikel" 

# Liste zum Sammeln der DataFrames
dataframes = []

# Alle .pkl-Dateien laden
for filename in os.listdir(input_dir):
    if filename.endswith(".pkl"):
        filepath = os.path.join(input_dir, filename)
        try:
            print(f"Lade: {filename}")
            df = pd.read_pickle(filepath)
            dataframes.append(df)
        except Exception as e:
            print(f"Fehler beim Laden von {filename}: {e}")

# Zusammenführen
merged_df = pd.concat(dataframes, ignore_index=True)
print(f"Zusammengeführt: {len(merged_df)} Zeilen")


output_path = os.path.join(input_dir, "merged_all.pkl")
merged_df.to_pickle(output_path)
print(f"Gespeichert unter: {output_path}")
