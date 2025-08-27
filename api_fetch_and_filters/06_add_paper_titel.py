import pandas as pd
import os

df = pd.read_pickle("data_deutsches_zeitungsportal_misc/nur_gueltige_artikel/merged_all.pkl")
input_dir = "data_deutsches_zeitungsportal_misc/original_data"

for filename in os.listdir(input_dir):
    if not filename.endswith(".pkl"):
        continue

    full_path = os.path.join(input_dir, filename)
    print(f"Verarbeite Datei: {filename}")
    
    try:
        df_main_data = pd.read_pickle(full_path)

        # Sicherstellen, dass benötigte Spalten vorhanden sind
        if 'page_id' in df_main_data.columns and 'paper_title' in df_main_data.columns:
            df_main_data = df_main_data[['page_id', 'paper_title']].drop_duplicates(subset='page_id')

            # Nur wenn paper_title schon existiert im df -> dann suffix verwenden
            if 'paper_title' in df.columns:
                df = df.merge(df_main_data, on='page_id', how='left', suffixes=('', '_new'))

                if 'paper_title_new' in df.columns:
                    df['paper_title'] = df['paper_title'].combine_first(df['paper_title_new'])
                    df.drop(columns=['paper_title_new'], inplace=True)
            else:
                # Erstes Mal: einfach mergen
                df = df.merge(df_main_data, on='page_id', how='left')
    except Exception as e:
        print(f"Fehler beim Verarbeiten von {filename}: {e}")

# Speichern
output_path = "merged_all.pkl"
df.to_pickle(output_path)
print("Fertig gespeichert in merged_all.pkl")
