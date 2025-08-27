import pandas as pd
import os
import re

# Zeitraum definieren
start_year = 1920
end_year = 1945

# Verzeichnis mit den Originaldateien
input_dir = "data_deutsches_zeitungsportal_misc" 
output_dir = os.path.join(input_dir, "nur_gueltige_artikel")
os.makedirs(output_dir, exist_ok=True)

# Gültige page_ids laden
ids_df = pd.read_pickle("zeitungen_mit_5_jahren.pkl")

# Extrahiere gültige page_ids
if isinstance(ids_df, pd.DataFrame) and 'page_id' in ids_df.columns:
    valid_ids = set(ids_df['page_id'])
else:
    valid_ids = set(ids_df)

print(f"{len(valid_ids)} gültige page_ids geladen.")

# Gewünschte Spalten
columns_to_keep = [
    'page_id',
    'pagenumber',
    'paper_title',
    'publication_date',
    'place_of_distribution',
    'language',
    'plainpagefulltext'
]

# Liste der exakt angegebenen Zeitungstitel
top_titles = [
    "Kölnische Zeitung. 1803-1945",
    "Deutscher Reichsanzeiger und Preußischer Staatsanzeiger",
    "Vorwärts",
    "Hamburger Tageblatt : Zeitung der Nationalsozialistischen Deutschen Arbeiterpartei",
    "Münsterischer Anzeiger : Westfälischer Merkur : Münsterische Volkszeitung : amtliches Organ des Gaues Westfalen-Nord der NSDAP und sämtlicher Behörden",
    "Süddeutsche Zeitung : für deutsche Politik und Volkswirtschaft",
    "Stuttgarter neues Tagblatt : südwestdeutsche Handels- und Wirtschafts-Zeitung",
    "Dresdner Nachrichten",
    "Dresdner neueste Nachrichten",
    "Schwäbischer Merkur : mit Schwäbischer Kronik und Handelszeitung : Süddeutsche Zeitung",
]



# Alle .pkl-Dateien im angegebenen Zeitraum durchgehen
for filename in os.listdir(input_dir):
    if not filename.endswith(".pkl"):
        continue

    # Jahr aus dem Dateinamen extrahieren
    match = re.match(r"newspapers_ger_(\d{4})_\d{2}\.pkl", filename)
    if not match:
        continue

    year = int(match.group(1))
    if year < start_year or year > end_year:
        continue

    output_file = filename.replace(".pkl", "_filtered.pkl")
    output_path = os.path.join(output_dir, output_file)

    if os.path.exists(output_path):
        print(f"Überspringe {output_file} (bereits vorhanden).")
        continue

    filepath = os.path.join(input_dir, filename)
    try:
        df = pd.read_pickle(filepath)

        df = df[df['paper_title'].isin(top_titles)]
        df_filtered = df[df['page_id'].isin(valid_ids)]

        # Nur gewünschte Spalten auswählen (falls vorhanden)
        existing_columns = [col for col in columns_to_keep if col in df_filtered.columns]
        df_filtered = df_filtered[existing_columns]
        
        #df_sampled = df_filtered
        
        # Ziehe zufällig bis zu 200 Artikel je Zeitung (paper_title)
        
        df_sampled = df_filtered.groupby('paper_title', group_keys=False, sort=False).apply(
            lambda g: g.sample(n=min(len(g), 50), random_state=42),
            include_groups=False
        ).reset_index(drop=True)
        




    except Exception as e:
        print(f"Fehler beim Verarbeiten von {filename}: {e}")
        continue

    # Speichern
    df_sampled.to_pickle(output_path)
    print(f"{filename}: {len(df_sampled)} Artikel gespeichert ({len(existing_columns)} Spalten).")
