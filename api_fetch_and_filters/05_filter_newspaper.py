import pandas as pd

# Lade den zusammengeführten DataFrame
df = pd.read_pickle("data_deutsches_zeitungsportal_misc/gefiltert/merged_all.pkl")

# Stelle sicher, dass das Datum im richtigen Format ist
df['year'] = pd.to_datetime(df['publication_date']).dt.year

# Zähle Artikel pro Zeitung und Jahr
yearly_counts = (
    df.groupby(['paper_title', 'year'])
    .size()
    .reset_index(name='article_count')
)

# Behalte nur Jahre mit >= 50 Artikeln
active_years = yearly_counts[yearly_counts['article_count'] >= 500]

# Funktion, um zu prüfen, ob eine Zeitung ≥5 aufeinanderfolgende aktive Jahre hat
def has_5_consecutive_years(group):
    years = sorted(group['year'].unique())
    for i in range(len(years) - 11):
        if all(years[i + j] == years[i] + j for j in range(12)):
            return True
    return False

# Wende Filterfunktion an
valid_papers = (
    active_years.groupby('paper_title')
    .filter(has_5_consecutive_years)
)

# Liste qualifizierter Zeitungen (einmalig, ohne Duplikate)
valid_titles = valid_papers['paper_title'].unique()
filtered_df = df[df['paper_title'].isin(valid_titles)]

# OPTIONAL: Speichern der Liste als DataFrame



#valid_titles_df = pd.DataFrame(valid_titles, columns=['paper_title'])

valid_titles_df = filtered_df
valid_titles_df.to_pickle("zeitungen_mit_5_jahren.pkl")
print(f"{len(valid_titles)} Zeitungen gefunden und gespeichert.")
