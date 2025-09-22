# install requirements once
import pickle
from mistralai import Mistral
import time



api_key = ""

model = "mistral-small-2506"

client = Mistral(api_key=api_key)


def call_mistral(text):

    promt = f"""
    "You are given an OCR-extracted newspaper article below from the period 1920–1945. "
    "Your task is to evaluate accordingly to the metric of criticism towards NSDAP. "
    "Primarily base your evaluation on keywords that indicate their opinion towards "
    "the NSDAP but also include the semantics of the text. "
    "Only return the score without explanation"
    "Range: -2 being critical and opposing, 0 being neutral, 2 being supportive towards NSDAP."
    "article:"
    "{text}"
    """

    chat_response = client.chat.complete(
        model= model,
        messages = [
            {
                "role": "user",
                "content": promt,
            },
        ]
    )
    chat_response = client.chat.complete(
    model= model,
    messages=[{"role": "user", "content": promt}],
    temperature=0.7,
    top_p=0.9,
    max_tokens=124,
    )
    return chat_response
    #print(chat_response.choices[0].message.content)


# 1) Load your merged pickle
with open('merged_all_added_score.pkl', 'rb') as f:
    df = pickle.load(f)

if 'score' not in df.columns:
    df['score'] = ""  

#df = df.head(25)

start_time = time.time()
report_every = 30


for i, row in df.iterrows():

    if row['score'] != "":
        continue

    elapsed = time.time() - start_time

    try:

        df.at[i, 'score'] = call_mistral(row['plainpagefulltext']).choices[0].message.content

        with open('merged_all_added_score.pkl', 'wb') as f:
                pickle.dump(df, f)    
        time.sleep(0.2)
    except Exception as e:
        print(f"Error at row {i}: {e}")
        time.sleep(60)
        continue  # oder break, je nach gewünschtem Verhalten

    if (i + 1) % report_every == 0:
        elapsed = time.time() - start_time
        avg_time_per_row = elapsed / (i + 1)
        remaining_rows = len(df) - (i + 1)
        estimated_remaining = avg_time_per_row * remaining_rows
        
        print(f"Processed {i + 1} rows.")
        print(f"Average time per row: {avg_time_per_row:.4f} seconds.")
        print(f"Estimated time remaining: {estimated_remaining / 60:.2f} minutes.\n")


