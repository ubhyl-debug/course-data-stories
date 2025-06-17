# Another Story

## In this story, we review the OCR text quality of news articles from the Nazi era and derive a metric that scores the perspective of the article towards the Nazi regime.

 **Abstract:** The Data Story utilizes news articles from the period 1920–1945, during which the Nazi party (NSDAP) rose to power. In the first chapters, the Data Story examines data from the [Deutsche Zeitungsportal](https://www.deutsche-digitale-bibliothek.de/newspaper) and evaluates its quality, especially the quality of the Optical Character Recognition (OCR) text. It identifies major issues in OCR text quality, resulting mainly from the limited capability of OCR to read old German fonts (Fraktur). Different methods of resolving this issue are discussed. In the following chapters, the text derives a metric to score the news articles based on their perspective toward the Nazi regime - scoring them in terms of approval or disapproval. The Data Story then clusters different news articles and major newspapers, visualizing their aggregated scores as derived from the published articles. In the final chapters, outliers are analyzed by linking them to historical events derived from the data source ???(tbd).


## Introduction


## Research questions
The Research question are seperated in two diffrent blocks:

**Noise**

1. How can noise/errors in the OCR text be defined and measured?
2. How can noise/errors in the OCR text be handled and mitigated?

**Scoring**

1. What are the relevant articles to determine what stance a newspaper had toward the NS regime?
2. What are the criteria to define what stance an article had toward the NS regime?
3. How can the stance toward the NS regime be quantified?

Ase methodolgy artefial intalagent (AI) based aproches such as Large Languge Models (LLMs) are used to quantify OCR text quality and drive scores to quantify articals regarding ther perspectiv. These scoring systems included Models such as distilgpt2, dbmdz/german-gpt2, google/byt5-small and bert-base-german-cased.

[//]: <  ![alt text](partitura-federation-1.png)>

![alt text](partitura-federation-1.png)

<div style="background-color: #fff3cd; padding: 10px; border: 1px solid #ffeeba; border-radius: 4px;">
<strong>Hinweis:</strong> Das Bild ist ein Platzhalter für unsere Darstellung, wie die Daten fließen.
</div>


## OCR and its potential errors

OCR (Optical Character Recognition) is a process that converts handwritten or printed text from images into machine-readable text. It is widely used and enables, for example, historians to scan pages of old historical books to extract their text.

![alt text](OCR_exsample.svg)

[link](https://www.deutsche-digitale-bibliothek.de/newspaper/item/PC26FTJGOJSB5WRMM4MZTEFKGOCRY4VW?query=Der%20Motor&issuepage=1)

The example above demonstrates that printed fonts based on the modern Latin alphabet tend to yield relatively accurate OCR results. In contrast, when processing texts in German Fraktur, recognition errors occur more frequently due to the script's divergence from the Latin alphabet.

![alt text](OCR_Text_Fraktur.svg)

[link](https://www.deutsche-digitale-bibliothek.de/newspaper/item/OBFCRDFM4NLVYKD6MDK2IQCIA7SHSBQ6?issuepage=1)

As shown in the image above, the OCR system fails to accurately recognize certain letters. For example, it transforms the original title "Diplomatenempfänge beim Führer" into "Tiptomniedempfunge deim Zahrer." The table below summarizes the character recognition errors made by the OCR program, indicating the incorrect characters and their correct counterparts.
 
| OCR Character | Correct Character | Example Error                  |
|---------------|-------------------|--------------------------------|
| T             | D                 | Tiptomniedempfunge → Diplomatenempfänge |
| i             | l                 | Tiptomniedempfunge → Diplomatenempfänge |
| p             | l                 | Tiptomniedempfunge → Diplomatenempfänge |
| t             | m                 | Tiptomniedempfunge → Diplomatenempfänge |
| n             | a                 | nied → aten                    |
| e             | a                 | empfunge → empfänge            |
| u             | ä                 | empfunge → empfänge            |
| g             | r                 | empfunge → empfänge            |
| d             | b                 | deim → beim                    |
| Z             | F                 | Zahrer → Führer                |
| a             | ü                 | Zahrer → Führer                |

The recognition errors make it difficult to understand the titles obtained from OCR output without access to the original text or its context. This poses a particular challenge for the application of natural language processing (NLP) methods, as incoherent words can significantly impair textual analysis-especially when dealing with linguistically complex or historically nuanced documents.

In the following two subchapters, the data storie evaluates methods for estimating OCR text quality in materials obtained from the Deutsches Zeitungsportal during the period 1920–1945. Begin by examining potential metrics for measuring OCR errors and identifying the challenges involved in their application.

### Potential metrics measuring errors

When analyzing OCR output, most evaluation methods rely on metrics that compute precision, such as:

$$
\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
$$

These metrics depend on a crucial prerequisite: the availability of a ground truth that defines the correct version of the text. Without such a reference, it is impossible to determine which tokens or characters are true positives or false positives.

Due to the extensive volume of articles, a corresponding ground truth is not available, rendering manual annotation infeasible. Consequently, this Datastorie employs perplexity as a heuristic metric to evaluate the relative quality of textual data across different historical periods.

Perplexity measures the confidence of a NLP in predicting subsequent tokens within a text. Lower perplexity values can correlate with higher text quality. To illustrate, consider the following example from 1936: _"Diplomatenempfänge beim..."_ ("Diplomatic receptions at the..."), compared to a distorted version: _"Tiptomniedempfunge deim..."_. 

As most readers would find it easier to predict the next word in the clearer phrase, a language model similarly exhibits higher confidence, reflected by lower perplexity, when processing less degraded, more coherent text. Thus, perplexity serves as a practical, though indirect, proxy for text quality in the absence of annotated ground truth.

It is important to note that perplexity remains a heuristic measure, as next-token prediction confidence also depends on how effectively a language model interprets textual context. Linguistically complex texts-such as newspaper from historical periods containing unconventional language use may yield elevated perplexity scores, despite being of high OCR text quality.

Nevertheless, perplexity provides a useful and consistent proxy for evaluating relative text quality when annotated ground truth is unavailable.

### OCR text quality

The perplexity in this data story was derived using the following language models from Hugging Face: `distilgpt2`, `dbmdz/german-gpt2`, `google/byt5-small`, and `bert-base-german-cased`.

```
def get_perplexity(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device) 
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return torch.exp(loss).item()
```

For each sampled text snippet (up to 512 tokens), the model was evaluated in inference mode, comparing the input sequence against itself as the prediction target. Internally, the model computes the cross-entropy loss over the token sequence, but instead of returning individual token-level losses, it uses the average loss over all tokens in the snippet. This average loss is then exponentiated to yield a single perplexity score per text snippet:

$$
Perplexity(x)=exp(L(x))
$$

where $L(x)$ is the mean negative log-likelihood across the tokens in the snippet $x$.

To use computing resources efficiently, perplexity was computed over the period January to May 1933, and from that set, $1.5\%$ was randomly selected for evaluation.

As mentioned in section [OCR and its potential errors](#ocr-and-its-potential-errors), OCR text from post-war newspapers is assumed to have higher accuracy, as most articles are written in the classical Latin alphabet. Therefore, articles from the period 1980–1994 were used, with $20\%$ of that set selected to compute perplexity.

Note that, due to copyright constraints, more articles from 1933 are available than from 1980–1994. This explains the difference in sampling rates between the two periods.

<div style="background-color: #fff3cd; padding: 10px; border: 1px solid #ffeeba; border-radius: 4px;">
<strong>Hinweis:</strong> Zum aktuellen Zeitpunkt (06/2025) wurde die Perplexity so ausgerechnet. Eine größere Stichprobe könnte bei Bedarf erneut durchgeführt werden
</div>

| Model             | 1933 Period (01/1933–05/1933) | 1980–1994 Period (01/1980–05/1994) | Ratio |
| ----------------- | ----------------------------- | ---------------------------------- | ----- |
| distilgpt2        | 195.37                        | 132.82                             | 1.47  |
| dbmdz/german-gpt2 | 236.74                        | 91.34                              | 2.58  |
| google/byt5-small | 1,971,675.58                  | 552,495.41                         | 3.57  |

The substantial increase in perplexity for the 1933 dataset compared to the 1980–1994 dataset suggests that data quality, such as OCR errors, may significantly affect language model performance. However, as already discussed in section [OCR and its potential errors](#ocr-and-its-potential-errors), this may also result from the language model, which may understand fewer historical contexts.

## Quantifying the perspective of a news article distorts the Nazi regime

### Model...


## Summary

## With a SPARQL query

```sparql linenums="1" title="Example query"
# List of research data portals
PREFIX fabio: <http://purl.org/spar/fabio/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX nfdicore: <https://nfdi.fiz-karlsruhe.de/ontology/>
PREFIX n4c: <https://nfdi4culture.de/id/>

SELECT (SAMPLE(?resource) AS ?entity) (SAMPLE(?label) AS ?name)
WHERE {
    ?resource rdf:type nfdicore:DataPortal,
      				fabio:Database .
    ?resource rdfs:label ?label .
}
GROUP BY ?resource
ORDER BY ?name
```
