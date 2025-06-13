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

Ase methodolgy we used artefial intalagent (AI) based aproches such as Large Languge Models (LLMs) to quantify OCR text quality and drive scores to quantify articals regarding ther perspectiv. These scoring systems included Models such as distilgpt2, dbmdz/german-gpt2, google/byt5-small and bert-base-german-cased.

[//]: <  ![alt text](partitura-federation-1.png)>

![alt text](partitura-federation-1.png)

Das Bild ist ein Platzhalter für unsere Darstellung, wie die Daten fließen.


## OCR text quality

OCR (Optical Character Recognition) is an image-to-text conversion process for handwritten or printed text. It is widely used and enables, for example, historians to scan pages of old historical books and obtain the text from these pages.

![alt text](<OCR exsample.svg>)


### OCR and its potential errors

### Potential metrics measuring errors


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
