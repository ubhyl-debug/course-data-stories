# Another Story

Here is some lorem ipsum text to fill the page.
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

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
