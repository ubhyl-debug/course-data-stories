# Telling Data Stories with Semantic Technologies and Generative AI

FIZ Karlsruhe – Leibniz Institute for Information Infrastructure
AIFB – Karlsruhe Institute of Technology
Academy of Sciences and Literature, Mainz

## Instructions

This GitHub repository contains the data story "Reading between the Lines", which was created during the summer semester 2025 in the course "Telling Data Stories with Semantic Technologies and Generative AI". 

📂 stories              → Final data story and resources  
📂 api_fetch_and_filters → Scripts for data sourcing & filtering (chronologically ordered)  
📂 Mistral_code          → Code for calling the Mistral model, visualization & enrichment  
📂 OCR_error_code        → Handling OCR errors, as described in the story  
📂 _Archiv               → Additional resources used during exploration (not required for reproducing results)  


1. **Start the Shmarql container with docker compose**: Navigate to the cloned repository and and run the following command:
   ```bash
   cd course-data-stories && docker compose up -d
   ```

There should now be a running instance of the NFDI4Culture Datastories running on your machine, it can be viewd at this URI:

[http://localhost:7015/](http://localhost:7015/)

Try to create a new file named 'index.md' in the 'stories' folder. You can fill it with any markdown text, and then refresh the
following page in your browser: http://localhost:7015/course/

This should show the text that was just entered in the 'index.md' file you created.


## More details

The source files for the production NFDI4CUlture Datastories repository can be found at the following URI:
https://gitlab.rlp.net/adwmainz/nfdi4culture/knowledge-graph/shmarql/datastories

You can view the source for more details on how to create your own data stories here.