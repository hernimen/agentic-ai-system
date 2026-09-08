# LIMS SOP Testing Guide

This guide gives a quick set of realistic questions to try against the PDFs already included in this project.

Use these checks to verify that the agent is answering from the loaded SOPs and manuals instead of guessing from general lab knowledge.

## Before you start

- Start the app with the current index:
  - `make ingest`
  - `make dev`
- Use a fresh thread or conversation ID for each test case so old answers do not leak into new ones.
- Prefer short natural-language questions that match how a lab user would ask them.

## 1) Autoclave / equipment manual questions

Source document:
- `data/equipment_manuals/sop_autoclave_operation_2023.pdf`

Suggested questions:

- What are the major hazards of using an autoclave?
- What are the common risks when autoclaving glassware or heated liquids?
- What hazards are associated with steam, vapors, or volatile chemicals in an autoclave?
- What burn risks should be considered when operating an autoclave?
- What are the key safety concerns for autoclave operation?
- What PPE or precautions are required before using the autoclave?
- What should the operator check before starting an autoclave cycle?
- What are the typical operating steps for an autoclave?

Good validation target:
- The answer should mention hazards such as broken glassware, super-heated liquids, steam burns, and heat burns.

## 2) General laboratory SOP questions

Source document:
- `data/sops/sop_general_lab_2016-01.pdf`

Suggested questions:

- What are the basic laboratory safety requirements in the general SOP?
- What general practices are required for safe laboratory operation?
- What are the laboratory rules for handling equipment and materials safely?
- What general documentation or procedural controls are described in the SOP?
- What operational expectations are covered by the general lab SOP?

Good validation target:
- The answer should stay anchored to the general lab SOP and not drift into unrelated procedures.

## 3) Hazardous spill response questions

Source document:
- `data/sops/sop_hazmat_spill_response.pdf`

Suggested questions:

- What should be done during a hazardous material spill response?
- What are the first steps for a chemical spill in the laboratory?
- What precautions should be taken when responding to a spill?
- Who should be notified in a hazardous spill event?
- What is the correct response flow for spill containment and cleanup?
- What is the difference between a general spill event and a hazardous chemical spill response?

Good validation target:
- The answer should be consistent with emergency-response and containment language from the spill SOP.

## 4) Biosafety / microbiological safety questions

Source document:
- `data/sops/biosafety_microbiological_biomedical_laboratories_6th_edition.pdf`

Suggested questions:

- What biosafety practices are required in a microbiological laboratory?
- What are the key biosafety principles described in the document?
- What are the relevant precautions for work with biological materials?
- What are the general requirements for safe microbiological practices?
- What biosafety controls are described for handling biomedical or microbiological materials?
- What are the operational safety expectations in a BMBL-style environment?

Good validation target:
- The answer should reflect general biosafety principles and not become a generic infection-control answer.

## 5) Cross-check questions for retrieval quality

These are useful for testing that the model does not answer from the wrong document.

- What does the autoclave SOP say about hazards?
- What does the biosafety SOP say about laboratory safety?
- What does the spill response SOP say about emergency handling?
- What is the general lab SOP usually about compared with the spill response SOP?
- What is the correct procedure for autoclave use, versus spill response, versus biosafety controls?

Good validation target:
- The model should stay within the relevant SOP and cite the right document context.

## 6) Quick regression checks

Try these after ingestion or after changing retrieval logic:

- autoclave hazards
- autoclave safety procedure
- biosafety laboratory precautions
- hazardous spill response steps
- general lab operational safety
- SOP-02 storage temperature
- SOP-07 incubation time

These questions are especially useful because they test both retrieval and restrictions.

## 7) Expected behavior

A good answer should:

- answer from source documents
- cite the correct document or section when available
- avoid unsupported assumptions
- keep the answer grounded to the relevant SOP or manual
- abstain or say the evidence is missing when the document does not support the claim

## 8) Useful tip

If a question seems to return a weak or unrelated answer, make sure:

- the API was restarted after ingestion
- the index was rebuilt with `make ingest`
- you are using a fresh thread ID
- the query is specific enough to match the correct SOP topic
