# MARDY - PaG

This repository contains Claim Identifier and Claim Classifier models trained on German newspaper articles. For more information, please refer to [this](https://www.aclweb.org/anthology/P19-1273.pdf) paper. 

## Dependencies
   - Python 3.x
   - PyTorch 1.1.0
   - HuggingFace's transformers 1.0.0


## Downloading pre-trained models
   - If you would like to use our pre-trained claim identifier/claim classifier models please download them from [here](http://bit.ly/37bBjbo)
   and put them into ```pag_mardy/claim_identifier/saved_models``` and ```pag_mardy/claim_classifier/saved_models```
   

## Using claim Identifier
   - Claim Identifier expects a conllu formatted file as input. Individual sentences should be splitted with a blank line
   and each non-empty line should consists of three tab separated fields: `WordIndex\tWord\tLabel`
    Example input file (`input.conllu`):
   ```
   
     	     1       Darunter        O
	     2       sollen  O
	     3       sich    O
	     4       auch    O
	     5       Kinder  O
	     6       befunden        O
	     7       haben   O
	     8       sowie   O
	     9       Gebrechliche    O
	     10      ,       O
	     11      die     O
	     12      auf     O
	     13      Gehhilfen       O
	     14      angewiesen      O
	     15      sind    O
	     16      .       O

	     1       Andere  O
	     2       seien   O
	     3       mit     O
	     4       Handschellen    O
	     5       gefesselt       O
	     6       gewesen O
	     7       .       O

	     1       Der     B-Claim
	     2       ehemalige       I-Claim
	     3       Bundesverfassungsrichter        I-Claim
	     4       Ernst   I-Claim
	     5       Gottfried       I-Claim
	     6       Mahrenholz      I-Claim
	     7       (       I-Claim
	     8       SPD     I-Claim
	     9       )       I-Claim
	     10      hat     I-Claim
	     11      diese   I-Claim
	     12      aktuellen       I-Claim
	     13      Abschiebungen   I-Claim
	     14      scharf  I-Claim
	     15      kritisiert      I-Claim
	     16      .       I-Claim
```

   
   - To run claim identifier one can use the following command:
      ```
      cd pag_mardy
      export CUDA_VISIBLE_DEVICES=0
      export model=./saved_models/claim_identifier_model.bin
      export out_fname="claim_identifier_predictions.out"
      export input_fname="./data/input.conllu"
      python run_tagger.py --file_dir ${input_fname} --load ${model} --tagger_predictions_conllu_fname ${out_fname}```
      
   - Model outputs predictions as in conllu format as well. 

    

## Using Claim Classifier
   - Unlike Claim Identifier, Claim Classifier works with tsv formatted files.
   - Example input file (`input.tsv`): There are seven fields only two of them (claim, major_classes) are mandatory to use pre-trained claim classifier model:
		- claim: A sequence of words to be claim. `e.g. Damit sollen straffällige Ausländer , aber auch Menschen ohne Aufenthaltsberechtigung einfacher abgeschoben und mit Wiedereinreisesperren belegt werden können .`
		- major_classes: Multi-hot vector of length nine. It represents major claims appeared in the dataset (controlling migration; residency; integration; domestic security; foreign policy; economy; society; procedures; other) `e.g. 0 1 0 1 0 0 0 1 0`
		- minor_classes: Sequence of integers separated by a single space. (Note that: Model does not use this field, you can set it to `NONE`)
		- paragraph: Paragraph where the claim appears. (Note that: Model does not use this field, you can set it to `NONE`)
		- json_id: Id of the corresponding article. (Note that: Model does not use this field, you can set it to `NONE`)
		- actor_values: Name of the actor(s) associated with the corresponding claim. (Note that: Model does not use this field, you can set it to `NONE`)
		- claim_stance: Stance of the claim. (Note that: Model does not use this field, you can set it to `NONE`)
	   
- A valid example:
   ```
   Damit sollen straffällige Ausländer , aber auch Menschen ohne Aufenthaltsberechtigung einfacher abgeschoben und mit Wiedereinreisesperren belegt werden können .        0 1 0 1 0 0 0 1 0       207 404 812     NONE    131203.json     NONE    NONE
   ```
   
   - To run claim classifier please run the following command:
   
   ```
   	cd pag_mardy
	export CUDA_VISIBLE_DEVICES=0
	export model=./saved_models/claim_classifier_model.bin
	export out_fname="claim_classifier_predictions.out"
	export input_fname="./data/input.tsv"
	python run_classifier.py --file_dir ${input_fname} --model_file ${model} --output_fname ${out_fname}      
	```


## Data:
   - First version of the dataset is available at [here](https://github.com/mardy-spp/mardy_acl2019).
   
  
## Web Interface:
- Claim Identifier is also accessable through the web interface at [here](http://193.196.52.50/). Note that, server-side code is working on an inefficient (cpu-only), free public node. It may take some time to response for large documents. 

