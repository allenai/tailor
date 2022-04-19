# Recreating the Tailor Generator

## Get the Data
1. <b>Download the Ontonotes Data</b>

    ```sh
    mkdir data/
    cd data
    pip install gdown
    gdown "https://drive.google.com/uc?id=1QjsTrNYE1PtkG4VRtG4H3lOw2auAddva"
    wget https://github.com/ontonotes/conll-formatted-ontonotes-5.0/archive/v12.tar.gz
    tar -xvzf v12.tar.gz
    tar -xvzf ontonotes-release-5.0_LDC2013T19.tgz
    cd ..
    ```

2. <b>Preprocess the data</b>

    The following command will create .gold_conll files for the Ontonotes 5.0 data:

    ```sh
    ./skeleton2conll.sh -D ./data/ontonotes-release-5.0/data/files/data ./data/conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0
    ```
     
    Then convert the data to <b>json</b> format:

    ```
    python data_process.py --step preprocess
    ```
    
    The above command will write json files to `/data/orig`. Example json:
    
    ```sh 
    {
      # domain id
      "domain": "bc_cctv", 
      # token list
      "words": ["Dear", "viewers", ",", "the", "China", "News", "program", "will", "end", "here", "."], 
      # list of all verb lemmas in sentence
      "lemmas": ["end"], 
      # list of Propbank frameset ids for all verbs in sentence
      "frameset_ids": ["01"], 
      # list of dicts with information for each lemma using the Propbank formalism
      "props": [{
          "lemma": "end", 
          # the index of the verb in question
          "vidx": 0, 
          # Propbank frameset id for the verb
          "frameset_id": "01"
          # a list of the BIO formats of the tag for each word, based on the current vidx,
          "tags": ["B-ARGM-DSP", "I-ARGM-DSP", "O", "B-ARG1", "I-ARG1", "I-ARG1", "I-ARG1", "B-ARGM-MOD", "B-V", "B-ARGM-LOC", "O"], 
          # natural language sentence augmented with tags
          "description": "[ARGM-DSP: Dear viewers] , [ARG1: the China News program] [ARGM-MOD: will] [V: end] [ARGM-LOC: here] .", 
       }]
    }
    ```

    #### More information about the Propbank formalism:
    The Propbank is [here](http://verbs.colorado.edu/propbank/framesets-english-aliases/).
    (described in [this paper](https://arxiv.org/pdf/2101.05779.pdf))
    
    A. Numbered arguments (A0-A5, AA): Arguments defining verb-specific roles. Their semantics depends on the verb and the verb usage in a sentence, or verb sense. The most frequent roles are A0 and A1. Commonly, 
    - `A0`: agent, 
    - `A1`: corresponds to the patient or theme of the proposition
    
    B. Adjuncts (AM-): General arguments that any verb may take optionally, 13 types in total: 
        
    - `AM-ADV`: general-purpose; 
    - `AM-CAU`: cause; 
    - `AM-DIR`: direction; 
    - `AM-DIS`: discourse marker; 
    - `AM-EXT`: extent; 
    - `AM-LOC`: location; 
    - `AM-MNR`: manner; 
    - `AM-MOD`: modal verb; 
    - `AM-NEG`: negation marker; 
    - `AM-PNC`: purpose; 
    - `AM-PRD`: predication; 
    - `AM-REC`: reciprocal; 
    - `AM-TMP`: temporal.
    
    C. References (R-): Arguments representing arguments realized in other parts of the sentence. The role of a reference is the same as the role of the referenced argument. The label is an R-tag prefixed to the label of the referent, e.g., R-A1.

## Format the Data for Tailor

   Create inputs for the Tailor generator. For details about input formats, see Section 2.2 of [our paper](https://arxiv.org/pdf/2107.07150.pdf).
   
   Run the following command to produce inputs for <b>unlikehood training</b>. In the output files, the rows with "reward=-1" correspond to negative prompts).
   
   ```sh
   python data_process.py --step input --prompt_identifier unlikelihood --use_unlikelihood
   ```      
  
   Or for standard MLE training:
   
   ```
    python data_process.py --step input --prompt_identifier mle
   ```
    
## Train the model

   ```
   bash finetune_ul.sh
   ```

