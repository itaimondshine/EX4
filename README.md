# EX4

## Install
* all the project's dependencies are listed in the file requirements.txt
* there are 2 steps for installing the project's dependencies:
  * run `pip3 install -r requirements.txt` from the project's root
  * run `python3 -m spacy download en_core_web_sm` to install the specific spacy model that we are using 

## Evaluation script
* As requested, this script receives 2 input arguments and prints the evaluation metrics 
per relation type. Notice that our system can predict only one relation type, so if this
script will be used on the predictions of our system it'll always show a single line for 
the target relation type that was set in the OUTPUT_TAG variable in config.py 
* to run, use `python3 eval.py <gold_annotations_file> <predicted_annotations_file>`

## Extraction script
