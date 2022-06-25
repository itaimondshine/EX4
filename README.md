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
* To run, use `python3 eval.py <gold_annotations_file> <predicted_annotations_file>`

## Extraction script
* Because our training process is relatively short (around 2-3 minutes), our extract.py script
runs the model training, and then creates predictions on the test dataset. This is why this
script requires the train set files as input as well as the target corpus file to predict on
* Please use the raw train corpus file (not the .processed version)
* To run, use `python3 extract.py <train_corpus_file> <train_annotations_file> <test_corpus_file_to_predict_on>`
  * please use this command from the root directory of the project, so the imports will work properly
  * this script creates 2 output files:
    * TRAIN.annotations.predicted.txt - contains the extracted relations of the train set
    * TEST.annotations.predicted.txt - contains the extracted relations of the test set
  * settings in the config.py file can be changed to affect the run, for example logs for the training process
    can be enabled by setting `CATBOOST_LOGGING_LEVEL = 'Info'`
