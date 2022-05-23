1.  eval: cross-validation
    python run.py -mode eval - event_type typhoon
        event_type: typhoon/quake
2.  train
    python run.py -mode train -event_type typhoon -saved_model_path ../data/saved_models/ 
3.  prediction 
    python run.py -mode prediction -event_type quake -saved_model_path ../data/saved_models/ -input_new_data_path ../data/unlabeled_data/new_data.csv -output_new_data_path ../data/output_data/new_data.csv


4.  dependencies 
    emoji==0.6.0
    HTMLParser==0.0.2
    nltk==3.5
    numpy==1.21.5
    pandas==1.1.5
    scikit_learn==1.1.1
    torch==1.9.0
    transformers==4.2.1

