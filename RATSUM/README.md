How to run:

1. First create a concept file from the text file
2. Run the summarization code over the concept file created in step1.

Details:

1. Download Twitter POS Tagger and set the path in line 27 of code concept_extraction.py
    https://code.google.com/archive/p/ark-tweet-nlp/downloads
2. Check the path of place files.
3. Check the sample data in sample_data folder.

Sample Run: 

1. python3 concept_extraction.py ../sample_data/input_data.txt place_files/nepal_place.txt ../sample_data/input_concept.txt
2. python3 EXCRIS.py ../sample_data/input_concept.txt place_files/nepal_place.txt input_summary.txt

Dependencies
    gurobipy==9.5.1
    nltk==3.5
    numpy==1.19.5
    scikit_learn==1.1.1
    textblob==0.17.1
    times==0.7

