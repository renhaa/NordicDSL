all: requirements download_data preprocess train_fasttest_supervised

requirements: 
	pip3 install -r requirements.txt 

download_data: 
	sh download_data.sh

preprocess:
	python3 preprocess.py ../data/raw_data/wikipedia/ data/wikipedia/

train_fasttest_supervised: 
	python3 train_fasttext_supervised.py data/wikipedia/ 
	
run_fasttest_supervised: 
	python3 run_fasttext_model.py data/wikipedia/fasttextmodel.ftz ../data/smalltest.txt
