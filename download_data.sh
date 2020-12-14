#!/bin/sh 

echo "DOWNLOADING small wiki test set :"
mkdir -p data/raw_data/wikipedia/
wget -c http://www.derczynski.com/resources/nordicdsl/dk.txt -O data/raw_data/wikipedia/dk.txt
wget -c http://www.derczynski.com/resources/nordicdsl/nb.txt -O data/raw_data/wikipedia/nb.txt
wget -c http://www.derczynski.com/resources/nordicdsl/fo.txt -O data/raw_data/wikipedia/fo.txt
wget -c http://www.derczynski.com/resources/nordicdsl/is.txt -O data/raw_data/wikipedia/is.txt
wget -c http://www.derczynski.com/resources/nordicdsl/nn.txt -O data/raw_data/wikipedia/nn.txt
wget -c http://www.derczynski.com/resources/nordicdsl/sv.txt -O data/raw_data/wikipedia/sv.txt

#CoNLL17 https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-1989
# echo "DOWNLOADING CoNLL17 data:"
# curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-1989{/Danish-annotated-conll17.tar,/Norwegian-Bokmaal-annotated-conll17.tar}

# Other options for CoNLL17:
# /Norwegian-Nynorsk-annotated-conll17.tar,/Old_Church_Slavonic-annotated-conll17.tar,
# /Persian-annotated-conll17.tar,/Polish-annotated-conll17.tar,
# /Portuguese-annotated-conll17.tar,/Romanian-annotated-conll17.tar,
# /Russian-annotated-conll17.tar,/Slovak-annotated-conll17.tar,
# /Slovenian-annotated-conll17.tar,/Spanish-annotated-conll17.tar,
# /Swedish-annotated-conll17.tar,/Turkish-annotated-conll17.tar,
# /Ukrainian-annotated-conll17.tar,/Urdu-annotated-conll17.tar,
# /Uyghur-annotated-conll17.tar,/Vietnamese-annotated-conll17.tar,
# /conll2017-surprise-languages.zip
# /Dutch-annotated-conll17.tar,/English-annotated-conll17.tar,
# /Estonian-annotated-conll17.tar,/Finnish-annotated-conll17.tar,/French-annotated-conll17.tar,/Galician-annotated-conll17.tar,/German-annotated-conll17.tar,/Greek-annotated-conll17.tar,/Hebrew-annotated-conll17.tar,/Hindi-annotated-conll17.tar,/Hungarian-annotated-conll17.tar,/Indonesian-annotated-conll17.tar,/Irish-annotated-conll17.tar,/Italian-annotated-conll17.tar,/Japanese-annotated-conll17.tar,/Kazakh-annotated-conll17.tar,/Korean-annotated-conll17.tar,/Latin-annotated-conll17.tar,/Latvian-annotated-conll17.tar,
# /word-embeddings-conll17.tar,/Ancient_Greek-annotated-conll17.tar,
# /Arabic-annotated-conll17.tar,/Basque-annotated-conll17.tar,
# /Bulgarian-annotated-conll17.tar,/Catalan-annotated-conll17.tar,
# /ChineseT-annotated-conll17.tar,/Croatian-annotated-conll17.tar,
# /Czech-annotated-conll17.tar,
    