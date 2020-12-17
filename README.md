# NordicDSL

This project investigated the Discriminating between Similar Languages (DSL) task. The project was an indepent MSc level project at the IT University of Copenhagen. 

It develop a machine learning based pipeline for automatic language identification for the Nordic languages. Concretely we will focus on discrimination between six similar Nordic languages: Danish, Swedish, Norwegian (Nynorsk), Norwegian (Bokmål), Faroese and Icelandic. Multiple neural and non-neural approaches were evaluated for this novel framing of a difficult task, across genres, leading to good results.

# Usage
To install requirements, download preproces the data and finally train a FastText model simply run `make`.

## Data and pretrained model
The dataset and a pretrained fasttext model can be downloaded from:  http://itu.dk/people/renha/NordicDSL/

# License 

## Dataset

<a rel="license" href="http://creativecommons.org/licenses/by-sa/3.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/3.0/88x31.png" /></a>
<br />The wikipedia dataset is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/3.0/">Creative Commons Attribution-ShareAlike 3.0 Unported License</a>.


## Fasttext
Facebook's FastText is licensed under a [MIT License](https://github.com/facebookresearch/fastText/blob/master/LICENSE)