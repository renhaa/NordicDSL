# NordicDSL

This project investigated the Discriminating between Similar Languages (DSL) task. The project was an indepent MSc level project at the IT University of Copenhagen. 

It develop a machine learning based pipeline for automatic language identification for the Nordic languages. Concretely we will focus on discrimination between six similar Nordic languages: Danish, Swedish, Norwegian (Nynorsk), Norwegian (Bokmål), Faroese and Icelandic. Multiple neural and non-neural approaches were evaluated for this novel framing of a difficult task, across genres, leading to good results.

# Usage
To install requirements, download preproces the data and finally train a FastText model simply run `make`.

## Data and pretrained model
The dataset and a pretrained fasttext model can be downloaded from:  http://itu.dk/people/renha/NordicDSL/

# Referring to this tool

If you use this tool, please cite the relevant paper, [Discriminating Between Similar Nordic Languages](https://aclanthology.org/2021.vardial-1.8/):

> Haas & Derczynski, 2021. "Discriminating Between Similar Nordic Languages". In Proceedings of the Eighth Workshop on NLP for Similar Languages, Varieties and Dialects

```
@inproceedings{haas-derczynski-2021-discriminating,
    title = "Discriminating Between Similar Nordic Languages",
    author = "Haas, Ren{\'e}  and
      Derczynski, Leon",
    booktitle = "Proceedings of the Eighth Workshop on NLP for Similar Languages, Varieties and Dialects",
    month = apr,
    year = "2021",
    address = "Kiyv, Ukraine",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.vardial-1.8",
    pages = "67--75",
    abstract = "Automatic language identification is a challenging problem. Discriminating between closely related languages is especially difficult. This paper presents a machine learning approach for automatic language identification for the Nordic languages, which often suffer miscategorisation by existing state-of-the-art tools. Concretely we will focus on discrimination between six Nordic languages: Danish, Swedish, Norwegian (Nynorsk), Norwegian (Bokm{\aa}l), Faroese and Icelandic.",
}
```

# License 

## Dataset

<a rel="license" href="http://creativecommons.org/licenses/by-sa/3.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/3.0/88x31.png" /></a>
<br />The wikipedia dataset is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/3.0/">Creative Commons Attribution-ShareAlike 3.0 Unported License</a>.


## Fasttext
Facebook's FastText is licensed under a [MIT License](https://github.com/facebookresearch/fastText/blob/master/LICENSE)
