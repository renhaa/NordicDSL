import wikipedia
import numpy as np
import random
import sys
import time
import requests

languages = ["dk", "sv", "nn", "nb", "fo","is"]
datafolder = "data/raw_data/"

def get_data(nr_summaries):
    for i in range(nr_summaries):
        for language in languages:
            wikipedia.set_lang(language)
            fname = datafolder + language + ".txt"
            summary = get_random_summary()
            if not summary == None:
                with open(fname, "a+") as f:
                    f.write(summary)
                    f.write("\n")

def get_random_summary():
    """ This method returns a random wikipedia summary.
    """
    #random page title
    p = wikipedia.random(42)

    # to to fetch the summary
    try:
        wikipage = wikipedia.page(p)
        return wikipage.summary

    except (wikipedia.DisambiguationError, wikipedia.PageError) as e:
        get_random_summary()
    except requests.exceptions.ConnectionError:
        time.sleep(0.1)
        get_random_summary()


def main():
    nr = int(sys.argv[1])
    get_data(nr)


if __name__ == '__main__':
    main()
