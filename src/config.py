
languages = ["dk", "sv", "nn", "nb", "fo", "is"]
datafolder = "data/raw_data/"
lang_to_label = dict(zip(languages,
                         [i for i in range(len(languages))]))

lang_to_label["da"] = 0

## The number of sentences pr language.
nr_datapoints = 50000
TEST_TRAIN_RATIO = 0.2

### Recognices characters
"https://en.wikipedia.org/wiki/Wikipedia:Language_recognition_chart"

## Define allowed character set
latin = 'abcdefghijklmnopqrstuvwxyz'
danish = 'æøå'
sweedish = 'åäöé'
island = 'áðéíóúýþæö'
fo = "áðíóúýæø"
all_chars = latin + danish + sweedish + island + fo
char_set = ''.join(sorted(''.join(set(all_chars))))
