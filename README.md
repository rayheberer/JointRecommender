# Movie Embeddings
I make vector representations of highly rated Movies, by embedding them using the skip-gram model often used for NLP.

## Data
The reviews are from the [MovieLens Dataset](https://grouplens.org/datasets/movielens/). Specifically, the "latest" dataset. In the words of the dataset maintainers:

> This dataset (ml-latest) describes 5-star rating and free-text tagging activity from [MovieLens](http://movielens.org), a movie recommendation service. It contains 26024289 ratings and 753170 tag applications across 45843 movies. These data were created by 270896 users between January 09, 1995 and August 04, 2017. This dataset was generated on August 04, 2017.

I have also uploaded the dataset as a [FloydHub dataset](https://www.floydhub.com/rayheberer/datasets/movielens-latest).

## Model

I implemented the model in Tensorflow using the techinques described in [this tutorial](https://www.tensorflow.org/tutorials/word2vec).