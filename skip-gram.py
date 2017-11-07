import pandas as pd
import numpy as np
import tensorflow as tf

import time
import random

def find_invalid_target(ratings_df, user, idx, window_size=4):
    start = idx - window_size
    if start < 0:
        return True
    end = idx + window_size + 1
    if end >= ratings_df.shape[0]:
        return True
    if ratings_df.userId.iloc[start] != user or ratings_df.userId.iloc[end] != user:
        return True
    return False

def get_target(ratings_df, idx, window_size=4):
    start = idx - window_size
    if start < 0:
        return False
    
    end = idx + window_size + 1
    if end >= len(ratings_df):
        return False
    
    user = ratings_df.userId.iloc[idx]
    if ratings_df.userId.iloc[start] == user and ratings_df.userId.iloc[end] == user:
        half1 = ratings.movieId.iloc[start:idx].tolist()
        half2 = ratings.movieId.iloc[idx+1:end].tolist()
        return half1 + half2
    else:
        return False

def find_num_valid(ratings_df, window_size=4):
    invalid_targets = 0
    ratings = ratings_df.shape[0]
    for idx in range(ratings):
        if find_invalid_target(ratings_df, ratings_df.userId.iloc[idx], idx, window_size):
            invalid_targets += 1
            
        if idx % 500000 == 0 and idx != 0:
            print("{}/{} entries processed".format(idx, len(ratings_df.userId)))
    print("Done")
    return ratings - invalid_targets

def fix_movie_indices(ratings_list, movies_df, vocab_to_int):
    fix_dict = movies_df.set_index('movieId')['title'].to_dict()
    fixed = []
    
    for movie in ratings_list:
        fixed.append(vocab_to_int[fix_dict[movie]])
    return fixed

# TODO: preprocess dataframes so that this is not necessary during training time

def get_batches(dataset, movies_df, vocab_to_int, n_batches, batch_size, window_size=4):

    for idx in range(0, len(dataset)-100, batch_size):
        x, y = [], []
        batch = dataset.iloc[idx:idx+batch_size+100] # buffer for invalid targets
                                                     # TODO: eliminate the need for this workaround
        
        ii = 0
        ix = 0
        while ii < batch_size:
            batch_x = batch.movieId.iloc[ix]
            batch_y = get_target(batch, ix, window_size)
            if batch_y:
                ii += 1
                batch_x = fix_movie_indices([batch_x], movies_df, vocab_to_int)
                batch_y = fix_movie_indices(batch_y, movies_df, vocab_to_int)
                y.extend(batch_y)
                x.extend([batch_x]*len(batch_y))
            ix += 1
            if ix >= batch_size + 100:
            	break
        x = np.array(x).reshape((-1,))
        if x.shape[0] < batch_size:
        	raise StopIteration
        yield x, y
    
movies = pd.read_csv('/ml-latest/movies.csv')
ratings = pd.read_csv('/ml-latest/ratings.csv')

high_ratings = ratings[ratings.rating >= 4]
high_ratings.drop(['rating', 'timestamp'], axis=1, inplace=True)

N_VALID = 10961843

int_to_vocab = {ii: movie for ii, movie in enumerate(list(movies['title']))}
vocab_to_int = {movie: ii for ii, movie in enumerate(list(movies['title']))}

train_graph = tf.Graph()
with train_graph.as_default():
    inputs = tf.placeholder(tf.int32, [None], name='inputs')
    labels = tf.placeholder(tf.int32, [None, None], name='labels')
    
n_vocab = len(int_to_vocab)
n_embedding = 200 # Number of embedding features 
with train_graph.as_default():
    embedding = tf.Variable(tf.random_uniform((n_vocab, n_embedding), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs)
    
# Number of negative labels to sample
n_sampled = 100
with train_graph.as_default():
    softmax_w = tf.Variable(tf.truncated_normal((n_vocab, n_embedding), stddev=0.1))
    softmax_b = tf.Variable(tf.zeros(n_vocab))
    
    # Calculate the loss using negative sampling
    loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b, 
                                      labels, embed,
                                      n_sampled, n_vocab)
    
    cost = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

with train_graph.as_default():
    ## From Thushan Ganegedara's implementation
    valid_size = 16 # Random set of words to evaluate similarity on.
    valid_window = 100
    # pick 8 samples from (0,100) and (1000,1100) each ranges. lower id implies more frequent 
    valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
    valid_examples = np.append(valid_examples, 
                               random.sample(range(1000,1000+valid_window), valid_size//2))

    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    
    # We use the cosine distance:
    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
    normalized_embedding = embedding / norm
    valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
    similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))

epochs = 1
batch_size = 32
window_size = 4

n_batches = N_VALID//batch_size

with train_graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=train_graph) as sess:
    iteration = 1
    loss = 0
    sess.run(tf.global_variables_initializer())

    for e in range(1, epochs+1):
        batches = get_batches(high_ratings, movies, vocab_to_int, n_batches, batch_size, window_size)
        start = time.time()
        for x, y in batches:
            
            feed = {inputs: x,
                    labels: np.array(y)[:, None]}
            train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)
            
            loss += train_loss
            
            if iteration % 100 == 0: 
                end = time.time()
                print("Epoch {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Avg. Training loss: {:.4f}".format(loss/100),
                      "{:.4f} sec/batch".format((end-start)/100))
                loss = 0
                start = time.time()
            
            #if iteration % 1000 == 0:
            #    # note that this is expensive (~20% slowdown if computed every 500 steps)
            #    sim = similarity.eval()
            #    for i in range(valid_size):
            #        valid_word = int_to_vocab[valid_examples[i]]
            #        top_k = 8 # number of nearest neighbors
            #        nearest = (-sim[i, :]).argsort()[1:top_k+1]
            #        log = 'Nearest to %s:' % valid_word
            #        for k in range(top_k):
            #            close_word = int_to_vocab[nearest[k]]
            #            log = '%s %s,' % (log, close_word)
            #        print(log)
            
            iteration += 1
    save_path = saver.save(sess, "/output/movie.ckpt")
    embed_mat = sess.run(normalized_embedding)