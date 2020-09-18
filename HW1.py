import matplotlib
import numpy as np
import pandas as pd
from collections import Counter
import math

from jedi.refactoring import inline
from tqdm import tqdm
from typing import List,Dict
from IPython.display import Image

from IPython.core.display import HTML
from pathlib import Path

from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download("stopwords")
nltk.download("punkt")
from string import punctuation, ascii_lowercase
from nltk.corpus import stopwords

DEBUG = True


"""
Recommended to start with a small number to get a feeling for the preprocessing with prints (N_ROWS_FOR_DEBUG = 2)
later increase this number for 5*10**3 in order to see that the code runs at reasonable speed. 
When setting Debug == False, our code implements bow.fit() in 15-20 minutes according to the tqdm progress bar. Your solution is not supposed to be much further than that.
"""
N_ROWS_FOR_DEBUG = 5*10**3

INPUT_FILE_PATH = Path("lyrics.csv")
BOW_PATH = Path("bow.csv")
N_ROWS = N_ROWS_FOR_DEBUG if DEBUG else None
CHUNCK_SIZE = 5 if DEBUG else 5*10**3
tqdm_n_iterations = N_ROWS//CHUNCK_SIZE +1 if DEBUG else 363*10**3//CHUNCK_SIZE + 1
COLS = [5]

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
allowed_symbols = set(l for l in ascii_lowercase)

def preprocess_sentence(sentence : str) -> List[str]:
    output_sentence = []
    for word in word_tokenize(sentence): # O(D) * O(V). number of documents times number of words
        word = word.lower() # 0(1)
        if word in stop_words:
            continue
        for char in word:
            if char not in allowed_symbols:
                word = word.replace(char, '')
        word = stemmer.stem(word)
        if len(word) <= 1:
            continue
        output_sentence.append(word)
    return output_sentence


def get_data_chuncks() -> List[str]:
    for i ,chunck in enumerate(pd.read_csv(INPUT_FILE_PATH, usecols = COLS, chunksize = CHUNCK_SIZE, nrows = N_ROWS)):
        chunck = chunck.values.tolist()
        yield [chunck[i][0] for i in range(len(chunck))]

class TfIdf:
    def __init__(self):
        self.unigram_count =  Counter()
        self.bigram_count = Counter()
        self.trigram_count = Counter()
        self.document_term_frequency = Counter()
        self.word_document_frequency = {}
        self.inverted_index = {}
        self.doc_norms = {}
        self.n_docs = -1
        self.sentence_preprocesser = preprocess_sentence
        self.bow_path = BOW_PATH

    def update_counts_and_probabilities(self, sentence :List[str],document_id:int) -> None:
        sentence_len = len(sentence)
        self.document_term_frequency[document_id] = sentence_len
        for i,word in enumerate(sentence):
            self.unigram_count[word] += 1
            if word in self.inverted_index:
                if document_id in self.inverted_index[word]:
                    self.inverted_index[word][document_id] += 1
                else:
                    self.inverted_index[word][document_id] = 1
            else:
                self.inverted_index.update({word:{document_id: 1}})
            if i < sentence_len - 1:
                bigram = word + "," + sentence[i+1]
                self.bigram_count[bigram] += 1
                if i < sentence_len - 2:
                    trigram = bigram + "," +sentence[i+2]
                    self.trigram_count[trigram] += 1


    def fit(self) -> None:
        for chunck in tqdm(get_data_chuncks(), total = tqdm_n_iterations):
            for sentence in chunck: # D * V
                self.n_docs += 1
                self.doc_norms[self.n_docs] = 0
                if not isinstance(sentence, str):
                    continue
                sentence = self.sentence_preprocesser(sentence) #O(V)
                if sentence:
                    self.update_counts_and_probabilities(sentence,self.n_docs) #O(V)
        self.save_bow() # bow is 'bag of words'
        self.compute_word_document_frequency()  # D*V
        self.update_inverted_index_with_tf_idf_and_compute_document_norm() # (D*V)*(D) + (D) = D^2*V

    def compute_word_document_frequency(self):
        for word in self.inverted_index.keys():
            self.word_document_frequency[word] = len(self.inverted_index[word])



    def update_inverted_index_with_tf_idf_and_compute_document_norm(self):
        for word in self.inverted_index.keys():
            for doc in self.inverted_index[word].keys():
                tf = self.inverted_index[word][doc] / self.document_term_frequency[doc]
                idf = np.log10(self.n_docs/self.word_document_frequency[word])
                self.inverted_index[word][doc] = tf*idf
                self.doc_norms[doc] += np.square(self.inverted_index[word][doc])


        for doc in self.doc_norms.keys():
            self.doc_norms[doc] = np.sqrt(self.doc_norms[doc])

    def save_bow(self):
        pd.DataFrame([self.inverted_index]).T.to_csv(self.bow_path)


from six.moves import cPickle

#run calss - comment this after first run
tf_idf = TfIdf()
tf_idf.fit()

#save object - comment this after first run
# f = open('obj_debug.save', 'wb')
# cPickle.dump(tf_idf, f, protocol=cPickle.HIGHEST_PROTOCOL)
# f.close()
#Read boject
#f = open('obj_debug.save', 'rb')
#tf_idf = cPickle.load(f)
#f.close()


print('unigram_count: ' + str(len(tf_idf.unigram_count)))
print('potential : ' + str(len(tf_idf.unigram_count) * (len(tf_idf.unigram_count) - 1)))
print('bigram_count: ' + str(len(tf_idf.bigram_count)))
print('trigram_count: ' + str(len(tf_idf.trigram_count)))
print('Done 1.1')




class DocumentRetriever:
    def __init__(self, tf_idf):
        self.sentence_preprocesser = preprocess_sentence
        self.vocab = set(tf_idf.unigram_count.keys())
        self.n_docs = tf_idf.n_docs
        self.inverted_index = tf_idf.inverted_index
        self.word_document_frequency = tf_idf.word_document_frequency
        self.doc_norms = tf_idf.doc_norms

    def rank(self,query : Dict[str,int],documents: List[Counter],metric: str ) -> Dict[str, float]:
        result = {} # key: DocID , value : float , simmilarity to query
        query_len = np.sum(np.array(list(query.values())))
        for word, v in documents.items():
            query[word]=query[word]/query_len*np.log10(self.n_docs/self.word_document_frequency[word])
            for doc in documents[word]:
                result[doc]=result.get(doc,0)+query[word]*self.inverted_index[word][doc]
        if metric == 'cosine':
            for doc in result.keys():
                result[doc]=result[doc]/self.doc_norms[doc]

        return result


    def sort_and_retrieve_k_best(self, scores: Dict[str, float],k :int):
        sorted_list_keys =  {key : v for key, v in sorted(scores.items(), key = lambda item: item[1], reverse=True)[:k]}
        return list(sorted_list_keys.keys())

    def reduce_query_to_counts(self, query : List)->  Counter:
        query_counts = Counter()
        for i, word in enumerate(query):
            query_counts[word] += 1
        return query_counts

    def get_top_k_documents(self,query : str, metric: str , k = 5) -> List[str]:
        query = self.sentence_preprocesser(query)
        query = [word for word in query if word in self.vocab] # filter nan
        query_bow = self.reduce_query_to_counts(query)
        relavant_documents = {word : self.inverted_index.get(word) for word in query}
        ducuments_with_similarity = self.rank(query_bow,relavant_documents, metric)
        return self.sort_and_retrieve_k_best(ducuments_with_similarity,k)


dr = DocumentRetriever(tf_idf)
query = "Better stop dreaming of the quiet life, 'cause it's the one we'll never know And quit running for that runaway bus 'cause those rosy days are few And stop apologizing for the things you've never done 'Cause time is short and life is cruel but it's up to us to change This town called malice"

cosine_top_k = dr.get_top_k_documents(query, 'cosine')
print(cosine_top_k)
inner_product_top_k = dr.get_top_k_documents(query, 'inner_product')
print(inner_product_top_k)

for index, song in enumerate(pd.read_csv(INPUT_FILE_PATH,usecols = [5]).iloc[cosine_top_k]['lyrics']):
    sep = "#"*50
    print(F"{sep}\nsong #{index} \n{song} \n{sep}")

print('Done part 1.2')


def get_bigrams(word):
    for ngram in nltk.ngrams(word, 2):
        yield "".join(list(ngram))

def get_trigrams(word):
    for ngram in nltk.ngrams(word, 3):
        yield "".join(list(ngram))

"""
for example - get_bigrams is a generator, which is an object we can loop on:
for ngram in get_bigrams(word):
    DO SOMETHING
"""

class NgramSpellingCorrector:
    def __init__(self, unigram_counts: Counter, get_n_gram: callable):
        self.unigram_counts = unigram_counts
        self.ngram_index = {}
        self.get_n_grams = get_n_gram

    def build_index(self) -> None:
        for word in self.unigram_counts.keys():
            for ngram in self.get_n_grams(word):
                if self.ngram_index.get(ngram):
                    self.ngram_index[ngram].append(word)
                else:
                    self.ngram_index[ngram] = [word]


    def get_top_k_words(self,word:str,k=5) -> List[str]:
        total_words = sum(self.unigram_counts.values())
        list_x = [ngram for ngram in self.get_n_grams(word)]
        similarity_dict = {}
        for ngram in list_x:
            if self.ngram_index.get(ngram):
                for real_word in self.ngram_index.get(ngram):
                    if not similarity_dict.get(real_word):
                        prior = self.unigram_counts[real_word] / total_words
                        list_y = [ngram for ngram in self.get_n_grams(real_word)]
                        similarity_dict[real_word] = prior * jaccard_similarity(list_x, list_y)

        return sorted(similarity_dict.keys(),key=lambda item : similarity_dict[item],reverse=True)[:k]


def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


class BigramSpellingCorrector(NgramSpellingCorrector):
    def __init__(self, unigram_counts: Counter):
        super().__init__(unigram_counts, get_bigrams)


class TrigramSpellingCorrector(NgramSpellingCorrector):
    def __init__(self, unigram_counts: Counter):
        super().__init__(unigram_counts, get_trigrams)


out_of_vocab_word = 'supercalifragilisticexpialidocious'
print("out of vocabulary word is: " + out_of_vocab_word)
bigram_spelling_corrector = BigramSpellingCorrector(tf_idf.unigram_count)
bigram_spelling_corrector.build_index()
bigram_spelling_corrector.get_top_k_words(out_of_vocab_word)
trigram_spelling_corrector = TrigramSpellingCorrector(tf_idf.unigram_count)
trigram_spelling_corrector.build_index()
trigram_spelling_corrector.get_top_k_words(out_of_vocab_word)
print('Done 1.4')



print('Start part 1.5')

# for the probability smoothing
NUMERATOR_SMOOTHING = 1 # alpha in https://en.wikipedia.org/wiki/Additive_smoothing
DENOMINATOR_SMOOTHING = 10**4 # d in https://en.wikipedia.org/wiki/Additive_smoothing
def sentence_log_probabilty(unigrams : Counter, bigrams  : Counter,trigrams : Counter, sentence: str):
    bigram_log_likelilhood, trigram_log_likelilhood = 0, 0
    words_in_sentence = sentence.split()
    n_words = len(words_in_sentence)
    for index, word in  enumerate(words_in_sentence):
        if index == 0:
            bigram_log_likelilhood += np.log((unigrams[word] + NUMERATOR_SMOOTHING) / (sum(unigrams.values()) + DENOMINATOR_SMOOTHING))
        elif index == 1:
            bigram_log_likelilhood += np.log(((bigrams[words_in_sentence[index-1]+","+word]  / sum(bigrams.values())) + NUMERATOR_SMOOTHING)/
                                             ((unigrams[words_in_sentence[index-1]] / sum(unigrams.values())) + DENOMINATOR_SMOOTHING))
            trigram_log_likelilhood += np.log((bigrams[words_in_sentence[index-1]+","+word] + NUMERATOR_SMOOTHING) / (sum(bigrams.values()) + DENOMINATOR_SMOOTHING))
            print ("test for itay: " + str(trigram_log_likelilhood))
        else:
            bigram_log_likelilhood += np.log(((bigrams[words_in_sentence[index-1]+","+word]  / sum(bigrams.values())) + NUMERATOR_SMOOTHING)/
                                              ((unigrams[words_in_sentence[index-1]] / sum(unigrams.values())) + DENOMINATOR_SMOOTHING))
            trigram_log_likelilhood += np.log(((trigrams[words_in_sentence[index-2]+","+words_in_sentence[index-1]+","+word] / sum(trigrams.values()))+ NUMERATOR_SMOOTHING) /
                                              ((bigrams[words_in_sentence[index-2]+","+words_in_sentence[index-1]] / sum(bigrams.values())) + DENOMINATOR_SMOOTHING))

            
    print(F"Bigram log likelihood is {bigram_log_likelilhood}")
    print(F"Trigram log likelihood is {trigram_log_likelilhood}")
    return



sentence = "spiderman spiderman does whatever a spider can"

sentence_log_probabilty(tf_idf.unigram_count, tf_idf.bigram_count, tf_idf.trigram_count, sentence)

## 1.51 Language model: B
#For each model what is the next word prediciton for the sentnence "i am"?
sentence_to_predict = "i am"

bigram_list = [bigram for bigram in tf_idf.bigram_count.keys() if str(bigram).startswith(sentence_to_predict.split()[-1] + ",")]
trigram_list = [trigram for trigram in tf_idf.trigram_count.keys() if str(trigram).startswith(sentence_to_predict.split()[-2] + "," + sentence_to_predict.split()[-1] + ",")]

dict_bigram = {bigram : np.log((tf_idf.bigram_count[bigram] + NUMERATOR_SMOOTHING) /(tf_idf.unigram_count[bigram.split(',')[0]] + NUMERATOR_SMOOTHING*DENOMINATOR_SMOOTHING)) for bigram in bigram_list}
if (bool(dict_bigram)): next_bigram_word = max(dict_bigram, key=dict_bigram.get)
else: next_bigram_word = max(tf_idf.unigram_count, key=tf_idf.unigram_count.get)


dict_trigram = {trigram : np.log((tf_idf.trigram_count[trigram] + NUMERATOR_SMOOTHING) / (tf_idf.bigram_count[trigram.split(',')[0]+","+trigram.split(',')[1]] + NUMERATOR_SMOOTHING*DENOMINATOR_SMOOTHING)) for trigram in trigram_list}
if (bool(dict_trigram)): next_trigram_word = max(dict_trigram, key=dict_trigram.get)
else: next_trigram_word = max(tf_idf.unigram_count, key=tf_idf.unigram_count.get)


print("Next word predicition according to bigram model is: " + next_bigram_word.split(',')[-1])
print("The full sentece is: " + sentence_to_predict + " " + next_bigram_word.split(',')[-1])
print("Next word predicition according to trigram model is: " + next_trigram_word.split(',')[-1])
print("The full sentece is: " + sentence_to_predict + " " + next_trigram_word.split(',')[-1])

print('Done 1.5')
