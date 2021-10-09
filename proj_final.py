import os, math, operator
from collections import defaultdict, Counter
from nltk.corpus import PlaintextCorpusReader
from nltk import FreqDist, word_tokenize
from nltk.util import ngrams
from sklearn.cross_validation import KFold
from svmutil import *
import pickle, itertools

Data_root = '/home1/c/cis530/project/data'
Train_set = Data_root + '/project_articles_train'
Test_set = Data_root + '/project_articles_test'


# project: authorship attribution
# lexical features: vocabulary richness (not very good): type-token ratio
# the first dozen of the most frequent words (include stop words, usually dominated by closed class words)
#   - function words are sued in a largely unconscious manner by the authors and they are topic-independent
# short English texts: up to 4-grams character features (tolerant to noise, computationally simplistic)
# the most frequent character n-grams are the most important features for stylistic purposes
# syntactic features: require robust and accurate NLP tools to perform parsing
# POS tag frequencies or POS tag n-gram frequencies (provide only a hint of the structural analysis of snetneces)
# unigram, bigrams, trigrams of morpho-syntactic tags (a large number of features)
# NLTK: sentence splitting, POS tagging, text chunking, partial parsing
# semantic features: functional features (associate certain words/phrases with semantic information)
# application-specific features: structural measures (such as the use of greetings in email messages)

# with a large feature set, one can apply feature selection algo to reduce the dimensionality of the representation,
# which also helps the classifier to avoid overfitting on the training data.
# feature selection: frequency (the best), odds-ratio, information gain (combination)

# Task: given a piece of writing, to determine whether or not Gina Kolata was the author
# Baseline system: an SVM that uses the top-1000 most common words in the training data as features
# training set: 12,113 excerpts of roughly 150-200 words
#               - 1: articles by Kolata
#               - 0: excerpts from other authors
# test set: 1346 excerpts
# k-fold cross-validation, k = 9


def standardize(rawexcerpt):    # convert a string input to a list of lowercase tokens
    return [w.encode('utf8') for w in word_tokenize(rawexcerpt.decode('utf8').lower())]


def load_training_excerpts(filepath):
    kolata = []
    nonkolata =[]
    with open(filepath, 'rU') as fin:
        for line in fin:
            if line.strip().split('\t')[1] == '0':
                nonkolata.append(line.strip().split('\t')[0])
            else:
                kolata.append(line.strip().split('\t')[0])
    return kolata, nonkolata


def load_testing_excerpts(filepath):
    excerpts = []
    with open(filepath, 'rU') as fin:
        for line in fin:
            excerpts.append(line.strip())
    return excerpts


def flatten(listoflists):
    return [item for sublist in listoflists for item in sublist]


def get_tf(sample):
    """
    :param sample: a list of lowercase strings
    :return: a dict with a key for each unique word in the input sample and int values giving the term frequency for each key
    """
    freqs = FreqDist(sample)
    return dict(freqs)


def get_idf(corpus): ## list of lists of lowercase strings, each being words in a doc
    df_dict = defaultdict(int)
    df_dict["<unk>"] = 1
    for excerpt in corpus:
        lstitems = list(set(excerpt))
        for item in lstitems: df_dict[item] += 1
    N = len(corpus)+0.0
    idf_dict = {item: math.log(N/v) for item,v in df_dict.items()}
    return idf_dict


# get logarithmic relative frequency:
# F_log(w_k, d_i) = log(1 + f(w_k, d_i) / f(d_i))
# f(w_k, d_i): the term frequency of type w_k in document d_i
# f(d_i): the number of types in document d_i
def get_logrel_tf(sample): # list of lowercase strings
    tf_dict = get_tf(sample)
    logreltf_dict = {item: math.log(1.0+ v/(len(tf_dict)+0.0)) for item,v in tf_dict.items()}
    return logreltf_dict


def get_logreltf_idf(logreltf_dict, idf_dict):
    l = []
    for x in logreltf_dict:
        if x in idf_dict:
            l.append((x, logreltf_dict[x]*idf_dict[x]))
        else:
            l.append((x, logreltf_dict[x]*idf_dict["<unk>"]))
    logreltf_idf = dict(l)
    return logreltf_idf


def get_precision(sample, ref):
    intersect = list(set(sample) & set(ref))
    precision = (len(intersect)+0.0)/len(sample)
    return precision

def get_recall(sample, ref):
    intersect = list(set(sample) & set(ref))
    recall = (len(intersect)+0.0)/len(ref)
    return recall


def create_feature_space(featlist):
    return dict([(voc, i+1) for (i, voc) in enumerate(set(featlist))])
    # i+1: index of features starts at 1 as required for libsvm format

def prep_data_for_svm(labels, training_data, feature_set, outfile):
    """
    :param labels: a list of 0s and 1s
    :param training_data: a list of lists of lowercase strings, each being an excerpt
    :param feature_set: a dict of feature set with index from 1 to V as the value
    :return: None
    """
    with open(outfile, 'w') as fout:
        idf_dict = get_idf(training_data)
        for i in range(len(training_data)):
            logreltf_dict = get_logrel_tf(training_data[i])
            logreltf_idf_dict = get_logreltf_idf(logreltf_dict, idf_dict)
            feat_index = []
            feat_value = []
            for k, v in feature_set.items():
                if k in logreltf_idf_dict:
                    feat_index.append(v)
                    feat_value.append(logreltf_idf_dict[k])
            # normalize the vector using the Euclidean norm
            d = math.sqrt(sum({feat_value[j] * feat_value[j] for j in range(len(feat_value))})+0.0)
            norm_feat_value = [float(value)/d for value in feat_value]
            outline = str(labels[i]) + ' ' + ' '.join(str(m)+':'+str(n) for m, n in zip(feat_index,norm_feat_value))
            #outline = str(labels[i]) + ' ' + ' '.join(str(m)+':'+str(n) for m, n in zip(feat_index,feat_value))
            print >> fout, outline


def combine_featvecs_svm(labels, training1_data, featset1, training2_data, featset2, pos_data, pos_feat, outfile):
    with open(outfile, 'w') as fout:
        idf_dict1 = get_idf(training1_data)
        idf_dict2 = get_idf(training2_data)
        for i in range(len(labels)):
            print i
            logreltf_dict1 = get_logrel_tf(training1_data[i])
            logreltf_idf_dict1 = get_logreltf_idf(logreltf_dict1, idf_dict1)
            feat_index1 = []
            feat_value1 = []
            feat_index2 = []
            feat_value2 = []
            for k, v in featset1.items():
                if k in logreltf_idf_dict1:
                    feat_index1.append(v)
                    feat_value1.append(logreltf_idf_dict1[k])
            logreltf_dict2 = get_logrel_tf(training2_data[i])
            logreltf_idf_dict2 = get_logreltf_idf(logreltf_dict2, idf_dict2)
            for s, t in featset2.items():
                if s in logreltf_idf_dict2:
                    feat_index2.append(t + len(featset1))
                    feat_value2.append(logreltf_idf_dict2[s])
            # normalize the vector using the Euclidean norm
            d1 = math.sqrt(sum({feat_value1[j] * feat_value1[j] for j in range(len(feat_value1))})+0.0)
            norm_feat_value1 = [float(value)/d1 for value in feat_value1]

            d2 = math.sqrt(sum({feat_value2[j] * feat_value2[j] for j in range(len(feat_value2))})+0.0)
            norm_feat_value2 = [float(value)/d2 for value in feat_value2]

            feat_index = [j for j in feat_index1]
            feat_index.extend([l for l in feat_index2])
            norm_feat_value = [j for j in norm_feat_value1]
            norm_feat_value.extend([l for l in norm_feat_value2])

            total_tags = len(pos_data)
            pos_dict = dict(FreqDist(pos_data[i]))
            for s, t in pos_feat.items():
                if s in pos_dict:
                    feat_index.append(t + len(featset1) + len(featset2))
                    norm_feat_value.append(float(pos_dict[s]) / float(total_tags))

            outline = str(labels[i]) + ' ' + ' '.join(str(m)+':'+str(n) for m, n in zip(feat_index,norm_feat_value))
            print >> fout, outline


def train_test_model(train_datafile, test_datafile):
    '''
    Use libsvm to train and test a model.
    Use libear kernel (option '-t 0')
    '''
    y_train, x_train = svm_read_problem(train_datafile)
    m = svm_train(y_train, x_train, '-t 0 -e .01 -m 1000 -h 0')
    y_test, x_test = svm_read_problem(test_datafile)
    p_label, p_acc, p_val = svm_predict(y_test, x_test, m)
    #print p_acc
    return p_label, p_acc, p_val


def get_mfw_feat(excerptlist, k):
    '''
    get most frequent k words as the feature space and return the k features
    :param excerptlist: a list of lists of paragraph
    :param k: integer
    :return: a list of k words
    '''
    excerpts = [standardize(x) for x in excerptlist]
    words = FreqDist(flatten(excerpts))
    return [ a for (a, b) in words.most_common(k)]



def get_mfngram_feat(excerptlist, n, k):
    """
    get most frequent character ngrams as the feature space and return the k features
    :param excerptlist: a list of lists of paragraph
    :param n, k: integer
    :return: a list of k ngrams
    """
    excerpts = [create_char_ngrams(x, n) for x in excerptlist]
    charNgrams = FreqDist(flatten(excerpts))
    #print "Number of ngrams: " + str(len(charNgrams))   #74483
    return [ a for (a, b) in charNgrams.most_common(k)]
    ## return all ngrams as feature
    #charNgrams = dict(FreqDist(flatten(excerpts)))
    #return [ a for a in charNgrams.iterkeys()]



def create_char_ngrams(sample, n):
    """
    :param sample: a list of strings
    :param n: int
    :return: a list of character ngrams
    """
    return [sample.lower()[i:i+n] for i in xrange(len(sample)-n)]

def load_stopwords():
    ret = set()
    with open("/home1/c/cis530/hw4/stopwords.txt",'rU') as f:
        for line in f:
            l = line.strip()
            if len(l) > 0:
                ret |= set(word_tokenize(l))
                #ret.add(l)
        f.close()
    return ret

def create_mapping(mapfile):
    '''
    Create a mapping from Penn Treebank tagset to Google Universal tagset
    using the tab-delimited mapping in mapfile.
    :param mapfile: str (relative path to mapfile)
    :return: dict
    '''
    d = {}
    with open(mapfile, 'rU') as fin:
        for line in fin:
            ptb,univ = line.split()
            d[ptb.strip()] = univ.strip()
    return d

def parse_taggedtrain(wsjfile, tagmap):
    '''
    Parse the tagged training or testing data file wsjfile.
    Return a list of lists of POS tags, where each inner list contains POS tags from a single sentence.
    In the output, the pos tag is mapped from the raw PTB tag in the wsjfile to its universal tag equivalent using tagmap.
    Assume whitespace word tokenization within sentences.
    Preprocessing = lowercasing of tokens
    :param wsjfile: str (path to wsjfile)
    :param tagmap: dict
    :return: list of lists of POS tags
    '''
    pos_tags = []
    with open(wsjfile, 'rU') as fin:
        for line in fin:
            excerpt = []
            for pair in line.strip().split(' '):
                if '/' in pair:
                    if pair.split('/')[1] in tagmap:
                        excerpt.append(tagmap[pair.split('/')[1]])
                    else:
                        excerpt.append(pair.split('/')[1])
            pos_tags.append(excerpt[:-1])
    return pos_tags

def parse_tagged(wsjfile, tagmap):
    '''
    Parse the tagged training or testing data file wsjfile.
    Return a list of lists of POS tags, where each inner list contains POS tags from a single sentence.
    In the output, the pos tag is mapped from the raw PTB tag in the wsjfile to its universal tag equivalent using tagmap.
    Assume whitespace word tokenization within sentences.
    Preprocessing = lowercasing of tokens
    :param wsjfile: str (path to wsjfile)
    :param tagmap: dict
    :return: list of lists of POS tags
    '''
    pos_tags = []
    with open(wsjfile, 'rU') as fin:
        for line in fin:
            excerpt = []
            for pair in line.strip().split(' '):
                if '/' in pair:
                    if pair.split('/')[1] in tagmap:
                        excerpt.append(tagmap[pair.split('/')[1]])
                    else:
                        excerpt.append(pair.split('/')[1])
            pos_tags.append(excerpt)
    return pos_tags

if __name__ == '__main__':

    pos_training, neg_training = load_training_excerpts(Train_set)
    ## len(pos_training) # 2531
    ## len(neg_training) # 9582

    #with open('pos_training.txt','w') as fout:
    #    for i in pos_training:
    #        print >> fout, i
    #with open('neg_training.txt','w') as fout:
    #    for i in neg_training:
    #        print >> fout, i

    #java -mx300m -classpath postagger-2006-05-21.jar edu.stanford.nlp.tagger.maxent.MaxentTagger -model wsj3t0-18-bidirectional/train-wsj-0-18 -file /home1/c/chenmao/proj/pos_training.txt > /home1/c/chenmao/proj/pos_train-POS-output.txt
    #java -mx300m -classpath postagger-2006-05-21.jar edu.stanford.nlp.tagger.maxent.MaxentTagger -model wsj3t0-18-bidirectional/train-wsj-0-18 -file /home1/c/chenmao/proj/neg_training.txt > /home1/c/chenmao/proj/neg_train-POS-output.txt
    #java -mx300m -classpath postagger-2006-05-21.jar edu.stanford.nlp.tagger.maxent.MaxentTagger -model wsj3t0-18-bidirectional/train-wsj-0-18 -file /home1/c/cis530/project/data/project_articles_test > /home1/c/chenmao/test-POS-output.txt


    #cv_pos = []
    #cv_neg = []
    ## HW3; using the libsvm library to implement a multi-class prediction support vector machine with a linear kernel
    ## Here we could use other kernels and implement a binary prediction SVM
    ## check if there are other better svm libraries than libsvm

    #kf_pos = KFold(2531, n_folds=9, shuffle=True, random_state=42)
    #for train_index, valid_index in kf_pos:
    #    cv_pos.append([list(train_index), list(valid_index)])
    #kf_neg = KFold(9582, n_folds=9, shuffle=True, random_state=42)
    #for train_index, valid_index in kf_neg:
    #    cv_neg.append([list(train_index), list(valid_index)])

    #fold_ct = 1
    #cv_results = [] # label, accuracy, value
    #sum_acc = 0.0

    ptb2univ = create_mapping('/home1/c/cis530/hw3/data/en-ptb.map')
    pos_train_POS = parse_tagged('/home1/c/chenmao/proj/pos_train-POS-output.txt', ptb2univ)
    neg_train_POS = parse_tagged('/home1/c/chenmao/proj/neg_train-POS-output.txt', ptb2univ)
    test_POS = parse_tagged('/home1/c/chenmao/test-POS-output.txt', ptb2univ)

    #for p, n in zip(cv_pos, cv_neg):
        # train: p[0] & n[0]
    #    train_excerpts = [pos_training[i] for i in p[0]]
    #    train_excerpts.extend([neg_training[i] for i in n[0]])
    #    train_labels = [1]*len(p[0])
    #    train_labels.extend([0]*len(n[0]))
        # validation: p[1] & n[1]
    #    valid_excerpts = [pos_training[i] for i in p[1]]
    #    valid_excerpts.extend([neg_training[i] for i in n[1]])
    #    valid_labels = [1]*len(p[1])
    #    valid_labels.extend([0]*len(n[1]))

    #    n_of_ngram = 4
    #    ngram_feat_dict = create_feature_space(get_mfngram_feat(train_excerpts, n_of_ngram, 20000))
    #    train_ngram_data = [create_char_ngrams(x,n_of_ngram) for x in train_excerpts]
    #    valid_ngram_data = [create_char_ngrams(x,n_of_ngram) for x in valid_excerpts]

    #    stopwds = load_stopwords() # len(stopwds) = 593
    #    wd_features = stopwds | set(get_mfw_feat(train_excerpts, 1000))
        #print "# of word featrues: " + str(len(wd_features)) # 830

    #    wd_feat_dict = create_feature_space(list(wd_features))
        #wd_feat_dict = create_feature_space(get_mfw_feat(train_excerpts, 1000))
    #    train_wd_data = [standardize(x) for x in train_excerpts]
    #    valid_wd_data = [standardize(x) for x in valid_excerpts]

        #train_file = 'cv' + str(fold_ct) + '_ngram_train.svm'
        #valid_file = 'cv' + str(fold_ct) + '_ngram_valid.svm'
        #train_file = 'cv' + str(fold_ct) + '_train.svm'
        #valid_file = 'cv' + str(fold_ct) + '_valid.svm'
    #    train_file = 'cv' + str(fold_ct) + '_ngram_wd_train.svm'
    #    valid_file = 'cv' + str(fold_ct) + '_ngram_wd_valid.svm'


        #prep_data_for_svm(train_labels, train_data, ngram_feat_dict, train_file)
        #prep_data_for_svm(valid_labels, valid_data, ngram_feat_dict, valid_file)
        #prep_data_for_svm(train_labels, train_data, wd_feat_dict, train_file)
        #prep_data_for_svm(valid_labels, valid_data, wd_feat_dict, valid_file)

        #train_ngram_featvec = feat_vec_for_svm(train_ngram_data, ngram_feat_dict)
        #train_wd_featvec = feat_vec_for_svm(train_wd_data, wd_feat_dict)


    #    combine_featvecs_svm(train_labels, train_ngram_data, ngram_feat_dict, train_wd_data, wd_feat_dict, train_file)
    #    combine_featvecs_svm(valid_labels, valid_ngram_data, ngram_feat_dict, valid_wd_data, wd_feat_dict, valid_file)

    #    p_label, p_acc, p_val = train_test_model(train_file, valid_file)
    #    cv_results.append([p_label, p_acc, p_val])
    #    print 'Model ' + str(fold_ct) + ' results:', p_acc #
    #    fold_ct += 1
    #    sum_acc += p_acc[0]

    #print "Average accuracy: " + str(sum_acc / len(cv_results))

    # 88.5412345378 % for using 1000 most frequent words as features
    # 86.6011457148 % for using 100 most frequent words as features
    # 88.2935686841 % for using 1000 most frequent 4-grams as features
    # 87.6413385086 % for using 1000 most frequent 5-grams as features
    # 87.501084679 % for using 1000 most frequent 3-grams as features
    # 88.9705268699 % for using 1000 most frequent 4-grams and 1000 most frequent words as features
    # leaderboard 1 (test1.txt): 87.7414561664 % for using 1000 most frequent 4-grams and 1000 most frequent words as features

    # 79.1381268444 % for using 1000 most frequent 4-grams and 1000 most frequent words as features without normalization
    # 89.300747968 % for doing L2 normalization separately for two different feature sets
    # 79.1876501952 % for doing L1 normalization separately for two different feature sets



    # 88.7890230318 % for doing L2 normalization separately for two different feature sets, with the first being stopwords list
    # 89.0614413001 % for doing L2 normalization separately for two different feature sets, with the first combining stopwords list & top-500 most common
    # 89.3585507629 % for doing L2 normalization separately for two different feature sets, with the first combining stopwords list & top-1000 most common
    # 89.680467832 % for doing L2 normalization separately for two different feature sets, with the first combining stopwords list & top-1000 most common, and second one 1500 ngrams
    # 90.5142325785 % for doing L2 normalization separately for two different feature sets, with the first combining stopwords list & top-1000 most common, and second one 2000 ngrams
    # 91.3398406222 % for doing L2 normalization separately for two different feature sets, with the first combining stopwords list & top-1000 most common, and second one 3000 ngrams
    # leaderboard 2 (test2.txt): 90.9361069837 % for doing L2 normalization separately for two different feature sets, with the first combining stopwords list & top-1000 most common, and second one 3000 ngrams

    # ~ 92.79881217520416 (Model1) % for doing L2 normalization separately for two different feature sets, with the first combining stopwords list & top-1000 most common, and second one 10000 ngrams



# ================= train on the whole training set and test on the testing set =================

    training = [i for i in pos_training]
    training.extend([j for j in neg_training])
    training_pos = [i for i in pos_train_POS]
    training_pos.extend([j for j in neg_train_POS])
    train_labels = [1] * len(pos_training)
    train_labels.extend([0] * len(neg_training))

    n_of_ngram = 4
    ngram_feat_dict = create_feature_space(get_mfngram_feat(training, n_of_ngram, 15000))
    training_ngram_data = [create_char_ngrams(x, n_of_ngram) for x in training]

    stopwds = load_stopwords() # len(stopwds) = 593
    wd_features = stopwds | set(get_mfw_feat(training, 1000))
    wd_feat_dict = create_feature_space(list(wd_features))
    training_wd_data = [standardize(x) for x in training]

    testing = load_testing_excerpts(Test_set)
    testing_pos = [i for i in test_POS]
    ## use [0]*len(x) if true labels are unavailable
    test_labels = [0] * len(testing)
    testing_ngram_data = [create_char_ngrams(x, n_of_ngram) for x in testing]
    testing_wd_data = [standardize(x) for x in testing]

    pos_tags = ['.','CONJ', 'NUM', 'X', 'DET', 'ADP', 'ADJ', 'VERB', 'NOUN', 'PRT', 'PRON', 'ADV']
    pos_feat_dict = create_feature_space(pos_tags)


    training_file = 'ngram_wd_training.svm'
    testing_file = 'ngram_wd_testing.svm'


    combine_featvecs_svm(train_labels, training_ngram_data, ngram_feat_dict, training_wd_data, wd_feat_dict, training_pos, pos_feat_dict, training_file)
    combine_featvecs_svm(test_labels, testing_ngram_data, ngram_feat_dict, testing_wd_data, wd_feat_dict, testing_pos, pos_feat_dict, testing_file)


    p_label, p_acc, p_val = train_test_model(training_file, testing_file)

    with open('test.txt', 'w') as fout:
        for l in p_label:
            print >> fout, int(l)

    # submission 1: acc = 0.877414561664 for using 1000 most frequent 4-grams and 1000 most frequent words as features

    # leaderboard 4 (test20000.txt): 93.4621099554 % for doing L2 normalization separately for two different feature sets, with the first combining stopwords list & top-1000 most common, and second one 20000 ngrams
    # leaderboard 5 (test15000_pos.txt): 94.4279346211 % for doing L2 normalization separately for two different feature sets, with the first combining stopwords list & top-1000 most common, and second one 20000 ngrams + pos tags
    # leaderboard 6 (test_real_15000_pos.txt): 93.4621099554 % for doing L2 normalization separately for two different feature sets, with the first combining stopwords list & top-1000 most common, and second one 15000 ngrams + pos tags


