""" Features

The objective of this task is to explore the corpus, deals.txt.

The deals.txt file is a collection of deal descriptions, separated by a new line, from which
we want to glean the following insights:

The Following code conducted NLP on the deals.txt, including parsing the document, extracting 
noun pharases, key word (noun pharases word) normaization, terms aggregation. The popularity is 
based on the frequency of key word in all the deals. Note that if a key word appears more than 
one time in a deal, it's counted only one. 
The final output is saved in termsPopularity.csv, from which one can see the popularity.  

1. What is the most popular term across all the deals?
Answer: I list top 20 terms as bellow. It is worth nothing to mention that the noun pharase extraction
is not perfect, there are still some adjective and verbs left. 
One can see top terms such as link, order, shop, price, gift, hotel et al. 

link	4782
shop	4318
save	4026
ship	3963
code	3093
product	2403
order	2321
page	2120
sale	1904
price	1829
deal	1782
gift	1739
offer	1666
book	1660
text	1634
coupon	1609
onlin	1574
day	1259
hotel	1253
home	1102

2. What is the least popular term across all the deals?
Answer: There are a lot of low frequency terms, many of which are product acronyms or short names.
One meaningful term is CD-DVD-storage. 
3. How many types of guitars are mentioned across all the deals?
Answer: According to the extracted terms, guitar	appeared in 88 deals. Hence, 88 types assuming each deal discusses 
a different guitar.
"""
import nltk
import csv
import re

from nltk.corpus import stopwords

def unique(seq):
    """Keep only unique words in a deal"""
    seen = set()
    seen_add = seen.add
    return [ x for x in seq if x not in seen and not seen_add(x)]
def createOrUpdate (word,dict_map):
    if (dict_map.has_key(word)):
        dict_map[word] += 1;
    else:
        dict_map[word] =1;
    return dict_map
def leaves(tree):
    """Finds NP (nounphrase) leaf nodes of a chunk tree."""
    for subtree in tree.subtrees(filter = lambda t: t.node=='NP'):
        yield subtree.leaves()
def normalise(word):
    """Normalises words to lowercase and stems and lemmatizes it."""
    word = word.lower()
    word = stemmer.stem_word(word)
    word = lemmatizer.lemmatize(word)
    return word
 
def acceptable_word(word):
    """Checks conditions for acceptable word: length, stopword."""
    word= word.lower()
    accepted = bool(2 <= len(word) <= 40 and bool(re.match('[a-z]+', word)) and word not in stopwords)
    return accepted
    
def saveDict(fn,dict_rap):
    """Write term frequency dictionary to a csv file"""
    f=open(fn, "wb")
    w = csv.writer(f)
    for key, val in dict_rap.items():
        w.writerow([key, val])
    f.close()
def get_terms(tree):
    """Extracting noun pharases from grammar tree"""
    for leaf in leaves(tree):
        term = [normalise(w) for w,t in leaf if acceptable_word(w)]
        yield term
stemmer = nltk.PorterStemmer()
lemmatizer = nltk.WordNetLemmatizer()
stopwords = stopwords.words('english')
terms2freq={};
sentence_re = r'''(?x) # set flag to allow verbose regexps
([A-Z])(\.[A-Z])+\.? # abbreviations, e.g. U.S.A.
| \w+(-\w+)* # words with optional internal hyphens
| \.\.\. # ellipsis
| [][,;"'?():-_`] # these are separate tokens
'''
grammar = r"""
NBAR:
{<NN.*|JJ>*<NN.*>} # Nouns and Adjectives, terminated with Nouns
NP:
{<NBAR>}
{<NBAR><IN><NBAR>} # Above, connected with in/of/etc...
"""
chunker = nltk.RegexpParser(grammar)
lineNum=0;
with open('../data/deals.txt','r') as fp:
    for line in fp:
        lineNum +=1
        if (lineNum%1000)==0:
            print "processed " + repr(lineNum) + " lines!"
        if len(line) ==0:
            continue
        line=line.strip()        
        final_words=[];
        toks = nltk.regexp_tokenize(line, sentence_re)
        postoks = nltk.tag.pos_tag(toks)
        tree = chunker.parse(postoks)
        # extract noun pharases
        terms = get_terms(tree)
        for term in terms:
            for word in term:
                final_words.append(word);
        #only keep unique terms for one deal        
        final_words = unique(final_words)
        #print final_words        
        for word in final_words:
            terms2freq = createOrUpdate(word,terms2freq)
fp.closed
#post processing: remove two frequent but less intersting terms
if (terms2freq.has_key("com")):
    terms2freq.pop("com", None)
if (terms2freq.has_key("free")):
    terms2freq.pop("free", None)
#min_value, max_value = sys.maxint,0
#for key, value in terms2freq.iteritems():
#    if (value < min_value):
#        min_value=value
#        min_key=key
#    if (value > max_value): 
#        max_value=value
#        max_key=key
#print "most popular term:"+ repr(max_key) + " , appear " + repr(max_value) + " times!"
#print "least popular term:"+ repr(min_key) + " , appear " + repr(min_value) + " times!"
# write to csv file    
saveDict("termFreq.csv",terms2freq)