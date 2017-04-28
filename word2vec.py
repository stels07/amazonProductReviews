import gensim
import numpy as np
import json
from nltk import tokenize
import numpy as np
from sklearn import manifold
from sklearn.feature_extraction.text import CountVectorizer
import string
import pickle



#print model.similarity('woman', 'man')
#model['computer']
#summaryArray, textArray, text, asin, overall= read_reviews("select_1.json")
def read_reviews(filename):
    z = json.loads(open(filename, "r").read())
    summaryArray = []
    text=[]
    overall=[]
    asin=[]
    textArray=[]
    summary_1=[]
    for i in range(0, len(z)):
        summary = z[i].get("summary")
        #print summary
		#nn = tokenize.sent_tokenize(z[i].get("reviewText"))

        exclude = set(string.punctuation)
        summary = ''.join(ch for ch in summary if ch not in exclude)
        #print summary
        count_vectorizer = CountVectorizer(stop_words='english')
        count_vectorizer.fit_transform([summary])

        #print "Vocabulary:", count_vectorizer.vocabulary_.keys()
        #print nn
        summaryArray.append(count_vectorizer.vocabulary_.keys())

        reviewText=z[i].get("reviewText")
        reviewText = ''.join(ch for ch in reviewText if ch not in exclude)
        count_vectorizer.fit_transform([reviewText])
        textArray.append(count_vectorizer.vocabulary_.keys())

        # print data[0]
        text.append(z[i].get("reviewText"))
        overall.append(z[i].get("overall"))
        asin.append(z[i].get("asin"))
        summary_1.append(z[i].get("summary"))

    print summaryArray
    print textArray
    return summaryArray,textArray, text, summary_1, asin,overall


def reduce_dimension(X):
	# 300 dimension to 2 dimension
    #print X
    if len(X)==0:
        return
    from time import time
    t0 = time()
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    trans_data = tsne.fit_transform(X).T
    t1 = time()
    print("t-SNE: %.2g sec" % (t1 - t0))
    return np.transpose(trans_data)


def write_to_json(q, p,tt):
    results = []
    for index in range(0, len(q)):
        result = {}
        #result["text"] = text[index]
        #result["summary"] = summary[index]
        #result["overall"]=overall[index]
        result["word"]=q[index][0]
        result["text"] = q[index][1]
        result["asin"] = q[index][2]
        result["x"] = p[index][0]
        result["y"] = p[index][1]
        result["sentiment"] = tt[index]
        result["overall"]=q[index][3]
        result["review_id"]=q[index][4]
        result["summary"]=q[index][5]
        results.append(result)
    print type(results)

    with open('office_v1.json', 'w') as fp:
		json.dump(results, fp, ensure_ascii=False, separators=(',', ':'))

def to_vector(model, X, usePickle): #X's first col is an array of words, return to_vector and then average (300 dim vector)
    if len(X)==0:
        return
    y=[]
    if(usePickle==False):
        not_in_val=[]
        for i in range(0,len(X)):
            if X[i,0] in not_in_val:
                continue
            if X[i,0] not in model.vocab:
                print "this is not in vocab ",X[i,0]
                not_in_val.append(X[i,0])
            else:
                y.append(X[i])
        with open("not_in_val.txt", "wb") as fp:
            pickle.dump(not_in_val, fp)
    else:
        with open("not_in_val.txt", "rb") as fp:
            b = pickle.load(fp)
        print "not in val words are ", b
        for i in range(0, len(X)):
            if X[i,0] in b:
                continue
            else:
                #Y=np.vstack((Y,X[i]))
                y.append(X[i])
    Y=np.asarray(y)
    print "y is ",y
    print "Y is ",Y
    #w=model[Y[0][0]]
    w=[]
    for i in range(0,len(Y)):
        #w=np.add(w,model[Y[i][0]])
        w.append(model[Y[i][0]])
    #print w
    #w=np.divide(w,((len(Y)-1)*0.001))
    W=np.asarray(w)
    print "dim of w is ", W.shape
    print "dim of Y is ", Y.shape
    #return np.hstack((w,Y[1:,:]))
    return (W, Y)

def bow_w_info(totalArray,text, asin, overall, summary):  # attach asin, sentiment, freq to each word
    print totalArray
    count=0
    p=[]
    for x in totalArray:
        for i in range(0,len(x)):
            p.append([x[i],text[count], asin[count], overall[count], count, summary[count]])   #create more text and asin to match the words
        count += 1
    p=np.asarray(p)
    print "the dim of p is ", p.shape
    #print "the list of matrix is ",p
    return np.asarray((p))

def add_sentiment(p):
    dicts_from_file = {}
    with open('AFINN-111.txt', 'r') as inf:
        for line in inf:
            (key, val) = line.split('\t')
            dicts_from_file[key] = val[:-1]
    print dicts_from_file
    res=[]
    for i in range(0,len(p)):
        if p[i] in dicts_from_file:
            res.append(dicts_from_file.get(p[i]))
        else:
            res.append(100)
    return np.asarray(res)

if __name__ == '__main__':

    summaryArray, textArray, text, summary, asin, overall= read_reviews("select_1.json")

    # Load Google's pre-trained Word2Vec model.
    model = gensim.models.Word2Vec.load_word2vec_format('./word2vec/GoogleNews-vectors-negative300.bin', binary=True)
    #model=gensim.models.KeyedVectors.load_word2vec_format('./word2vec/GoogleNews-vectors-negative300.bin', binary=True)
    p=[]
    totalArray=[]
    for i in range(0,len(summaryArray)):
        totalArray.append(summaryArray[i]+textArray[i])
    # for d in totalArray:
    #     p.append(to_vector(model,d))
    q=bow_w_info(totalArray,text,asin, overall,summary)
    print q
    #t=reduce_dimension(p)
    #write_to_json(totalArray,text, overall, t)
    (w, X)=to_vector(model, q, True)   #need to delete the whole row if the word is removed. Convert the first col of q
    #print w
    tt=add_sentiment(X[:,0])    #return sentiment. take an array and attach the sentiment label at the last col
    t=reduce_dimension(w)  #take the first col
    write_to_json(X, t, tt)






