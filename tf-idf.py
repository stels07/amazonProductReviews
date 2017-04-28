import json
from pprint import pprint

with open('review_all_montreal.json') as data_file:
    data = json.load(data_file)   #data is a dict

#pprint (data)
index=0
keys=data.keys()

#+++
business_data=[]
with open('business_info_all.json') as json_data:
    d = json.load(json_data)
    for dd in d:
        attr = {}
        for key, value in dd.iteritems():
            #print key, value
            attr.update({key:value})
        business_data.append(attr)

#pprint (business_data[0:10])
data_list=[] # to store business info, find by business_id

for key in keys:
    for i in business_data:
        if(i.get("business_id")==key):
            #print i.get("business_id")
            data_list.append([i.get("categories"),i.get('stars'), i.get('review_count'), i.get('latitude'), i.get('longitude')])

import numpy as np
#print np.array(data_list)[:,1]  #[4 4 4 ..., 3.5 3.5 3.5]
print np.array(data_list)[:,1].shape  #(563,)
#print np.array(data_list)[:,1]
#++++++++

#print (len(keys) )  #563 businesses
count=0
data_sample=[]
for d in data.get(keys[index]).get("reviews"):
    #pprint (data.get(keys[index]).get("reviews").get(d))
    review_for_star=data.get(keys[index]).get("reviews").get(d)
    for p in review_for_star:
        data_sample.append(p.get("text"))
    count+=len(data.get(keys[index]).get("reviews").get(d))
print (count )   # output review count of a certain resturant
print(len(data_sample))


all_business=[]
for index in range(0,len(keys)):
    # str is the string for all reviews for a business
    str=""
    for d in data.get(keys[index]).get("reviews"):
        review_for_star = data.get(keys[index]).get("reviews").get(d)
        for p in review_for_star:
            str=str+p.get("text")
    all_business.append(str)
#print all_business[0]

from sklearn.feature_extraction.text import CountVectorizer



docs=all_business

# train_set = ("The sky is blue.", "The sun is bright.")
# docs = ("The sun in the sky is bright.",
# "We can see the shining sun, the bright sun.")


count_vectorizer = CountVectorizer(stop_words='english')
count_vectorizer.fit_transform(docs)

print "Vocabulary:", count_vectorizer.vocabulary_
print "size of terms",  len(count_vectorizer.vocabulary_)

freq_term_matrix = count_vectorizer.transform(docs)
#print freq_term_matrix.todense()

from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer(norm="l2")
tfidf.fit(freq_term_matrix)
#print "IDF:", tfidf.idf_

tf_idf_matrix = tfidf.transform(freq_term_matrix)
print len(tf_idf_matrix.todense())

#print tf_idf_matrix.todense()[1,2]
#print type(count_vectorizer.vocabulary_.keys()[0])
#print count_vectorizer.vocabulary_.keys()[0].encode('ascii','replace')

#+++++++++++++++++++numpy to json, too slow
# results=[]
# for index in range(0, len(keys)-550):
#     result = {}
#     result["business_id"]=keys[index]
#     for i in range(0,len(count_vectorizer.vocabulary_.keys())-35800):
#         result[count_vectorizer.vocabulary_.keys()[i].encode('ascii','replace')]=tf_idf_matrix.todense()[index,i]
#     results.append(result)
#
# print "the first key is ",keys[0]
# print results[0]
#
# with open('result.json', 'w') as fp:
#     json.dumps(result,fp)


# a = tf_idf_matrix.todense()
# numpy.savetxt("foo.csv", a, delimiter=",")

x=tf_idf_matrix.todense()
X=np.array(x)

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn import manifold
from matplotlib import offsetbox


from time import time
t0 = time()
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
trans_data = tsne.fit_transform(X).T
t1 = time()
print("t-SNE: %.2g sec" % (t1 - t0))

# a = trans_data
# np.savetxt("foo.csv", a, delimiter=",")

# fig = plt.figure(figsize=(15, 8))
#
# colors=np.array(data_list)[:,1]*2
# colors=colors.astype(int)
# #print colors
# #colors=X[:,4]
#
# ax = fig.add_subplot(1,1,1)
# plt.scatter(trans_data[0], trans_data[1], c=colors, cmap=plt.cm.rainbow)
# #plt.scatter(trans_data[0], trans_data[1])
# plt.title("t-SNE (%.2g sec)" % (t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
# plt.axis('tight')


# plt.show()

print "trans_data dim is ", trans_data.shape

results=[]
for index in range(0, len(keys)):
    result = {}
    result["business_id"]=keys[index]
    result["x"]=trans_data[0][index]
    result["y"]=trans_data[1][index]
    result["cat"]=data_list[index][0]
    result["stars"]=data_list[index][1]
    result["review_count"]=data_list[index][2]
    results.append(result)

#data_list.append([i.get("categories"),i.get('stars'), i.get('review_count'), i.get('latitude'), i.get('longitude')])
print "the first key is ",keys[0]
print results[0]

import io
with open('result.json', 'w') as fp:
    json.dump(results,fp, ensure_ascii=False, separators=(',', ':'))

