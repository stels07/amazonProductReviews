import gzip
import json
from urllib import urlopen

# def parse(path):
#   g = gzip.open(path, 'r')
#   for l in g:
#     yield json.dumps(eval(l))
#
# f = open("output.strict", 'w')
# for l in parse("reviews_Books_5.json.gz"):
#   f.write(l + '\n')

# import ijson
#
# f = urlopen('output.strict')
# objects = ijson.items(f, '')
# print objects
# for o in objects:
#     print o
# cities = (o for o in objects if o['type'] == 'city')
# for city in cities:
#     do_something_with(city)

with open('select.json') as data_file:
    content = data_file.readlines()
    print content[1]
    content = [x+"," for x in content]
print content[1]
with open('select_1.json', 'w') as f:
     f.writelines("%s\n" % l for l in content)


# with open('review_for_039X.json') as data_file:
#     content = json.loads(data_file)
# from nltk import tokenize
# import numpy as np
# z = json.loads(open("Electronics_5.json", "r").read())
# # print z[0].get("reviewText")
# # nn=tokenize.sent_tokenize(z[0].get("reviewText"))
# # print nn
# # nnn=tokenize.sent_tokenize(z[1].get("reviewText"))
# # print nnn
# # nn=np.append(nn,nnn)
# # print nn

# # print len(zz)
# data=[]
# n=np.asarray(data)
# for i in range(0,len(z)):
#     nn=tokenize.sent_tokenize(z[i].get("reviewText"))
#     #print nn
#     n=np.append(n,nn)



# print n.shape
# print n[0:10]

