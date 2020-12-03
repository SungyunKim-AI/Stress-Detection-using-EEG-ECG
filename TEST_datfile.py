import _pickle as cPickle

with open('s01.dat', 'rb') as f:
     x = cPickle.load(f, encoding='latin1')

label = x['labels']
data = x['data']

print(label.shape)
print(data.shape)