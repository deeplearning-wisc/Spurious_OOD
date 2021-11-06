import sys
import numpy as np
import matplotlib.pyplot as plt

dataset = sys.argv[1]
name = sys.argv[2]
epoch = sys.argv[3]
spood = ''
if dataset == 'waterbird':
    spood = 'water'
if dataset == 'celebA':
    spood = 'celebA_ood'


ida = np.ndarray([])
hna = np.ndarray([])
waa = np.ndarray([])

d = ''

# read in activations

with open(f'experiments/{dataset}/{name}/activations/activations_id_at_epoch_{epoch}_e0123.npy', 'rb') as f:
    idas = np.load(f) # id (waterbirds on water background)
print('ID shape', idas.shape)

with open(f'experiments/{dataset}/{name}/activations/activations_SVHN_at_epoch_{epoch}.npy', 'rb') as f:
    hnas = np.load(f) # house numbers
# print('NSP shape', hnas.shape)

with open(f'experiments/{dataset}/{name}/activations/activations_{spood}_at_epoch_{epoch}.npy', 'rb') as f:
    waas = np.load(f) # spoood (water without bird)
# print('SP shape', waas.shape)

# make average activation patterns

ida = idas.mean(axis=0).reshape([512])
hna = hnas.mean(axis=0).reshape([512])
waa = waas.mean(axis=0).reshape([512])

s =''
s = s + 'average activation shape' + str(ida.shape) + '\n'

n = ida.shape[0] # num elems

# get top activated neurons

ida10 = np.argsort(ida)[-10:]
hna10 = np.argsort(hna)[-10:]
waa10 = np.argsort(waa)[-10:]

# out of n top neurons for each dataset, see how many are in common
s = s + 'neurons in common\n' 

# levels = [10, 100, 1000, 10000, 12544]
levels = [10, 100, int(n/2), n]

for l in levels:
    idatop = np.argsort(ida)[-l:]
    hnatop = np.argsort(hna)[-l:]
    waatop = np.argsort(waa)[-l:]

    all3 = [e for e in waatop if (e in hnatop) and (e in idatop)] # neurons in all 3 sets

    s = s + f'\n{l} top neurons\n'
    s = s + '\tall 3:        \t' + str(len(all3)/l) + '\n'
    s = s + '\tSP  & ID:     \t' + str(sum([(e in waatop) for e in idatop])/l) + '\n'
    s = s + '\tNSP & ID:     \t' + str(sum([(e in hnatop) for e in idatop])/l) + '\n'
    s = s + '\tNSP & SP:     \t' + str(sum([(e in hnatop) for e in waatop])/l) + '\n'
    s = s + '\tSP  & ID only:\t' + str(sum([(e in waatop) for e in idatop if e not in all3])/l) + '\n'
    s = s + '\tNSP & ID only:\t' + str(sum([(e in hnatop) for e in idatop if e not in all3])/l) + '\n'
    s = s + '\tNSP & SP only:\t' + str(sum([(e in hnatop) for e in waatop if e not in all3])/l) + '\n'

# cosine similarity

# def cossim(a, b):
#     return np.dot(a,b) / (a.shape[0]*b.shape[0])

# idawaasim = np.zeros((idas.shape[0],waas.shape[0]))
# idahnasim = np.zeros((idas.shape[0],hnas.shape[0]))
# waahnasim = np.zeros((waas.shape[0],hnas.shape[0]))

# for i in range(idas.shape[0]):
#     for j in range(waas.shape[0]):
#         idawaasim[i,j] = cossim(idas[i], waas[i])
#     for j in range(hnas.shape[0]):
#         idahnasim[i,j] = cossim(idas[i], hnas[i])
# for i in range(waas.shape[0]):
#     for j in range(hnas.shape[0]):
#         waahnasim[i,j] = cossim(waas[i], hnas[i])

# s = s + '\nsimilarity\n'
# s = s + '\tID and SP\t' + str(idawaasim.mean()) + '\n'
# s = s + '\tID and NSP\t' + str(idahnasim.mean()) + '\n'
# s = s + '\tNSP and SP\t' + str(waahnasim.mean()) + '\n'

print(s)


with open(f'experiments/{dataset}/{name}/contributions.txt', 'w') as f:
    f.write(s + '\n')

# PLOTTING

# sort all in order of ID size
idaOrder = np.argsort(ida)
idasort = ida[idaOrder]
hnasort = hna[idaOrder]
waasort = waa[idaOrder]

plt.plot(range(n), idasort, 'o', ms=1.5, label='ID')
plt.plot(range(n), hnasort, 'o', ms=1.5, label='NSP')
plt.plot(range(n), waasort, 'o', ms=1.5, label='SP')

plt.title('average contribution per unit to final resnet18 linear layer, by data type')
plt.ylabel('contribution')
plt.xlabel('unit')
plt.legend(loc="upper left", markerscale=5, fontsize=12)
plt.savefig(f'experiments/{dataset}/{name}/contributions_plot.png') 
