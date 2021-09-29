import numpy as np
import matplotlib.pyplot as plt

ida = np.ndarray([])
hna = np.ndarray([])
waa = np.ndarray([])

# read in activations

with open('activations/waterbird/erm_lr1e3wd1e3/erm_lr1e3wd1e3/activations_id_at_epoch_30.npy', 'rb') as f:
    idas = np.load(f) # id (waterbirds on water background)
print('ID shape', idas.shape)

with open('activations/waterbird/erm_lr1e3wd1e3/erm_lr1e3wd1e3/activations_SVHN_at_epoch_30.npy', 'rb') as f:
    hnas = np.load(f) # house numbers
print('NSP shape', hnas.shape)

with open('activations/waterbird/erm_lr1e3wd1e3/erm_lr1e3wd1e3/activations_water_at_epoch_30.npy', 'rb') as f:
    waas = np.load(f) # spoood (water without bird)
print('SP shape', waas.shape)

# make average activation patterns

ida = idas.mean(axis=0)
hna = hnas.mean(axis=0)
waa = waas.mean(axis=0)

print('average activation shape', ida.shape)

n = ida.shape[0] # num elems

# get top activated neurons

ida10 = np.argsort(ida)[-10:]
hna10 = np.argsort(hna)[-10:]
waa10 = np.argsort(waa)[-10:]

print('\ntop ID    neurons:', ida10)
print('top SVHN  neurons:', hna10)
print('top water neurons:', waa10)

# out of n top neurons for each dataset, see how many are in common
print('neurons in common')

# levels = [10, 100, 1000, 10000, 12544]
levels = [10, 100, int(n/2), n]

for l in levels:
    idatop = np.argsort(ida)[-l:]
    hnatop = np.argsort(hna)[-l:]
    waatop = np.argsort(waa)[-l:]

    all3 = [e for e in waatop if (e in hnatop) and (e in idatop)] # neurons in all 3 sets

    print('\n', l, 'top neurons')
    print('\tall 3:        \t', len(all3)/l)
    print('\tSP  & ID:     \t', sum([(e in waatop) for e in idatop])/l)
    print('\tNSP & ID:     \t', sum([(e in hnatop) for e in idatop])/l)
    print('\tNSP & SP:     \t', sum([(e in hnatop) for e in waatop])/l)
    print('\tSP  & ID only:\t', sum([(e in waatop) for e in idatop if e not in all3])/l)
    print('\tNSP & ID only:\t', sum([(e in hnatop) for e in idatop if e not in all3])/l)
    print('\tNSP & SP only:\t', sum([(e in hnatop) for e in waatop if e not in all3])/l)

# cosine similarity

def cossim(a, b):
    return np.dot(a,b) / (a.shape[0]*b.shape[0])

print('\nsimilarity')
print('\tID and SP\t', cossim(ida, waa))
print('\tID and NSP\t', cossim(ida, hna))
print('\tNSP and SP\t', cossim(hna, waa))

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
plt.savefig('activations/plotSP.png') 
