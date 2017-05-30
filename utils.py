import numpy as np
# To run background with matplotlib
# use these two lines
import matplotlib as mpl
mpl.use('Agg') 
import matplotlib.pyplot as plt

def plot_feats(actual_feats, generated_feats, epoch, index, checkpoint_dir):
    idx = np.random.randint(actual_feats.shape[0])
    actual_feat = actual_feats[idx]
    generated_feat = generated_feats[idx]
    plt.figure()
    plt.plot(actual_feat, 'r')
    plt.plot(generated_feat, 'b')
    plt.savefig('{}/figures/pulses_epoch{}_index{}.png'.format(checkpoint_dir, epoch, index))
    plt.close()

def read_binary_file(file, dim=1):
    f = open(file, 'rb')
    data = np.fromfile(f, dtype=np.float32)
    assert data.shape[0] % dim == 0.
    data = data.reshape(-1, dim)
    return data, data.shape[0]
