######################################################
# Original implementation by KinWaiCheuk: https://github.com/KinWaiCheuk/Triplet-net-keras
######################################################


from sklearn.manifold import TSNE
import numpy as np
import matplotlib.patheffects as PathEffects
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf


def scatter(outdir, x, labels, subtitle=None):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40,
                    c=palette[labels.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    if subtitle != None:
        plt.suptitle(subtitle)

    plt.savefig(outdir + "/" + subtitle)


def tsne_plot(outdir, name, x_train, x_test, y_train, y_test):
    tsne = TSNE()
    train_tsne_embeds = tsne.fit_transform(x_train[:512])
    scatter(outdir, train_tsne_embeds, y_train[:512], "Samples from Train Data, {}".format(name))

    eval_tsne_embeds = tsne.fit_transform(x_test[:512])
    scatter(outdir, eval_tsne_embeds, y_test[:512], "Samples from Test Data, {}".format(name))


