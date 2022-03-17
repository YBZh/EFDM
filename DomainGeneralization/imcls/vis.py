import argparse
import torch
import os.path as osp
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


def normalize(feature):
    norm = np.sqrt((feature**2).sum(1, keepdims=True))
    return feature / (norm + 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='/home/yabin/syn_project/mixstyle-release-master/imcls/output_vis/pacs/Vanilla2/resnet18/random/art_painting/seed/embed_hist.pt', help='path to source file')
    parser.add_argument('--dst', type=str, default='', help='destination directory')
    parser.add_argument('--method', type=str, default='none', help='tnse, pca or none')
    args = parser.parse_args()

    if not args.dst:
        args.dst = osp.dirname(args.src)

    print('Loading file from "{}"'.format(args.src))
    file = torch.load(args.src)

    embed = file['embed']
    domain = file['domain']
    dnames = file['dnames']
    cate = file['label']


    # embed = embed[cate==1] ## only the dog feature 1 is good
    # domain = domain[cate==1]

    #dim = embed.shape[1] // 2
    #embed = embed[:, dim:]

    #domain = file['label']
    #dnames = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']

    nd_src = len(dnames)
    # embed = normalize(embed)

    print('Loaded features with shape {}'.format(embed.shape))

    embed2d_path = osp.join(args.dst, 'embed2d_' + args.method + '.pt')

    # if osp.exists(embed2d_path):
    #     embed2d = torch.load(embed2d_path)
    #     print('Loaded embed2d from "{}"'.format(embed2d_path))
    #
    # else:
    if args.method == 'tsne':
        print('Dimension reduction with t-SNE (dim=2) ...')
        tsne = TSNE(
            n_components=2, metric='euclidean', verbose=1,
            perplexity=50, n_iter=1000, learning_rate=200.
        )
        embed2d = tsne.fit_transform(embed)

        torch.save(embed2d, embed2d_path)
        print('Saved embed2d to "{}"'.format(embed2d_path))

    elif args.method == 'pca':
        print('Dimension reduction with PCA (dim=2) ...')
        pca = PCA(n_components=2)
        embed2d = pca.fit_transform(embed)

        torch.save(embed2d, embed2d_path)
        print('Saved embed2d to "{}"'.format(embed2d_path))

    elif args.method == 'none':
        # the original embedding is 2-D
        embed2d = embed

    avai_domains = list(set(domain.tolist()))
    avai_domains.sort()

    print('Plotting ...')

    SIZE = 3
    COLORS = ['C0', 'C9', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    LEGEND_MS = 3

    # fig, ax = plt.subplots()

    for d in avai_domains:
        d = int(d)
        e = embed2d[domain == d]

        # cate_d = cate[domain == d]
        # e = e[cate_d==0]
        #
        # """
        # label = '$D_{}$'.format(str(d + 1))
        # if d < nd_src:
        #     label += ' ($\mathcal{S}$)'
        # else:
        #     label += ' ($\mathcal{N}$)'
        # """
        label = dnames[d]
        print(label)

        # ax.scatter(
        #     e[:, 0],
        #     e[:, 1],
        #     s=SIZE,
        #     c=COLORS[d],
        #     edgecolors='none',
        #     label=label,
        #     alpha=1,
        #     rasterized=False
        # )

        ##################### plot the feature histogram of each domain
        e = embed[domain == d]  ## N * d
        ee = e.ravel()
        fig, ax = plt.subplots()
        plt.hist(ee, color=，bins=100)
        plt.xlabel('Values', fontsize=16)
        plt.ylabel('Numbers', fontsize=16)
        ax.set_xlim(-3, 6)
        ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
        figname = label + '_feature_dist.pdf'
        plt.savefig(osp.join(args.dst, figname), bbox_inches='tight')
        plt.close()

    #ax.legend(loc='upper left', fontsize=10, markerscale=LEGEND_MS)
    # ax.legend(fontsize=14, markerscale=LEGEND_MS)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_xlim(-3, 10)
    # #ax.set_ylim(-LIM, LIM)
    #
    # figname = 'embed_mean_var_concat.pdf'
    # fig.savefig(osp.join(args.dst, figname), bbox_inches='tight')
    # plt.close()


if __name__ == '__main__':
    main()
