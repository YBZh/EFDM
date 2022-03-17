from torch.nn import functional as F
import torch
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.modeling.ops import random_efdmix, crossdomain_efdmix
import numpy as np
import os.path as osp
# import ipdb

@TRAINER_REGISTRY.register()
class Vanilla2(TrainerX):
    """Vanilla baseline.

    Slightly modified for mixstyle.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        mix = cfg.TRAINER.VANILLA2.MIX

        if mix == 'random':
            self.model.apply(random_efdmix)
            print('EFDMix: random mixing')

        elif mix == 'crossdomain':
            self.model.apply(crossdomain_efdmix)
            print('EFDMix: cross-domain mixing')

        else:
            raise NotImplementedError

    def forward_backward(self, batch):
        input, label = self.parse_batch_train(batch)

        output = self.model(input)
        loss = F.cross_entropy(output, label)
        self.model_backward_and_update(loss)

        loss_summary = {
            'loss': loss.item(),
            'acc': compute_accuracy(output, label)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    @torch.no_grad()
    def vis(self):
        self.set_model_mode('eval')
        output_dir = self.cfg.OUTPUT_DIR
        source_domains = self.cfg.DATASET.SOURCE_DOMAINS
        print('Source domains:', source_domains)

        out_embed = []
        out_embed2 = []
        out_domain = []
        out_label = []

        out_embed_mean = []
        out_embed_var = []
        out_embed_third = []
        out_embed_fourth = []
        out_embed_infinity = []

        split = self.cfg.TEST.SPLIT
        data_loader = self.val_loader if split == 'val' else self.test_loader

        print('Extracting style features')

        for batch_idx, batch in enumerate(data_loader):
            input = batch['img'].to(self.device)
            label = batch['label']
            domain = batch['domain']
            impath = batch['impath']

            # model should directly output features or style statistics
            # raise NotImplementedError
            output = self.model(input) ## feature: N*C*W*H

            ## 1. mean, variance
            mu = output.mean(dim=[2, 3])
            var = output.var(dim=[2, 3])
            sig = (var + 1e-8).sqrt()
            mu_var = torch.cat((mu, var), dim=1)
            mu_var = mu_var.cpu().numpy()

            out_embed_mean.append(mu.cpu().clone().numpy())
            out_embed_var.append(sig.cpu().clone().numpy())

            ## 0-1 normalized feature.
            mu = output.mean(dim=[2, 3], keepdim=True)
            var = output.var(dim=[2, 3], keepdim=True)
            sig = (var + 1e-8).sqrt()
            mu, sig = mu.detach(), sig.detach()
            x = (output - mu) / sig  ## N*C*W*H


            B, C, W, H = x.size(0), x.size(1), x.size(2), x.size(3)
            x_view = x.view(B, C, -1)
            value_x = x_view
            # value_x, index_x = torch.sort(x_view) ## B*C*D
            value_x = value_x[:, 1:2, :] ## only one channel.
            value_x = value_x.view(B, -1).cpu().numpy()
            # print(value_x)

            third_order = torch.pow(x_view, 3).mean(-1).cpu().numpy()
            fourth_order = torch.pow(x_view, 4).mean(-1).cpu().numpy()
            infinity_order = torch.max(x_view, -1)[0].cpu().numpy()

            out_embed_infinity.append(infinity_order)
            out_embed_third.append(third_order)
            out_embed_fourth.append(fourth_order)

            # output = output.cpu().numpy()
            # out_embed.append(output)
            out_embed.append(mu_var)
            out_embed2.append(value_x)
            out_domain.append(domain.numpy())
            out_label.append(label.numpy()) # CLASS LABEL

            print('processed batch-{}'.format(batch_idx + 1))

        out_embed = np.concatenate(out_embed, axis=0)
        out_embed2 = np.concatenate(out_embed2, axis=0)
        out_embed_mean = np.concatenate(out_embed_mean, axis=0)
        out_embed_var = np.concatenate(out_embed_var, axis=0)
        out_embed_third = np.concatenate(out_embed_third, axis=0)
        out_embed_fourth = np.concatenate(out_embed_fourth, axis=0)
        out_embed_infinity = np.concatenate(out_embed_infinity, axis=0)
        out_domain = np.concatenate(out_domain, axis=0)
        out_label = np.concatenate(out_label, axis=0)
        print('shape of feature matrix:', out_embed.shape)
        out = {
            'embed': out_embed,
            'domain': out_domain,
            'dnames': source_domains,
            'label': out_label
        }
        out_path = osp.join(output_dir, 'embed.pt')
        torch.save(out, out_path)
        print('File saved to "{}"'.format(out_path))


        print('shape of feature matrix:', out_embed2.shape)
        out = {
            'embed': out_embed2,
            'domain': out_domain,
            'dnames': source_domains,
            'label': out_label
        }
        out_path = osp.join(output_dir, 'embed_hist.pt')
        torch.save(out, out_path)
        print('File saved to "{}"'.format(out_path))


        print('shape of feature matrix:', out_embed_mean.shape)
        out = {
            'embed': out_embed_mean,
            'domain': out_domain,
            'dnames': source_domains,
            'label': out_label
        }
        out_path = osp.join(output_dir, 'embed_mean.pt')
        torch.save(out, out_path)
        print('File saved to "{}"'.format(out_path))


        print('shape of feature matrix:', out_embed_var.shape)
        out = {
            'embed': out_embed_var,
            'domain': out_domain,
            'dnames': source_domains,
            'label': out_label
        }
        out_path = osp.join(output_dir, 'embed_var.pt')
        torch.save(out, out_path)
        print('File saved to "{}"'.format(out_path))





        print('shape of feature matrix:', out_embed_third.shape)
        out = {
            'embed': out_embed_third,
            'domain': out_domain,
            'dnames': source_domains,
            'label': out_label
        }
        out_path = osp.join(output_dir, 'embed_third.pt')
        torch.save(out, out_path)
        print('File saved to "{}"'.format(out_path))


        print('shape of feature matrix:', out_embed_fourth.shape)
        out = {
            'embed': out_embed_fourth,
            'domain': out_domain,
            'dnames': source_domains,
            'label': out_label
        }
        out_path = osp.join(output_dir, 'embed_fourth.pt')
        torch.save(out, out_path)
        print('File saved to "{}"'.format(out_path))


        print('shape of feature matrix:', out_embed_infinity.shape)
        out = {
            'embed': out_embed_infinity,
            'domain': out_domain,
            'dnames': source_domains,
            'label': out_label
        }
        out_path = osp.join(output_dir, 'embed_infinity.pt')
        torch.save(out, out_path)
        print('File saved to "{}"'.format(out_path))

