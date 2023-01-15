import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import time
import net
from function import adaptive_instance_normalization, coral
from function import adaptive_mean_normalization
from function import adaptive_std_normalization
from function import exact_feature_distribution_matching, histogram_matching

def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None, style_type='adain'):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        if style_type == 'adain':
            base_feat = adaptive_instance_normalization(content_f, style_f)
        elif style_type == 'adamean':
            base_feat = adaptive_mean_normalization(content_f, style_f)
        elif style_type == 'adastd':
            base_feat = adaptive_std_normalization(content_f, style_f)
        elif style_type == 'efdm':
            base_feat = exact_feature_distribution_matching(content_f, style_f)
        elif style_type == 'hm':
            feat = histogram_matching(content_f, style_f)
        else:
            raise NotImplementedError
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        if style_type == 'adain':
            feat = adaptive_instance_normalization(content_f, style_f)
        elif style_type == 'adamean':
            feat = adaptive_mean_normalization(content_f, style_f)
        elif style_type == 'adastd':
            feat = adaptive_std_normalization(content_f, style_f)
        elif style_type == 'efdm':
            feat = exact_feature_distribution_matching(content_f, style_f)
        elif style_type == 'hm':
            feat = histogram_matching(content_f, style_f)
        else:
            raise NotImplementedError
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str,
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder.pth')
parser.add_argument('--style_type', type=str, default='adain', help='adain | adamean | adastd | efdm')
parser.add_argument('--test_style_type', type=str, default='', help='adain | adamean | adastd | efdm')
# Additional options
parser.add_argument('--content_size', type=int, default=512,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=512,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')
parser.add_argument('--photo', action='store_true',
                    help='apply on the photo style transfer')
# Advanced options
parser.add_argument('--preserve_color', action='store_true',
                    help='If specified, preserve color of the content image')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')
parser.add_argument(
    '--style_interpolation_weights', type=str, default='',
    help='The weight for blending the style of multiple style images')

args = parser.parse_args()
if not args.test_style_type:
    args.test_style_type = args.style_type

print('Note: the style type: %s and the pre-trained model: %s should be consistent' % (args.style_type, args.decoder))
print('The test style type is:', args.test_style_type)

# do_interpolation = True  ### When this is set as True, we aim to generate output images with mixed styles; therefore, multiple style images should be inputed, for example:
###CUDA_VISIBLE_DEVICES=0 python test.py --content input/content/avril.jpg --style input/style/picasso_self_portrait.jpg,input/style/sketch.png,input/style/the_resevoir_at_poitiers.jpg,input/style/woman_in_peasant_dress.jpg \
#--style_type adahist --test_style_type adahist  --crop --decoder ./experiments_efdm/decoder_iter_160000.pth.tar --output  inter_four_style

do_interpolation = False   ## only one style image is required.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dir = Path(args.output + '_' + args.style_type + '_' + args.test_style_type)
output_dir.mkdir(exist_ok=True, parents=True)

# Either --content or --contentDir should be given.
assert (args.content or args.content_dir)
if args.content:
    content_paths = [Path(args.content)]
else:
    content_dir = Path(args.content_dir)
    content_paths = [f for f in content_dir.glob('*')]

# Either --style or --styleDir should be given.
assert (args.style or args.style_dir)
if args.style:
    style_paths = args.style.split(',')
    if len(style_paths) == 1:
        style_paths = [Path(args.style)]
    else:
        do_interpolation = True
        # assert (args.style_interpolation_weights != ''), \
        #     'Please specify interpolation weights'
        # weights = [int(i) for i in args.style_interpolation_weights.split(',')]
        # interpolation_weights = [w / sum(weights) for w in weights]
else:
    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]

decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

vgg.to(device)
decoder.to(device)

content_tf = test_transform(args.content_size, args.crop)
style_tf = test_transform(args.style_size, args.crop)

timer = []
for content_path in content_paths:
    if do_interpolation:
        # one content image, 4 style image
        style = torch.stack([style_tf(Image.open(str(p))) for p in style_paths])
        content = content_tf(Image.open(str(content_path))) \
            .unsqueeze(0).expand_as(style)
        style = style.to(device)
        content = content.to(device)
        list = []
        steps = [1, 0.75, 0.5, 0.25, 0]
        for i in steps:
            for j in steps:
                list.append([i*j, i*(1-j), (1-i)*j, (1-i)*(1-j)])
        count = 1
        for interpolation_weights in list:
            with torch.no_grad():
                output = style_transfer(vgg, decoder, content, style,
                                        args.alpha, interpolation_weights, style_type=args.test_style_type)
            output = output.cpu()
            output_name = output_dir / '{:s}_interpolate_{:s}_{:s}'.format(
                content_path.stem, str(count), args.save_ext)
            save_image(output, str(output_name))
            count+=1

        #### content & style trade-off.
        # alpha = [0.0, 0.25, 0.5, 0.75, 1.0]
        # for style_path in style_paths:
        #     content = content_tf(Image.open(str(content_path)))
        #     style = style_tf(Image.open(str(style_path)))
        #     if args.preserve_color:
        #         style = coral(style, content)
        #     style = style.to(device).unsqueeze(0)
        #     content = content.to(device).unsqueeze(0)
        #     ## replace the style image with Gaussian noise
        #     # style.normal_(0,1)
        #     # style = torch.rand(style.size()).to(device)
        #     ### for paired images.
        #     if args.photo:
        #         if content_path.stem[2:] == style_path.stem[3:]:
        #             for sample_alpha in alpha:
        #                 with torch.no_grad():
        #                     output = style_transfer(vgg, decoder, content, style,
        #                                             sample_alpha, style_type=args.test_style_type)
        #                 output = output.cpu()
        #                 output_name = output_dir / '{:s}_stylized_{:s}{:s}{:s}'.format(
        #                     content_path.stem, style_path.stem, str(sample_alpha), args.save_ext)
        #                 save_image(output, str(output_name))
        #     else:
        #         for sample_alpha in alpha:
        #             with torch.no_grad():
        #                 output = style_transfer(vgg, decoder, content, style,
        #                                         sample_alpha, style_type=args.test_style_type)
        #             output = output.cpu()
        #             output_name = output_dir / '{:s}_stylized_{:s}{:s}{:s}'.format(
        #                 content_path.stem, style_path.stem, str(sample_alpha), args.save_ext)
        #             save_image(output, str(output_name))
    else:  # process one content and one style
        for style_path in style_paths:
            content = content_tf(Image.open(str(content_path)))
            style = style_tf(Image.open(str(style_path)))
            if args.preserve_color:
                style = coral(style, content)
            style = style.to(device).unsqueeze(0)
            content = content.to(device).unsqueeze(0)
            ## replace the style image with Gaussian noise
            # style.normal_(0,1)
            # style = torch.rand(style.size()).to(device)
            ### for paired images.
            if args.photo:
                if content_path.stem[2:] == style_path.stem[3:]:
                    with torch.no_grad():
                        start_time = time.time()
                        output = style_transfer(vgg, decoder, content, style,
                                                args.alpha, style_type=args.test_style_type)
                        timer.append(time.time() - start_time)
                        print(timer)

                    output = output.cpu()
                    output_name = output_dir / '{:s}_stylized_{:s}{:s}'.format(
                        content_path.stem, style_path.stem, args.save_ext)
                    save_image(output, str(output_name))
            else:
                with torch.no_grad():
                    output = style_transfer(vgg, decoder, content, style,
                                            args.alpha, style_type=args.test_style_type)
                output = output.cpu()
                output_name = output_dir / '{:s}_stylized_{:s}{:s}'.format(
                    content_path.stem, style_path.stem, args.save_ext)
                save_image(output, str(output_name))
print(torch.FloatTensor(timer).mean())
