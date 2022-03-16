import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
photo_flag = True  # True for photo realistic style, false for normal style transfer.
interplate_flag = False  # for mixed intermediate domain
# adahist --> efdm
# adavar --> adastd
# adarealhist --> hm

if photo_flag:
    content_file_list = os.listdir('./input2/content/')
    style_file_list = os.listdir('./input2/style/')
    adain_file = './output2_adain_adain/'
    adamean_file = './output2_ours_adamean_adamean/'
    adavar_file = './output2_ours_adavar_adavar/'
    adahist_file = './output2_ours_adahist_adahist/'
    adahist_mean = './output2_ours_adahist_adamean/'
    adahist_var = './output2_ours_adahist_adavar/'
    adahist_in = './output2_ours_adahist_adain/'
    adarealhist = './output2_ours_adarealhist_adarealhist/'
    ada_gaty = '../cmd_styletransfer/photo_output_gaty_no_centerresize/'
    ada_centermoment = '../cmd_styletransfer/photo_output_no_centerresize/'
    adain_inpterplate = './output2_interplate_adain_adain/'
    adasort_inpterplate = './output2_interplate_adahist_adahist/'
    adarealhist_inpterplate = './output2_interplate_adarealhist_adarealhist/'
    # output_file = './output2_concate_four/'
    output_file = './output2_concate/'
    # output_file = './output2_interplate_all/'
else:
    content_file_list = os.listdir('./input/content/')
    style_file_list = os.listdir('./input/style/')
    adain_file = './output_adain_adain/'
    adamean_file = './output_ours_adamean_adamean/'
    adavar_file = './output_ours_adavar_adavar/'
    adahist_file = './output_ours_adahist_adahist/'
    adahist_mean = './output_ours_adahist_adamean/'
    adahist_var = './output_ours_adahist_adavar/'
    adahist_in = './output_ours_adahist_adain/'
    adarealhist = './output_adarealhist_adarealhist/'
    ada_gaty = '../cmd_styletransfer/style_output_gaty_no_centerresize/'
    ada_centermoment = '../cmd_styletransfer/style_output_no_centerresize/'
    adain_inpterplate = './output_interplate_adain_adain/'
    adasort_inpterplate = './output_interplate_adahist_adahist/'
    adarealhist_inpterplate = './output_interplate_adarealhist_adarealhist/'
    # output_file = './output_concate_four/'
    output_file = './output_concate/'
    # output_file = './output_interplate_all/'



def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    # if crop:
    transform_list.append(transforms.CenterCrop(size))
    # transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform
resize = test_transform(512, True)


# interplate_four_style_file = './inter_four_style_adahist_adahist/'
# style1 = 'input/style/picasso_self_portrait.jpg'
# style2 = 'input/style/sketch.png'
# style3 = 'input/style/the_resevoir_at_poitiers.jpg'
# style4 = 'input/style/woman_in_peasant_dress.jpg'
# fs = []
# for i in range(35):
#     yu = i / 7
#     if i % 7 == 0:
#         if yu > 3:
#             fs.append(resize(Image.open((style3))))
#         else:
#             fs.append(resize(Image.open((style1))))
#     elif (i+1) % 7 == 0:
#         if yu > 3:
#             fs.append(resize(Image.open((style4))))
#         else:
#             fs.append(resize(Image.open((style2))))
#     else:
#         image_name = interplate_four_style_file + 'avril_interpolate_' + str(i-int(yu)*2) + '_.jpg'
#         fs.append(Image.open((image_name)))
# ncol = 7
# nrow = 5
# x, y = fs[0].size
#
# cvs = Image.new('RGB', (x * ncol, y * nrow))
# for i in range(len(fs)):
#     px, py = x * (i % ncol), y * int(i / ncol)
#     cvs.paste(fs[i], (px, py))
# transferred_name = 'avril_interpolate_four.jpg'
# cvs.save(interplate_four_style_file + transferred_name)




for content_file in content_file_list:
    if photo_flag:
        content_file_item = './input2/content/' + content_file
    else:
        content_file_item = './input/content/' + content_file
    content_im = Image.open(content_file_item, 'r')
    content_im = resize(content_im)
    for style_file in style_file_list:
        if photo_flag:
            style_file_item = './input2/style/' + style_file
        else:
            style_file_item = './input/style/' + style_file
        style_im = Image.open(style_file_item, 'r')
        style_im = resize(style_im)
        if not photo_flag or content_file.split('.')[0][2:] == style_file.split('.')[0][3:]:
            transferred_name = content_file.split('.')[0] + '_stylized_' + style_file.split('.')[0] + '.jpg'
            adain_image = adain_file + transferred_name
            adamean_image = adamean_file + transferred_name
            adavar_image = adavar_file + transferred_name
            adahist_image = adahist_file + transferred_name
            adahist_in_image = adahist_in + transferred_name
            adahist_var_image = adahist_var + transferred_name
            adahist_mean_iamge = adahist_mean + transferred_name
            adarealhist_mean_image = adarealhist + transferred_name
            ada_gaty_image = ada_gaty + transferred_name
            ada_centermoment_image = ada_centermoment + transferred_name
            fs = []


            ### for interplate
            if interplate_flag:
                alpha_list = [0.0, 0.25, 0.5, 0.75, 1.0]
                # fs.append(content_im)
                # for alpha in alpha_list:
                #     transferred_name_element = content_file.split('.')[0] + '_stylized_' + style_file.split('.')[0] + str(alpha) + '.jpg'
                #     adain_image = adain_inpterplate + transferred_name_element
                #     fs.append(Image.open(adain_image, 'r'))
                # fs.append(style_im)
                #
                # fs.append(content_im)
                # for alpha in alpha_list:
                #     transferred_name_element = content_file.split('.')[0] + '_stylized_' + style_file.split('.')[0] + str(alpha) + '.jpg'
                #     adain_image = adarealhist_inpterplate + transferred_name_element
                #     fs.append(Image.open(adain_image, 'r'))
                # fs.append(style_im)

                # fs.append(content_im)
                for alpha in alpha_list:
                    transferred_name_element = content_file.split('.')[0] + '_stylized_' + style_file.split('.')[0] + str(alpha) + '.jpg'
                    adain_image = adasort_inpterplate + transferred_name_element
                    fs.append(Image.open(adain_image, 'r'))
                fs.append(style_im)

                ncol = 6
                nrow = 1
                transferred_name = content_file.split('.')[0] + '_stylized_' + style_file.split('.')[0] + 'in_hist_sort' + '.jpg'
            else:
                # pass
                fs.append(content_im)
                fs.append(style_im)
                # fs.append(resize(Image.open((ada_gaty_image), 'r')))
                # fs.append(resize(Image.open((ada_centermoment_image), 'r')))
                # fs.append(Image.open(adain_image, 'r'))
                # fs.append(Image.open(adarealhist_mean_image, 'r'))
                # fs.append(Image.open(adahist_image, 'r'))

                # fs.append(Image.open(adarealhist_mean_image, 'r'))
                # fs.append(Image.open(adain_image, 'r'))
                # fs.append(Image.open(adahist_mean_iamge, 'r'))
                # fs.append(Image.open(adahist_var_image, 'r'))
                # fs.append(Image.open(adahist_in_image, 'r'))
                # fs.append(Image.open(adahist_image, 'r'))
                fs.append(Image.open(adamean_image, 'r'))
                fs.append(Image.open(adavar_image, 'r'))
                fs.append(Image.open(adain_image, 'r'))
                fs.append(Image.open(adarealhist_mean_image, 'r'))
                fs.append(Image.open(adahist_image, 'r'))
                ncol = 7
                nrow = 1
            x,y = fs[0].size

            cvs = Image.new('RGB',(x*ncol,y*nrow))
            for i in range(len(fs)):
                px, py = x*(i%ncol), y*int(i/ncol)
                cvs.paste(fs[i],(px,py))
            cvs.save(output_file + transferred_name)