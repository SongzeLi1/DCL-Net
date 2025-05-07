import glob

stride = 512
filenames = glob.glob('/pubdata/lisongze/docimg/exam/docimg2jpeg/test/testorig_crop512stride{}/*.jpg'.format(stride)) + \
            glob.glob(
                '/pubdata/lisongze/docimg/exam/docimg2jpeg/test/test_images_crop512stride{}/*.jpg'.format(stride)) + \
            glob.glob('/pubdata/lisongze/docimg/exam/docimg2jpeg/test/PSMosaic_crop512stride{}/*.jpg'.format(stride)) + \
            glob.glob('/pubdata/lisongze/docimg/exam/docimg2jpeg/test/test_tamper_crop512stride{}/*.jpg'.format(stride))
filenames.sort()

image_name_list = glob.glob('/pubdata/lisongze/docimg/exam/docimg2jpeg/test/testorig/*.jpg') + \
                  glob.glob('/pubdata/lisongze/docimg/exam/docimg2jpeg/test/test_images/*.jpg') + \
                  glob.glob('/pubdata/lisongze/docimg/exam/docimg2jpeg/test/PSMosaic/*.jpg') + \
                  glob.glob('/pubdata/lisongze/docimg/exam/docimg2jpeg/test/test_tamper/*.jpg')
image_name_list.sort()

for image_name in image_name_list:
    if 'orig' in image_name or 'mosaic' in image_name:
        imggt = 0
    else:
        imggt = 1
    image_name_ = image_name.split('/')[-1]
    print(image_name_)
    filenames_ = [blockname.split('/')[-1] for blockname in filenames]
    # for blockname in filenames_:
    #     label = image_name_.split(".", -1)[0]
    #     eq = blockname[0:blockname[0:blockname.rfind('_')].rfind('_')]
    #     if image_name_.split(".", -1)[0] == blockname[0:blockname[0:blockname.rfind('_')].rfind('_')]:
    #         print(dict[blockname])
    preds = [int(dict[blockname]) for blockname in filenames_ if
             image_name_.split(".", -1)[0] == blockname[0:blockname[0:blockname.rfind('_')].rfind('_')]]
    print(preds)
    if 1 in preds:
        imgpred = 1
    else:
        imgpred = 0