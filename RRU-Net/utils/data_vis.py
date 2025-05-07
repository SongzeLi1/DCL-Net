import matplotlib.pyplot as plt


def plot_img_and_mask(img, mask):
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    a.set_title('Input image')
    plt.imshow(img)

    b = fig.add_subplot(1, 2, 2)
    b.set_title('Output mask')
    plt.imshow(mask)

    plt.show()

def plot_img_pmap_mask(img, pmap, mask):
    fig = plt.figure()
    a = fig.add_subplot(1, 3, 1)
    a.set_title('Input image')
    a.imshow(img)

    b = fig.add_subplot(1, 3, 2)
    b.set_title('Output pmap')
    b.imshow(pmap)

    c = fig.add_subplot(1, 3, 3)
    c.set_title('Output mask')
    c.imshow(mask)

    plt.show()
    # plt.savefig('temp.png')



