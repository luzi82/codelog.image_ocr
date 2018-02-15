from captcha.image import ImageCaptcha
import numpy as np
from scipy import ndimage
import os

FONT_LIST = [
    os.path.join('resource_set','font_set','Roboto','Roboto-Regular.ttf'),
]

_image_captcha_dict = {}
def get_image_captcha(w, h):
    global FONT_LIST
    key = (w,h)
    if key not in _image_captcha_dict:
        _image_captcha_dict[key] = ImageCaptcha(width=w, height=h, fonts=FONT_LIST)
    return _image_captcha_dict[key]

# paints the string in a random location the bounding box
# also uses a random font, a slight random rotation,
# and a random amount of speckle noise

def paint_text(text, w, h):
    image_captcha = get_image_captcha(w,h)
    img = image_captcha.generate_image(text)
    img = np.expand_dims(img, 0)

    return img

if __name__ == '__main__':
    import pylab
    X = paint_text('0123', 128, 60)
    print('shape={}'.format(X.shape))
    print('min={}, max={}'.format(X.min(),X.max()))
    X = X.reshape((60,128,3))
    print('shape={}'.format(X.shape))
    pylab.imshow(X)
    #pylab.colorbar()
    pylab.show()
