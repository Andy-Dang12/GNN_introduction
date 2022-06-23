from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

from PIL import Image
import numpy as np
import cv2

config = Cfg.load_config_from_name('vgg_seq2seq')

# config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
# config['weights'] = 'https://drive.google.com/uc?id=1nTKlEog9YFK74kPyX0qLwCWi60_YHHk4'
config['weights'] = 'weights/transformerocr.pth'
config['cnn']['pretrained']=False
config['device'] = 'cpu'
config['predictor']['beamsearch']=False
detector = Predictor(config)


def image2word(image:np.ndarray) -> str:
    #! assert len(image.shape) < 3, "have to convert to gray image"
    # image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = Image.fromarray(image, mode="L")
    return detector.predict(image)
    
    
if __name__ == '__main__':
    from time import time
    img = 'dataset/vietOCR/13221_16.jpg'
    img = Image.open(img)
    
    start = time()
    s = detector.predict(img)
    end = time()
    print('runtime: ', end-start)
    print(s)
