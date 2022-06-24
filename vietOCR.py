from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

from PIL import Image
import numpy as np
from typing import List, Iterator


config = Cfg.load_config_from_name('vgg_seq2seq')

# config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
# config['weights'] = 'https://drive.google.com/uc?id=1nTKlEog9YFK74kPyX0qLwCWi60_YHHk4'
config['weights'] = 'weights/transformerocr.pth'
config['cnn']['pretrained']=False
config['device'] = 'cpu'
config['predictor']['beamsearch']=False
detector = Predictor(config)


def img2word(img:np.ndarray, return_prob=False) -> str:
    #! assert len(image.shape) < 3, "have to convert to gray image"
    # image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img = Image.fromarray(img, mode="L")
    return detector.predict(img, return_prob)
    

def imgs2words(imgs:Iterator[np.ndarray], return_prob=False) -> List[str]:
    imgs = [Image.fromarray(img, mode="L") for img in imgs]
    return detector.predict_batch(imgs, return_prob)


if __name__ == '__main__':
    from time import time    
    
    img1 = 'dataset/vietOCR/13221_16.jpg'
    img2 = 'dataset/vietOCR/036200006617.jpeg'
    # img = Image.open(img)
    
    # start = time()
    # s = detector.predict(img)
    # end = time()
    # print('runtime: ', end-start)
    # print(s)
    
    
    # import cv2
    # imgs = [cv2.imread(img1, 0), cv2.imread(img2, 0)]
    # ss = imgs2words(imgs)
    # for s in ss:
    #     print(s)