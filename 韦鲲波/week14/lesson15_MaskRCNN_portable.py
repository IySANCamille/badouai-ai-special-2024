import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from PIL import Image,ImageDraw
import cv2
import numpy as np

class MRCNN:
    def __init__(self, img):
        model = maskrcnn_resnet50_fpn(pretrained=True)
        model = model.eval()
        self.model = model.to('cuda')
        self.img = Image.open(img).convert('RGB')


    def img_preprocess(self):
        preprocessing = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        output = preprocessing(self.img).unsqueeze(0)

        return output

    def predict(self):
        data = self.img_preprocess()
        data = data.to('cuda')

        with torch.no_grad():
            prediction = self.model(data)

        return prediction


    def result_draw(self, img, prediction):
        image = cv2.imread(img)
        instance_colors = {}
        for pred in prediction:
            masks = pred['masks'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            for i, (mask, label, score) in enumerate(zip(masks, labels, scores)):
                if score > 0.5:
                    mask = mask[0]
                    mask = (mask > 0.5).astype(np.uint8)
                    if i not in instance_colors:
                        instance_colors[i] = (
                        np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
                    color = instance_colors[i]
                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(image, contours, -1, color, 2)

        cv2.imshow('Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    img_path = 'street.jpg'
    model = MRCNN(img_path)
    prediction = model.predict()
    model.result_draw(img_path, prediction)

