import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image,ImageDraw

class Fstrcnn:
    def __init__(self, img):
        model = fasterrcnn_resnet50_fpn(pretrained=True)
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

    def result_draw(self):
        boxes = self.predict()[0]['boxes'].cpu().numpy()
        labels = self.predict()[0]['labels'].cpu().numpy()
        scores = self.predict()[0]['scores'].cpu().numpy()
        draw = ImageDraw.Draw(self.img)

        for box, label, score in zip(boxes, labels, scores):
            if score > 0.5:
                q1 = (box[0], box[1])
                q2 = (box[2], box[3])
                draw.rectangle([q1, q2], outline='red', width=2)
                print(str(label))
                draw.text((box[0], box[1] - 10), str(label), fill='red')

        self.img.show()


if __name__ == '__main__':
    img_path = 'fasterrcnn简单版/street.jpg'
    model = Fstrcnn(img_path)
    model.result_draw()









