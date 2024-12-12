import PIL.Image as Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


# 加载图片
def MyLoader(path):
    return Image.open(path).convert('RGB')  # 将原始图片转化为对应的像素值


# 转换处理
transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # 转换成Tensor
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  # 归一化
])


def synthesize(path):
    img = MyLoader(path)  # 载入图片
    img = transform(img)  # 图像增强处理
    img = img.view(1, 3, 224, 224)  # 向四维转换（使用DataLoader可能更好 --- 比较懒）
    return img


# 训练使用
class MyDataset(Dataset):
    def __init__(self, data, transform, loder):
        self.data = data
        self.transform = transform
        self.loader = loder

    def __getitem__(self, item):
        img, label = self.data[item]
        img = self.loader(img)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)