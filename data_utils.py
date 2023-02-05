
import os
import cv2
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader

class ObjDetectDataset(Dataset):
    """
    Dataset for the object detection task. Reads directly from files. 
    Idea is to use less memory - may be able to plug in with pre-trained resnet model.
    """

    def __init__(self, filenames, data_dir, label_encoder, bbox_format = 'coco', transforms = None):
        self.filenames = filenames
        self.data_dir = data_dir
        self.label_encoder = label_encoder
        self.bbox_format = bbox_format
        self.transforms = transforms

    def __getitem__(self, index):

        file_to_do = self.filenames[index]

        # read image as numpy file for transforms
        image = getImg(file_to_do +'.jpg', self.data_dir)

        
        # read xml with bounding box and labels
        bl = getBBox(file_to_do + '.xml', self.data_dir, format = self.bbox_format)
        #https://stackoverflow.com/questions/19339/transpose-unzip-function-inverse-of-zip
        labels, bboxes = list(zip(*bl))
        labels = self.label_encoder.transform(labels)

        # apply transforms using albumentations library
        if self.transforms:
            transformed = self.transforms(image = image, bboxes = bboxes, class_labels = labels)
            image = transformed['image']
            bboxes = torch.Tensor(transformed['bboxes'])
            labels = torch.Tensor(transformed['class_labels'])


        return (image, labels, bboxes)

    def __len__(self):
        return len(self.filenames)

def getImg(file, data_dir):
    """Read in an image file"""
    return cv2.imread(os.path.join(data_dir, file))

def getBBox(file, data_dir, format = 'coco'):
    """
    Extract bbox and label from our xml files.
    Converts to coco bbox format by default
    """
    tree = ET.parse(os.path.join(data_dir, file))
    objects = tree.getroot().findall('object')

    if format == 'pascal_voc':
    # use pascal_voc format - [x_min, y_min, x_max, y_max]
        bbox = [(
                o.find('name').text.strip(' \\'), # label
                        [
                        int(o.find('bndbox').find('xmin').text),
                        int(o.find('bndbox').find('ymin').text),
                        int(o.find('bndbox').find('xmax').text),
                        int(o.find('bndbox').find('ymax').text)
                        ] 
        ) for o in objects]
    # use coco format - [x_min, y_min, width, height]
    elif format == 'coco':
            bbox = [(
                o.find('name').text.strip(' \\'), # label
                        [
                        int(o.find('bndbox').find('xmin').text),
                        int(o.find('bndbox').find('ymin').text),
                        # width
                        int(o.find('bndbox').find('xmax').text) - int(o.find('bndbox').find('xmin').text),
                        # height
                        int(o.find('bndbox').find('ymax').text) - int(o.find('bndbox').find('ymin').text)
                        ] 
        ) for o in objects]

    return bbox

def getData(files, data_dir):

    """Imports all data in a raw dictionary format"""

    out = {}
    for file in files:
        name = file.split('.')[0]
        if name not in out.keys():
            out[name] = {'bbox':None, 'img':[]}
        if '.xml' in file:
            bb = getBBox(file, data_dir)
            out[name]['bbox'] = bb
        elif '.jpg' in file:
            img = getImg(file, data_dir)
            out[name]['img'] = img
    return out
    
