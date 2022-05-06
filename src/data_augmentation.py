import os
import csv
import glob
import torch
import random
import torchvision
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size, **kwargs):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.kwargs = kwargs
    
    def __resize__(self, img, new_size):
        if type(img) is torch.Tensor:
            img = torchvision.transforms.ToPILImage()(img)
            img = torchvision.transforms.functional.resize(img,
                                                           new_size,
                                                           **self.kwargs)
            img = torchvision.transforms.ToTensor()(img)
        
        else:
            img = torchvision.transforms.functional.resize(img,
                                                           new_size,
                                                           **self.kwargs)
        
        return img

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        h, w = image.shape[1:3]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = self.__resize__(image, (new_h, new_w))
        mask = self.__resize__(mask, (new_h, new_w))

        sample['image'], sample['mask'] = image, mask

        return sample


class RandomPad(object):
    """Randomly add padding to the image in a sample.

    Args:
        pad (int): pixels to pad around the image.
        
        probability (float): probability the transform is applied from 0 to 1.
    """

    def __init__(self, pad=10, probability=1.0, **kwargs):
        assert isinstance(pad, int)
        self.pad = pad
        assert isinstance(probability, float)
        assert probability <= 1.0
        self.probability = probability
        self.kwargs = kwargs
    
    def __pad__(self, img, padding):
        if type(img) is torch.Tensor:
            img = torchvision.transforms.ToPILImage()(img)
            img = torchvision.transforms.functional.pad(img, padding,
                                                        **self.kwargs)
            img = torchvision.transforms.ToTensor()(img)
        
        else:
            img = torchvision.transforms.functional.pad(img, padding,
                                                        **self.kwargs)
        
        return img

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        if random.random() < self.probability:
            image = self.__pad__(image, self.pad)
            mask = self.__pad__(mask, self.pad)

        sample['image'], sample['mask'] = image, mask

        return sample

    
class RandomNoise(object):
    """Randomly add noise to the image in a sample. This will only be applied
        to the image and not the mask.

    Args:
        scale (float): Scale the random noise from 0 to 1.
        
        probability (float): probability the transform is applied from 0 to 1.
    """

    def __init__(self, scale=0.2, probability=1.0):
        assert isinstance(scale, float)
        self.scale = scale
        assert isinstance(probability, float)
        assert probability <= 1.0
        self.probability = probability

    def __call__(self, sample):
        image = sample['image']
        
        if random.random() < self.probability:
            noise = torch.randn_like(image)
            image = image + torch.abs(noise * self.scale)

        sample['image'] = image

        return sample


class RandomPerspective(object):
    """Randomly alter perspective of the image in a sample.

    Args:
        distortion_scale (float): Argument to control the degree of distortion
            and ranges from 0 to 1. Default is 0.5.
        
        probability (float): Probability of the image being transformed.
            Default is 0.5.
    """

    def __init__(self, distortion_scale=0.5, probability=0.5, **kwargs):
        assert isinstance(distortion_scale, float)
        self.distortion_scale = distortion_scale
        assert isinstance(probability, float)
        self.probability = probability
        self.kwargs = kwargs
    
    def __perspective__(self, img, params):
        if type(img) is torch.Tensor:
            img = torchvision.transforms.ToPILImage()(img)
            img = torchvision.transforms.functional.perspective(img, *params,
                                                                **self.kwargs)
            img = torchvision.transforms.ToTensor()(img)
        
        else:
            img = torchvision.transforms.functional.perspective(img, *params,
                                                                **self.kwargs)
        
        return img

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        h, w = image.shape[1:3]
        
        perspective = torchvision.transforms.RandomPerspective(self.distortion_scale,
                                                               self.probability)
        params = perspective.get_params(w, h, self.distortion_scale)
        
        if random.random() < self.probability:
            image = self.__perspective__(image, params)
            mask = self.__perspective__(mask, params)

        sample['image'], sample['mask'] = image, mask

        return sample

    
class RandomRotation(object):
    """Randomly rotate the image in a sample.

    Args:
        degrees (tuple): Desired range of degrees for rotation in form
            (min, max).
        
        interval (int): Step size between rotation angles to choose.
        
        probability (float): probability the transform is applied from 0 to 1.
    """

    def __init__(self, degrees, interval=1, probability=1.0, **kwargs):
        assert isinstance(degrees, tuple)
        self.degrees = degrees
        assert isinstance(interval, int)
        self.interval = interval
        assert isinstance(probability, float)
        assert probability <= 1.0
        self.probability = probability
        self.kwargs = kwargs
    
    def __rotate__(self, img, angle):
        if type(img) is torch.Tensor:
            img = torchvision.transforms.ToPILImage()(img)
            img = torchvision.transforms.functional.rotate(img, angle,
                                                           **self.kwargs)
            img = torchvision.transforms.ToTensor()(img)
        
        else:
            img = torchvision.transforms.functional.rotate(img, angle,
                                                           **self.kwargs)
        
        return img

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        lo = self.degrees[0] // self.interval
        hi = self.degrees[1] // self.interval
        
        if random.random() < self.probability:
            angle = random.randint(lo, hi) * self.interval
            image = self.__rotate__(image, angle)
            mask = self.__rotate__(mask, angle)

        sample['image'], sample['mask'] = image, mask

        return sample

    
class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
        
        probability (float): probability the transform is applied from 0 to 1.
    """

    def __init__(self, output_size, probability=1.0):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        assert isinstance(probability, float)
        assert probability <= 1.0
        self.probability = probability
    
    def __center_crop__(self, img, size):
        if type(img) is torch.Tensor:
            img = torchvision.transforms.ToPILImage()(img)
            img = torchvision.transforms.functional.center_crop(img, size)
            img = torchvision.transforms.ToTensor()(img)
        
        else:
            img = torchvision.transforms.functional.center_crop(img, size)
        
        return img

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        h, w = image.shape[1:3]
        new_h, new_w = self.output_size

        if random.random() < self.probability:
            # apply random crop to image and mask
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            image = image[:,
                          top: top + new_h,
                          left: left + new_w]
            mask = mask[:,
                        top: top + new_h,
                        left: left + new_w]
        else:
            # apply center crop to image and mask
            image = self.__center_crop__(image, self.output_size)
            mask = self.__center_crop__(mask, self.output_size)
            

        sample['image'], sample['mask'] = image, mask

        return sample


class SARDataset(Dataset):
    def __init__(self, data_dir, band_list, mask_dir, transforms=None):
        self.samples = []
        self.data_dir = data_dir
        self.band_list = band_list
        self.mask_dict = self.loadMasks(mask_dir)
        self.transforms = transforms

        file_name = os.path.join(data_dir, band_list[0] + "band", "results.csv")
        # with open(data_dir + band_list[0] + "band/results.csv", "r") as csvfile:
        with open(file_name, "r") as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                self.samples.append(row)
    
    def loadMasks(self, mask_dir):
        masks = {}
        # for path in glob.glob(f"{mask_dir}/*.png"):
        for path in glob.glob(os.path.join(mask_dir, "*.png")):
            mask_img = Image.open(path)
            file_name = os.path.basename(path)
            mask_name = (file_name.split(".png")[0])
            if mask_name.endswith("Mask"):
                mask_name = mask_name[:-4]
            masks[mask_name] = mask_img
        return masks
    
    def __len__(self):
        return len(self.samples)      
    
    def __getitem__(self, idx):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        currSample = self.samples[idx]
        imageName = "0000" + currSample["uuid"] + ".png"
        imageName = imageName[-8:]
        
        process = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor()
        ])

        image = None
        for i, b in enumerate(self.band_list):
            img_path = os.path.join(self.data_dir, b + "band", str(imageName))
            # newImage = Image.open(self.data_dir + b + "band/" + str(imageName))
            newImage = Image.open(img_path)
            newImage = process(newImage)
            if image == None:
                image = torch.Tensor(len(self.band_list), newImage.shape[1], newImage.shape[2])
            image[i] = newImage
        currSample["image"] = image
        try:
            currSample["mask"] = process(self.mask_dict[currSample["run_name"]])
        except KeyError:
            # for the augmented dataset, mask name matches image name
            currSample["mask"] = process(self.mask_dict[imageName[:-4]])

        
        if self.transforms is not None:
            currSample = self.transforms(currSample)
        
        return currSample
