
import torch
import torch.nn as nn
import math

class DoubleConv2d(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(DoubleConv2d, self).__init__()
	
		self.doubleconv2d = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)
	
	def forward(self, x):
		return self.doubleconv2d(x)

class UpSample(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(UpSample, self).__init__()

		self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		self.conv2d = DoubleConv2d(in_channels, out_channels)
	
	def forward(self, normal, concat):
		x = self.upsample(normal)

		# concat is a different size than x, so trim
		delH = concat.size()[2] - x.size()[2]
		delW = concat.size()[3] - x.size()[3]

		x = nn.functional.pad(x, [delW // 2, delW - delW // 2,
					  delH // 2, delH - delH // 2])
		x = torch.cat([concat, x], dim=1)
		return self.conv2d(x)

class DownSample(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(DownSample, self).__init__()

		self.downsample = nn.Sequential(
			nn.MaxPool2d(kernel_size=2),
			DoubleConv2d(in_channels, out_channels)
		)
	
	def forward(self, x):
		return self.downsample(x)

class OutDoubleConv2d(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(OutDoubleConv2d, self).__init__()

		self.outdoubleconv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=2),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=2),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)
	
	def forward(self, x):
		return self.outdoubleconv(x)

class Unet(nn.Module):
	def __init__(self, in_channels, out_channels, band_list):
		super(Unet, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.band_list = band_list

		self.inp = DoubleConv2d(in_channels, 64)
		self.d1 = DownSample(64, 128)
		self.d2 = DownSample(128, 256)
		self.d3 = DownSample(256, 512)
		self.d4 = DownSample(512, 512)
		self.u1 = UpSample(1024, 256)
		self.u2 = UpSample(512, 128)
		self.u3 = UpSample(256, 64)
		self.u4 = UpSample(128, 32)
		self.out = OutDoubleConv2d(32, out_channels)
	
	def forward(self, x):
		x0 = self.inp(x)
		x1 = self.d1(x0)
		x2 = self.d2(x1)
		x3 = self.d3(x2)
		x4 = self.d4(x3)
		x = self.u1(x4, x3)
		x = self.u2(x, x2)
		x = self.u3(x, x1)
		x = self.u4(x, x0)
		return self.out(x)

class ClassifierSmall(nn.Module):
	def __init__(self, band_list, in_channels=1, out_channels=1):
		super(ClassifierSmall, self).__init__()

		self.in_channels = in_channels
		self.intermediate = 50
		self.out_channels = out_channels
		self.band_list = band_list

		self.conv = nn.Sequential(
			nn.Conv2d(in_channels, 20, kernel_size=3),
			nn.BatchNorm2d(20),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=4),
		)

		self.net = nn.Sequential(
			nn.Linear(20 * 255 * 255, self.intermediate),
			nn.ReLU(inplace=True),
			nn.Linear(self.intermediate, out_channels)
		)

	def forward(self, inputImage, inputMask=None):
		# inputMask is only here for the interface
		x = self.conv(inputImage)
		x = x.view(x.shape[1] * x.shape[2] * x.shape[3])
		return self.net(x)

class ClassifierLarge(nn.Module):
	def __init__(self, band_list, in_channels=1, out_channels=1):
		super(ClassifierLarge, self).__init__()

		self.in_channels = in_channels
		self.intermediate1 = 70
		self.intermediate2 = 30
		self.out_channels = out_channels
		self.band_list = band_list

		self.conv = nn.Sequential(
			nn.Conv2d(in_channels, 10, kernel_size=3),
			nn.BatchNorm2d(10),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(10, 20, kernel_size=3),
			nn.BatchNorm2d(20),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(20, 30, kernel_size=3),
			nn.BatchNorm2d(30),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2)
		)

		self.net = nn.Sequential(
			nn.Linear(30 * 126 * 126, self.intermediate1),
			nn.Linear(self.intermediate1, self.intermediate2),
			nn.Linear(self.intermediate2, out_channels)
		)

	def forward(self, inputImage):
		x = self.conv(inputImage)
		x = x.view(x.shape[1] * x.shape[2] * x.shape[3])
		return self.net(x)

class ClassifierUnet(nn.Module):
	def __init__(self, band_list, in_channels=1, out_channels=1, n_batch=1):
		super(ClassifierUnet, self).__init__()

		self.in_channels = in_channels
		self.intermediate1 = 70
		self.intermediate2 = 30
		self.out_channels = out_channels
		self.band_list = band_list

		self.conv = nn.Sequential(
			nn.Conv2d(in_channels + 1, 10, kernel_size=3),
			nn.BatchNorm2d(10),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(10, 20, kernel_size=3),
			nn.BatchNorm2d(20),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(20, 30, kernel_size=3),
			nn.BatchNorm2d(30),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2)
		)

		self.net = nn.Sequential(
			nn.Linear(n_batch * 30 * 126 * 126, self.intermediate1),
			nn.Linear(self.intermediate1, self.intermediate2),
			nn.Linear(self.intermediate2, out_channels)
		)

	def forward(self, inputImage, inputMask=None):
		if inputMask is not None:
			x = self.conv(torch.cat((inputImage.detach(), inputMask.detach()), 1))
		else:
			image2 = inputImage[0][0].unsqueeze(0).unsqueeze(0)
			x = self.conv(torch.cat((inputImage.detach(), image2.detach()), 1))
			#x = self.conv(inputImage)
		x = x.view(x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3])
		return self.net(x)

