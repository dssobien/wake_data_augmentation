import torch
import torch.nn as nn

# from PicsArtHack-binary-segmentation github repo
class DiceLoss:
	def __init__(self, use_cuda=False, bce_weight=0.5, eps=1e-8, weight=None, smooth=1.0):
		self.bce = nn.BCEWithLogitsLoss(weight=weight)
		if use_cuda:
			self.bce.cuda()
		self.bce_weight = bce_weight
		self.eps = eps
		self.smooth = smooth

	def __call__(self, input, target):
		loss = self.bce_weight * self.bce(input, target)

		if self.bce_weight < 1.0:
			target = (target > 0.999).float()
			input = torch.sigmoid(input)
			intersection = (target * input).sum().float()
			union = target.sum().float() + input.sum().float()
			score = (2.0 * intersection + self.smooth) / (union + self.smooth + self.eps)
			loss -= (1.0 - self.bce_weight) * torch.log(score)
		return loss
