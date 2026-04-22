import torch
import torch.nn as nn


class DoubleConv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.block = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
		)

	def forward(self, x):
		return self.block(x)


class DownBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv = DoubleConv(in_channels, out_channels)

	def forward(self, x):
		return self.conv(self.pool(x))


class UpBlock(nn.Module):
	def __init__(self, in_channels, skip_channels, out_channels):
		super().__init__()
		self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
		self.conv = DoubleConv(out_channels + skip_channels, out_channels)

	def forward(self, x, skip):
		x = self.up(x)

		diff_y = skip.size(2) - x.size(2)
		diff_x = skip.size(3) - x.size(3)
		if diff_y != 0 or diff_x != 0:
			x = nn.functional.pad(
				x,
				[diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
			)

		x = torch.cat([skip, x], dim=1)
		return self.conv(x)


class UNet(nn.Module):
	def __init__(self, in_channels=3, num_classes=23, base_channels=64):
		super().__init__()

		self.enc1 = DoubleConv(in_channels, base_channels)
		self.enc2 = DownBlock(base_channels, base_channels * 2)
		self.enc3 = DownBlock(base_channels * 2, base_channels * 4)
		self.enc4 = DownBlock(base_channels * 4, base_channels * 8)
		self.bottleneck = DownBlock(base_channels * 8, base_channels * 16)

		self.dec4 = UpBlock(base_channels * 16, base_channels * 8, base_channels * 8)
		self.dec3 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
		self.dec2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
		self.dec1 = UpBlock(base_channels * 2, base_channels, base_channels)

		self.classifier = nn.Conv2d(base_channels, num_classes, kernel_size=1)

	def forward(self, x):
		s1 = self.enc1(x)
		s2 = self.enc2(s1)
		s3 = self.enc3(s2)
		s4 = self.enc4(s3)

		b = self.bottleneck(s4)

		x = self.dec4(b, s4)
		x = self.dec3(x, s3)
		x = self.dec2(x, s2)
		x = self.dec1(x, s1)

		return self.classifier(x)
