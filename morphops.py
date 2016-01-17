import numpy as np
from IPython.nbformat import v2

import cv2
import matplotlib.pyplot as plt


class morphopsonbinimgs():
	def __init__(self, filename):
		self.img = cv2.imread(filename, 0)  # reads grayscale directly
		plt.ion()
		plt.subplot(1, 2, 1)
		plt.title('Gray scale')
		plt.imshow(self.img, 'gray')
		ret, self.binimg = cv2.threshold(self.img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		plt.subplot(1, 2, 2)
		plt.title('Binary with otsu method')
		self.rows, self.columns = self.img.shape
		self.create_windows()
		self.pad_image()

		plt.imshow(self.binimg, 'gray')

	def create_windows(self):
		self.cross5 = [[False, False, True, False, False], [False, False, True, False, False],
					   [True, True, True, True, True], [False, False, True, False, False],
					   [False, False, True, False, False]]
		self.cross5 = np.bool(self.cross5)
		self.square3 = np.ones((3, 3)) == np.ones((3, 3))

	def pad_image(self):
		self.pad_rows = 3
		self.pad_columns = 3
		pad_rows = self.pad_rows
		pad_columns = self.pad_columns
		newimage = np.empty((self.rows + 2 * pad_rows, self.columns + 2 * pad_columns))
		newimage[pad_rows:-pad_rows, pad_columns:-pad_columns] = self.binimg
		# newimage[pad_rows:-pad_rows,pad_columns:-pad_columns] = self.binimg

		self.paddedbinimage = newimage

	def dilate_cross(self):
		img = self.paddedbinimage;
		rows, cols = img.shape
		newimage = np.empty_like(self.binimg)
		for i in range(self.pad_rows, self.rows + self.pad_rows):
			for j in range(self.pad_columns, self.columns + self.pad_columns):
				roi = self.getROIpixels(2, i, j)
				roipixelarray = roi[self.cross5]
				newimage[i - self.pad_rows, j - self.pad_columns] = np.all(roipixelarray)
			# print ('j',j)
			# print ('i',i)
		self.dil_crossimg = newimage

	def dilate_square(self):
		img = self.paddedbinimage;
		rows, cols = img.shape
		newimage = np.empty_like(self.binimg)
		for i in range(self.pad_rows, self.rows + self.pad_rows):
			for j in range(self.pad_columns, self.columns + self.pad_columns):
				roi = self.getROIpixels(2, i, j)
				roipixelarray = roi[self.square3]
				newimage[i - self.pad_rows, j - self.pad_columns] = np.all(roipixelarray)
			# print ('j',j)
			# print ('i',i)
		self.dil_squareimg = newimage

	def erode_cross(self):
		img = self.paddedbinimage;
		rows, cols = img.shape
		newimage = np.empty_like(self.binimg)
		for i in range(self.pad_rows, self.rows + self.pad_rows):
			for j in range(self.pad_columns, self.columns + self.pad_columns):
				roi = self.getROIpixels(2, i, j)
				roipixelarray = roi[self.cross5]
				newimage[i - self.pad_rows, j - self.pad_columns] = np.any(roipixelarray)
			# print ('j',j)
			# print ('i',i)
		self.erod_crossimg = newimage

	def erode_square(self):
		img = self.paddedbinimage;
		rows, cols = img.shape
		newimage = np.empty_like(self.binimg)
		for i in range(self.pad_rows, self.rows + self.pad_rows):
			for j in range(self.pad_columns, self.columns + self.pad_columns):
				roi = self.getROIpixels(2, i, j)
				roipixelarray = roi[self.square3]
				newimage[i - self.pad_rows, j - self.pad_columns] = np.any(roipixelarray)
			# print ('j',j)
			# print ('i',i)
		self.erod_squareimg = newimage

	def median_square(self):
		img = self.paddedbinimage;
		rows, cols = img.shape
		newimage = np.empty_like(self.binimg)
		for i in range(self.pad_rows, self.rows + self.pad_rows):
			for j in range(self.pad_columns, self.columns + self.pad_columns):
				roi = self.getROIpixels(2, i, j)
				roipixelarray = roi[self.square3]
				newimage[i - self.pad_rows, j - self.pad_columns] = np.median(roipixelarray)
			# print ('j',j)
			# print ('i',i)
		self.median_squareimg = newimage

	def median_cross(self):
		img = self.paddedbinimage;
		rows, cols = img.shape
		newimage = np.empty_like(self.binimg)
		for i in range(self.pad_rows, self.rows + self.pad_rows):
			for j in range(self.pad_columns, self.columns + self.pad_columns):
				roi = self.getROIpixels(2, i, j)
				roipixelarray = roi[self.cross5]
				newimage[i - self.pad_rows, j - self.pad_columns] = np.median(roipixelarray)
			# print ('j',j)
			# print ('i',i)
		self.median_crossimg = newimage


	def all_ops(self):
		self.dilate_cross()
		self.dilate_square()
		self.erode_cross()
		self.erode_square()
		self.median_cross()
		self.median_square()


		plt.close('all')
		plt.figure()
		plt.subplot(4, 2, 1)
		plt.title('Original Grayscale Image')
		plt.imshow(self.img, 'gray')
		plt.subplot(4, 2, 2)
		plt.title('Binary Grayscale Image')
		plt.imshow(self.binimg, 'gray')
		plt.subplot(4, 2, 3)
		plt.title('Dilated with cross')
		plt.imshow(self.dil_crossimg, 'gray')
		plt.subplot(4, 2, 4)
		plt.title('Dilated with square')
		plt.imshow(self.dil_squareimg, 'gray')
		plt.subplot(4, 2, 5)
		plt.title('Eroded with cross')
		plt.imshow(self.erod_crossimg, 'gray')
		plt.subplot(4, 2, 6)
		plt.title('Eroded with square')
		plt.imshow(self.erod_squareimg, 'gray')
		plt.subplot(4, 2, 7)
		plt.title('Median Filter with cross')
		plt.imshow(self.median_crossimg, 'gray')
		plt.subplot(4, 2, 8)
		plt.title('Median Filter with square')
		plt.imshow(self.median_squareimg, 'gray')
	def getROIpixels(self, sqradius, row, col):
		# gives a square worth of pixels centered around at row,col from the binary
		# image. row and col are specified in normal numpy numbering. ie o to start. squareradius is the number of pixels
		# padding. Set it to 2 to get a 5 by 5 square
		try:
			return (self.paddedbinimage[row - sqradius:row + sqradius + 1, col - sqradius:col + sqradius + 1])
		except(IndexError):
			raise ('Check radius of the square. Index exceeding the padded images maximum')


a = morphopsonbinimgs('5.1.13.tiff')
