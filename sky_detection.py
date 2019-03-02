#!/usr/bin/python

import os,cv2,numpy

HISTOGRAM_DIMENSION_SIZE=32

skyness_lookup_table=None

def init(lookup_table_fname):
	global skyness_lookup_table

	skyness_lookup_table=numpy.load(lookup_table_fname,allow_pickle=False)

def calc_image_skyness(img):
	global skyness_lookup_table

	if skyness_lookup_table is None:
		init(os.path.dirname(__file__) + '/skyness.npy')

	quantised_img=numpy.floor_divide(img,256/HISTOGRAM_DIMENSION_SIZE)
	skyness_img=skyness_lookup_table[quantised_img[:,:,0],quantised_img[:,:,1],quantised_img[:,:,2]]

	erosion_kernel=numpy.ones((30,30),numpy.uint8)
	cv2.erode(skyness_img,erosion_kernel,skyness_img)

	return skyness_img

if __name__ == '__main__':
	import sys

	operation_mode=sys.argv[1]

	if operation_mode not in ('train','detect','score'):
		print 'Invalid operation_mode',operation_mode
		exit(0)

	if operation_mode == 'train':
		output_fname,nonsky_images_directory,sky_images_directory=sys.argv[2:]

		cumulative_histogram=None
		class_histograms=[None,None]

		for is_sky,dirname in ((False,nonsky_images_directory),(True,sky_images_directory)):
			for fname in os.listdir(dirname):
				if not fname.lower().endswith('.jpg') and not fname.lower().endswith('.jpeg'):
					continue
				# if hash(fname) % 100:
				#	continue
				img=cv2.imread(dirname + '/' + fname)
				histogram=cv2.calcHist([img],[0,1,2],None,[HISTOGRAM_DIMENSION_SIZE]*3,[0,256]*3)
				if class_histograms[int(is_sky)] is None:
					class_histograms[int(is_sky)]=histogram
				else:
					class_histograms[int(is_sky)]+=histogram

		total_histogram=sum(class_histograms)
		counts_scaling_coeff=float(numpy.sum(class_histograms[0])) / float(numpy.sum(class_histograms[1])) - 1

		skyness_lookup_table=numpy.zeros((HISTOGRAM_DIMENSION_SIZE,) * 3,numpy.uint8)

		# print total_histogram

		bin_counts=[]
		for idx1 in range(HISTOGRAM_DIMENSION_SIZE):
			for idx2 in range(HISTOGRAM_DIMENSION_SIZE):
				for idx3 in range(HISTOGRAM_DIMENSION_SIZE):
					skyness=0
					for box_size_length in range(1,HISTOGRAM_DIMENSION_SIZE+1):
						slice_index=[]
						for idx in (idx1,idx2,idx3):
							min_idx=max(0,idx - box_size_length/2)
							slice_index.append(slice(min_idx,min_idx+box_size_length))
						counts_sum=numpy.sum(total_histogram[tuple(slice_index)])
						if counts_sum >= 1000:
							sky_counts_sum=numpy.sum(class_histograms[1][tuple(slice_index)])
							additional_sky_count=sky_counts_sum * counts_scaling_coeff
							skyness=int((sky_counts_sum + additional_sky_count) * 255 / \
																	(counts_sum + additional_sky_count))
							break

					skyness_lookup_table[idx1,idx2,idx3]=skyness
					bin_counts.append((box_size_length,counts_sum,sky_counts_sum/counts_sum,idx1,idx2,idx3))

		if output_fname:
			numpy.save(output_fname,skyness_lookup_table,allow_pickle=False)
		else:
			for row in sorted(bin_counts):
				print row

	elif operation_mode == 'detect':
		skyness_fname,input_fname,output_fname=sys.argv[2:]

		init(skyness_fname)
		cv2.imwrite(output_fname,calc_image_skyness(cv2.imread(input_fname)))

	elif operation_mode == 'score':
		skyness_fname=sys.argv[2]

		for input_fname in sys.argv[3:]:
			im=calc_image_skyness(cv2.imread(input_fname))
			score=numpy.sum(im) * 999 / (255.0 * im.shape[0] * im.shape[1])
			print '%03.0f %s' % (score,input_fname)
