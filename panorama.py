#!/usr/bin/python
# -*- encoding: latin-1 -*-

import sys,math,operator,re,xml.sax.handler,xml.sax,numpy,cv2,matplotlib.pyplot
import iterative_optimiser

RESIZE_FACTOR=8
IMG_HISTOGRAM_SIZE=10

detector_patch_size=31
detector=cv2.ORB_create(nfeatures=500,patchSize=detector_patch_size)

bf_matcher=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
clahe=cv2.createCLAHE(clipLimit=40,tileGridSize=(8,8))

class ImageKeypoints:
	class Keypoints:
		def __init__(self,img=None,x1=None,x2=None,y1=None,y2=None):
			self.descriptors=None
			self.xys=[]

			if img is None:
				return

			kp,self.descriptors=detector.detectAndCompute(img[y1:y2,x1:x2],None)

			if self.descriptors is not None:
				for keypoint in kp:
					self.xys.append((	int(x1 + keypoint.pt[0]),
										int(y1 + keypoint.pt[1])))

		def __iadd__(self,kp):
			if kp.descriptors is not None:
				if self.descriptors is None:
					self.descriptors=kp.descriptors
				else:
					self.descriptors=numpy.concatenate((self.descriptors,kp.descriptors))

			self.xys.extend(kp.xys)
			return self

	def __init__(self,fname):
		self.img=cv2.imread(fname)
		if RESIZE_FACTOR != 1:
			self.img=cv2.resize(self.img,(0,0),fx=1.0 / RESIZE_FACTOR,fy=1.0 / RESIZE_FACTOR)

		# self.img=cv2.Laplacian(self.img,cv2.CV_8U,ksize=5)	# somewhat works
		# self.img=cv2.Canny(self.img,10,20)

		self.channels=[]

		self.add_channel(self.img)

		b,g,r=cv2.split(self.img)
		self.add_channel(cv2.subtract(
							cv2.add(cv2.transform(b,numpy.array((0.5,))),
									128),
							cv2.transform(r,numpy.array((0.5,)))))

			##### Build self.histogram[] #####

		total_keypoints=0
		keypoint_counts=[[0 for i in range(IMG_HISTOGRAM_SIZE)] for j in range(IMG_HISTOGRAM_SIZE)]
		x_bin_size=(self.img.shape[1] + IMG_HISTOGRAM_SIZE/2) / IMG_HISTOGRAM_SIZE
		y_bin_size=(self.img.shape[0] + IMG_HISTOGRAM_SIZE/2) / IMG_HISTOGRAM_SIZE
		for chan in self.channels:
			total_keypoints+=len(chan.xys)
			for x,y in chan.xys:
				keypoint_counts[min(IMG_HISTOGRAM_SIZE-1,x / x_bin_size)] \
															[min(IMG_HISTOGRAM_SIZE-1,y / y_bin_size)]+=1

		self.histogram=[]
		for x_idx in range(IMG_HISTOGRAM_SIZE):
			x=(x_idx + 0.5) * x_bin_size
			for y_idx in range(IMG_HISTOGRAM_SIZE):
				y=(y_idx + 0.5) * y_bin_size
				if keypoint_counts[x_idx][y_idx]:
					self.histogram.append((x,y,keypoint_counts[x_idx][y_idx] / float(total_keypoints)))

		print '%s %s keypoints' % (fname,'+'.join([str(len(chan.xys)) for chan in self.channels]))

	def add_channel(self,img):
		if len(img.shape) >= 3:
			img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

		img=clahe.apply(img)

		x_splits=tuple([i*img.shape[1]/5 for i in range(5+1)])
		y_splits=tuple([i*img.shape[0]/5 for i in range(5+1)])

		self.channels.append(ImageKeypoints.Keypoints())

		for x_idx in range(len(x_splits)-1):
			for y_idx in range(len(y_splits)-1):
				self.channels[-1]+=ImageKeypoints.Keypoints(img,
						x_splits[x_idx],min(img.shape[1],x_splits[x_idx+1] + 2*detector_patch_size),
						y_splits[y_idx],min(img.shape[0],y_splits[y_idx+1] + 2*detector_patch_size))

	def show_img_with_keypoints(self,channel_idx,highlight_indexes=tuple()):
		for idx,xy in enumerate(self.channels[channel_idx].xys):
			highlight=(idx in highlight_indexes)
			color=(255,0,0) if highlight else (0,255,0)
			cv2.circle(self.img,xy,(15 if highlight else 10) / RESIZE_FACTOR,color,-1)

		matplotlib.pyplot.imshow(self.img)
		matplotlib.pyplot.show()

def calc_shift_for_angle(img1,img2,matches,angle_deg):
	angle_sin=math.sin(math.radians(angle_deg))
	angle_cos=math.cos(math.radians(angle_deg))

	img1_size=(img1.img.shape[1],img1.img.shape[0])
	img2_size=(img2.img.shape[1],img2.img.shape[0])

	histogram_bin_pixels=int(50 / RESIZE_FACTOR)

	xd2_add=(1-angle_cos) * img2_size[0] + angle_sin * img2_size[1]
	yd2_add=(1-angle_cos) * img2_size[1] - angle_sin * img2_size[0]

	xy_deltas=[]
	histogram=dict()
	for distance,x1,y1,x2,y2 in matches[:1000]:
		xd=x1 - int(x2 * angle_cos - y2 * angle_sin + xd2_add)
		yd=y1 - int(x2 * angle_sin + y2 * angle_cos + yd2_add)

		# print 'ZZZ',xd*RESIZE_FACTOR,yd*RESIZE_FACTOR,distance

		idx=(xd / histogram_bin_pixels,yd / histogram_bin_pixels)
		xy_deltas.append((idx,xd,yd))

		if idx not in histogram:
			histogram[idx]=0
		histogram[idx]+=1

	best_idx=None
	best_count=0
	best_coverage=1
	for idx,count in histogram.items():
		if count < 5:
			continue

		img2_keypoints_coverage=0

		for x2,y2,keypoints_coverage in img2.histogram:
			x1=x2 * angle_cos - y2 * angle_sin + xd2_add + xd
			if x1 < 0 or x1 > img1_size[0]:
				continue
			y1=x2 * angle_sin + y2 * angle_cos + yd2_add + yd
			if y1 < 0 or y1 > img1_size[1]:
				continue

			img2_keypoints_coverage+=keypoints_coverage

		img2_keypoints_coverage=max(0.1,img2_keypoints_coverage)

		# area_coverage=max(0.1,max(0,1 - abs(idx[0] * histogram_bin_pixels) / float(img2_size[0])) * \
		#						max(0,1 - abs(idx[1] * histogram_bin_pixels) / float(img2_size[1])))

		if best_count / float(best_coverage) <= count / float(img2_keypoints_coverage):
			best_count=count
			best_coverage=img2_keypoints_coverage
			best_idx=idx

	if best_idx is None:
		return (0,1,tuple(),0,0)

	inlier_indexes=[]
	for x_shift in (-1,0,+1):
		for y_shift in (-1,0,+1):
			inlier_indexes.append((best_idx[0] + x_shift,best_idx[1] + y_shift))

	xd_sum=0
	yd_sum=0
	inliers=[]
	for match_idx,(idx,xd,yd) in enumerate(xy_deltas):
		if idx in inlier_indexes:
			xd_sum+=xd
			yd_sum+=yd
			inliers.append(match_idx)

	xd_sum*=RESIZE_FACTOR
	yd_sum*=RESIZE_FACTOR

	best_count+=len(inliers)

	# print angle_deg,best_count,best_coverage,xd_sum/len(inliers),yd_sum/len(inliers)

	return (best_count,best_coverage,inliers,xd_sum / len(inliers),yd_sum / len(inliers))

def calc_shift_ratio(xd,yd):
	abs_shifts=(abs(xd),abs(yd))
	return min(abs_shifts) / float(max(1,max(abs_shifts)))

def find_matches(img1,img2):
	global bf_matcher

	matches=[]
	for chan1,chan2 in zip(img1.channels,img2.channels):
		for m in bf_matcher.match(chan1.descriptors,chan2.descriptors):
			matches.append((m.distance,	chan1.xys[m.queryIdx][0],chan1.xys[m.queryIdx][1],
										chan2.xys[m.trainIdx][0],chan2.xys[m.trainIdx][1]))

	matches.sort(key=operator.itemgetter(0))

	debug_str='%d matches' % len(matches)

	if not matches:
		return (debug_str,'')

	debug_str+=', distances %.0f:%.0f' % (matches[0][0],matches[:30][-1][0])

	best_angle_deg=0
	best_score=0
	best_count=0
	best_coverage=1
	best_inliers=None	# In increasing matches[] index order
	best_xd=0
	best_yd=0

	for angle_deg in range(-180,+180,1):
		count,coverage,inliers,xd,yd=calc_shift_for_angle(img1,img2,matches,angle_deg)
		score=int(count / float(coverage))

		if score > best_score:
			best_score=score
			best_count=count
			best_coverage=coverage
			best_inliers=inliers
			best_xd=xd
			best_yd=yd
			best_angle_deg=angle_deg

	debug_str+=', %+ddeg, score %d/%.2f=%d, shift %+dpx,%+dpx' % (best_angle_deg,best_count,
															best_coverage,best_score,best_xd,best_yd)
	abs_shifts=(abs(best_xd),abs(best_yd))
	shift_ratio=min(abs_shifts) / float(max(1,max(abs_shifts)))

	decision_value=best_score - (100 + min(100,abs(best_angle_deg) * 3) + shift_ratio * 60)		#!!!
	decision_value=best_score*0.00726 + best_count*0.1049 + abs(best_angle_deg)*-0.0482 + (-1.885)

	if decision_value < 0 or best_score <= 0:
		return (debug_str,'')

	# src_pts=numpy.float32([(x1,y1) for distance,x1,y1,x2,y2 in matches[:1000]]).reshape(-1,1,2)
	# dst_pts=numpy.float32([(x2,y2) for distance,x1,y1,x2,y2 in matches[:1000]]).reshape(-1,1,2)
	# homography_matrix=cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0)[0]
	# print homography_matrix

	max_dist=(img1.img.shape[0] + img1.img.shape[1]) / 50
	representative_xy_pairs=[]
	output_string=''
	for i in best_inliers:
		distance,x1,y1,x2,y2=matches[i]
		for xy_pair in representative_xy_pairs:
			if (x1 - xy_pair[0])**2 + (y1 - xy_pair[1])**2 < max_dist*max_dist:
				break
		else:
			new_xy_pair=(x1,y1,x2,y2,i)
			representative_xy_pairs.append(new_xy_pair)
			output_string+='                <point x1="%d" y1="%d" x2="%d" y2="%d"/>\n' % \
												tuple([value*RESIZE_FACTOR for value in new_xy_pair[:4]])
			if len(representative_xy_pairs) >= 15:
				break

	return (debug_str,output_string)

def print_matches_for_images(image_fnames):
	images=[ImageKeypoints(fname) for fname in image_fnames]

	link_stats=[[] for img in images]

	for idx1,img1 in enumerate(images):
		for idx2,img2 in enumerate(images):
			if idx2 <= idx1:
				continue

			debug_str,output_string=find_matches(img1,img2)

			print '        <!-- image %d<-->%d: %s -->' % (idx1,idx2,debug_str)

			if output_string:
				print '        <match image1="%d" image2="%d">\n            <points>\n%s            </points>\n        </match>' % \
																				(idx1,idx2,output_string)
				link_stats[idx1].append(idx2)
				link_stats[idx2].append(idx1)

	print 'Link stats:'

	for idx,linked_images in enumerate(link_stats):
		print '#%d links: %s' % (idx,' '.join(map(str,linked_images)))

keyword_args={}
positional_args=[]

for arg in sys.argv[1:]:
	if arg.startswith('--'):
		keyword,_,value=arg.partition('=')
		if keyword not in keyword_args:
			keyword_args[keyword]=value
		else:
			if not isinstance(keyword_args[keyword],list):
				keyword_args[keyword]=[keyword_args[keyword],]
			keyword_args[keyword].append(value)
	else:
		positional_args.append(arg)

if not positional_args:
	print '''Usage:
	%s IMAGE-FNAME
		Extract and show keypoints from one image

	%s IMAGE-FNAME IMAGE-FNAME [IMAGE-FNAME ...]
		Match a set of images to each other, and print out raw matches for an AutoPano XML file

	%s --testcase-fname=PANO-XML-FNAME [--print-training-data] MATCHES-XML-FNAME
		Optimise the matches classifier (decision_value formula) against a testcase, using a raw matches file as input''' % \
		((sys.argv[0],) * 3)

	exit(1)

correct_matches=None
testcase_fname=keyword_args.get('--testcase-fname')
if testcase_fname:
	correct_matches=dict()

	class kolor_xml_handler(xml.sax.handler.ContentHandler):
		def __init__(self):
			self.image_fnames=[]

		def startElement(self,name,attrs):
			global correct_matches

			if name == 'def':
				fname=attrs.get('filename')
				for fname2 in self.image_fnames:
					correct_matches[tuple(sorted((fname,fname2)))]=False
				self.image_fnames.append(fname)
			elif name == 'match':
				img1=self.image_fnames[int(attrs.get('image1'))]
				img2=self.image_fnames[int(attrs.get('image2'))]
				correct_matches[tuple(sorted((img1,img2)))]=True

	parser=xml.sax.make_parser()
	parser.setContentHandler(kolor_xml_handler())
	parser.parse(open(testcase_fname,'r'))

	print_training_data=('--print-training-data' in keyword_args)
	training_data=[]
	optimiser_params=(('score',iterative_optimiser.FloatParam(0,1)),
					('count',iterative_optimiser.FloatParam(0,1)),
					('angle_deg_limit50',iterative_optimiser.FloatParam(-1,0)),
					('shift_ratio',iterative_optimiser.FloatParam(-1,0)),
					)

	def optimiser_test_func(params):
		global training_data

		scores=[]
		total_correct_matches=0
		for e in training_data:
			scores.append((sum(value*weight for value,weight in zip(e[1:],params)),e[0]))
			total_correct_matches+=int(bool(e[0]))

		scores.sort()

		best_correct_predictions=0
		best_threshold=0

		# print

		cumulative_correct_matches=0
		prev_score=None
		for idx,(score,is_correct_match) in enumerate(scores):
			correct_predictions_at_this_threshold=(idx - cumulative_correct_matches) + \
													(total_correct_matches - cumulative_correct_matches)
			# print idx,is_correct_match,cumulative_correct_matches,total_correct_matches,correct_predictions_at_this_threshold,score

			cumulative_correct_matches+=int(bool(is_correct_match))
			if correct_predictions_at_this_threshold > best_correct_predictions:
				best_correct_predictions=correct_predictions_at_this_threshold
				if prev_score is None:
					prev_score=score
				best_threshold=0.5 * (score + prev_score)
				prev_score=score

		return (best_correct_predictions,str(best_threshold))

	tries=0
	nonzero_tries,nonzero_successes=0,0
	correct_preditions_with_zero_score=0

	image_fnames=[]
	for line in open(positional_args[0],'r'):
		if line.strip().endswith(' keypoints'):
			image_fnames.append(line.partition(' ')[0].rpartition('/')[2])
			continue

		m=re.search(r'<!-- image ([0-9]+)<-->([0-9]+): .* ([-+]*[0-9.]+)deg, score ([0-9.]+)/[0-9.]+=([0-9.]+), shift ([-+]*[0-9.]+)px, *([-+]*[0-9.]+)px',line)
		if not m:
			continue

		fields=m.groups()
		img_idx1=int(fields[0])
		img_idx2=int(fields[1])
		angle_deg=float(fields[2])
		count=float(fields[3])
		score=float(fields[4])
		x_shift=int(fields[5])
		y_shift=int(fields[6])

		fnames_pair=tuple(sorted((image_fnames[img_idx1],image_fnames[img_idx2])))
		is_correct_match=correct_matches.get(fnames_pair)
		if is_correct_match is None:
			print 'Warning: image pair %s %s not present in testcases' % fnames_pair
			continue

		if score > 0:
			shift_ratio=calc_shift_ratio(x_shift,y_shift)
			training_data.append((int(is_correct_match),score,count,min(50,abs(angle_deg)),shift_ratio))

			if print_training_data:
				print '%d 1:%s 2:%s 3:%s 4:%s' % training_data[-1]

			decision_value=score - (100 + min(100,abs(angle_deg) * 3) + shift_ratio * 60)
			decision_value=score*0.00726 + count*0.1049 + min(50,abs(angle_deg))*-0.0482 + (-1.885)		# trained for test-pano-2chan-kpcoverage2.mypoints
			decision_value=score*-0.00017 + count*0.157 + min(50,abs(angle_deg))*-0.0372 + shift_ratio*-1.341 + (-2.595)		# trained for test-pano-2chan-resize8-kpcoverage2-liblinear.mypoints
			# decision_value=score*0.02 + count*0.74 + min(50,abs(angle_deg))*-0.24 + shift_ratio*-0.38 + (-16.145447725)		# iterative_optimiser trained for test-pano-2chan-resize8-kpcoverage2-liblinear.mypoints

			predicted=(decision_value >= 0)
			nonzero_successes+=int(predicted == is_correct_match)
			nonzero_tries+=1
		else:
			predicted=False
			decision_value=-1.11111
			correct_preditions_with_zero_score+=int(not is_correct_match)

		tries+=1

		if not print_training_data and predicted != is_correct_match:
			print is_correct_match,decision_value,line.strip()

	if not print_training_data and tries:
		print 'Successes: %u/%u %.2f%% (nonzero links only)' % (nonzero_successes,nonzero_tries,
																	nonzero_successes*100.0/nonzero_tries)
		print 'Successes: %u/%u %.2f%%' % (nonzero_successes + correct_preditions_with_zero_score,
								tries,(nonzero_successes + correct_preditions_with_zero_score)*100.0/tries)

		# print 'Optimising with the following parameters:', \
		#									' '.join(map(operator.itemgetter(0),optimiser_params))
		best_params=iterative_optimiser.optimise([p[1] for p in optimiser_params],
																			optimiser_test_func,30,False)
		iterative_optimiser_successes,threshold_str=optimiser_test_func(best_params)
		print 'Successes: %u/%u %.2f%% (iterative_optimiser with params %s and threshold %s)' % \
							(iterative_optimiser_successes + correct_preditions_with_zero_score,
							tries,
							(iterative_optimiser_successes + correct_preditions_with_zero_score) * \
																							100.0/tries,
							' '.join(map(str,best_params)),threshold_str)

if len(positional_args) == 1:
	ImageKeypoints(positional_args[0]).show_img_with_keypoints(0)
else:
	print_matches_for_images(positional_args)

# img1.show_img_with_keypoints([matches[xy_pair[4]].queryIdx for xy_pair in representative_xy_pairs])
# if len(sys.argv) >= 1+3:
#	cv2.imwrite(sys.argv[3],img1.img)
