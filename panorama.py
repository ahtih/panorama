#!/usr/bin/python
# -*- encoding: latin-1 -*-

import sys,math,operator,re,xml.sax.handler,xml.sax,gc,itertools,multiprocessing,numpy,cv2,matplotlib.pyplot
import iterative_optimiser,exif

RESIZE_FACTOR=4
KEYPOINT_BLOCKS=5
IMG_HISTOGRAM_SIZE=10

max_procs=8	#!!!

detector_patch_size=31
detector=cv2.ORB_create(nfeatures=1000,patchSize=detector_patch_size)

keypoint_matcher=cv2.FlannBasedMatcher({'algorithm': 6, 'table_number': 6, 'key_size': 12,
										'multi_probe_level': 1},{'checks': 50})
clahe=cv2.createCLAHE(clipLimit=40,tileGridSize=(16,16))

classifier_params=(0.01,0.88,-0.11,-0.22,+8.130)

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

	def __init__(self,fname,deallocate_image=False):
		self.img=cv2.imread(fname)
		if RESIZE_FACTOR != 1:
			self.img=cv2.resize(self.img,(0,0),fx=1.0 / RESIZE_FACTOR,fy=1.0 / RESIZE_FACTOR)

		self.img_shape=tuple(self.img.shape)

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

		self.channel_keypoints=tuple([len(chan.xys) for chan in self.channels])

		if deallocate_image:
			self.img=None

	def add_channel(self,img):
		global KEYPOINT_BLOCKS

		if len(img.shape) >= 3:
			img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

		img=clahe.apply(img)

		x_splits=tuple([i*img.shape[1]/KEYPOINT_BLOCKS for i in range(KEYPOINT_BLOCKS+1)])
		y_splits=tuple([i*img.shape[0]/KEYPOINT_BLOCKS for i in range(KEYPOINT_BLOCKS+1)])

		self.channels.append(ImageKeypoints.Keypoints())

		for x_idx in range(len(x_splits)-1):
			for y_idx in range(len(y_splits)-1):
				self.channels[-1]+=ImageKeypoints.Keypoints(img,
						x_splits[x_idx],min(img.shape[1],x_splits[x_idx+1] + 2*detector_patch_size),
						y_splits[y_idx],min(img.shape[0],y_splits[y_idx+1] + 2*detector_patch_size))

	def calc_keypoints_coverage(self,img2_size,angle_sin,angle_cos,x_add,y_add):
		coverage_sum=0

		for bin_x,bin_y,keypoints_coverage in self.histogram:
			x=bin_x * angle_cos - bin_y * angle_sin + x_add
			if x < 0 or x > img2_size[0]:
				continue
			y=bin_x * angle_sin + bin_y * angle_cos + y_add
			if y < 0 or y > img2_size[1]:
				continue

			coverage_sum+=keypoints_coverage

		return coverage_sum

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

	img1_size=(img1.img_shape[1],img1.img_shape[0])
	img2_size=(img2.img_shape[1],img2.img_shape[0])

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

		img2_keypoints_coverage=max(0.1,img2.calc_keypoints_coverage(img1_size,angle_sin,angle_cos,
																	xd2_add + idx[0]*histogram_bin_pixels,
																	yd2_add + idx[1]*histogram_bin_pixels))

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

def calc_classifier_decision_value(inputs,params):
	decision_value=sum(value*weight for value,weight in zip(inputs,params))
	if len(params) > len(inputs):
		decision_value-=params[-1]
	return decision_value

def find_matches(img1,img2):
	global keypoint_matcher

	matches=[]
	for chan1,chan2 in zip(img1.channels,img2.channels):
		for m,m2 in keypoint_matcher.knnMatch(chan1.descriptors,chan2.descriptors,k=2):
			if m.distance < 0.8 * m2.distance:
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

	decision_value=calc_classifier_decision_value(
						(best_score,best_count,min(50,abs(best_angle_deg)),shift_ratio),classifier_params)
	if decision_value < 0 or best_score <= 0:
		return (debug_str,'')

	# src_pts=numpy.float32([(x1,y1) for distance,x1,y1,x2,y2 in matches[:1000]]).reshape(-1,1,2)
	# dst_pts=numpy.float32([(x2,y2) for distance,x1,y1,x2,y2 in matches[:1000]]).reshape(-1,1,2)
	# homography_matrix=cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0)[0]
	# print homography_matrix

	max_dist=(img1.img_shape[0] + img1.img_shape[1]) / 50
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

def worker_func(args):
	try:
		global images_with_keypoints
		idx1,idx2=args
		return args + tuple(find_matches(images_with_keypoints[idx1],images_with_keypoints[idx2]))
	except KeyboardInterrupt:
		return None

def print_matches_for_images(output_fd):
	global max_procs,images_with_keypoints

	link_stats=[[] for img in images_with_keypoints]

	gc.collect()
	worker_pool=multiprocessing.Pool(max_procs) if max_procs > 1 else None

	worker_args=[]
	for idx1,img1 in enumerate(images_with_keypoints):
		for idx2,img2 in enumerate(images_with_keypoints):
			if idx2 > idx1:
				worker_args.append((idx1,idx2))
	if worker_pool is not None:
		results=worker_pool.imap(worker_func,worker_args,5)		#!!! Try imap_unordered()
	else:
		results=itertools.imap(worker_func,worker_args)

	for idx1,idx2,debug_str,output_string in results:
		print >>output_fd,'        <!-- image %d<-->%d: %s -->' % (idx1,idx2,debug_str)

		if output_string:
			print >>output_fd,'        <match image1="%d" image2="%d">\n            <points>\n%s            </points>\n        </match>' % \
																				(idx1,idx2,output_string)
			link_stats[idx1].append(idx2)
			link_stats[idx2].append(idx1)

	print >>output_fd,'<!-- Link stats: -->'

	for idx,linked_images in enumerate(link_stats):
		print >>output_fd,'<!-- #%d links: %s -->' % (idx,' '.join(map(str,linked_images)))

def get_focal_length_35mm(fname):
	tags=exif.read_exif(fname)
	focal_length=exif.exif_focal_length(tags)
	if focal_length is None or focal_length < 1e-6:
		return None

	sensor_x_mm,sensor_y_mm=exif.exif_sensor_size_mm(tags)

	focal_multiplier=math.sqrt(36**2 + 24**2) / math.sqrt(sensor_x_mm**2 + sensor_y_mm**2)

	return focal_length * focal_multiplier

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
		Extract and show keypoints from a single image

	%s [--output-fname=PANO-XML-FNAME] IMAGE-FNAME IMAGE-FNAME [IMAGE-FNAME ...]
		Match a set of images to each other, and write the resulting AutoPano XML file to stdout or output file

	%s --testcase-fname=PANO-XML-FNAME ... [--print-training-data] [--nowarn] MATCHES-XML-FNAME ...
		Optimise the matches classifier (decision_value formula) against testcases, using raw matches files as input''' % \
		((sys.argv[0],) * 3)

	exit(1)

correct_matches=None
testcase_fnames=keyword_args.get('--testcase-fname')
if testcase_fnames:
	if not isinstance(testcase_fnames,list):
		testcase_fnames=(testcase_fnames,)

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

	for testcase_fname in testcase_fnames:
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
			scores.append((calc_classifier_decision_value(e[1:],params),e[0]))
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

		return (best_correct_predictions,'%+.3f' % (best_threshold,))

	tries=0
	nonzero_tries,nonzero_successes=0,0
	correct_preditions_with_zero_score=0

	for matches_name in positional_args:
		image_fnames=[]
		for line in open(matches_name,'r'):
			if line.strip().endswith(' keypoints') or line.strip().endswith(' keypoints -->'):
				image_fnames.append(line.rpartition('/')[2].partition(' ')[0])
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
				if '--nowarn' not in keyword_args:
					print 'Warning: image pair %s %s not present in testcases' % fnames_pair
				continue

			if score > 0:
				shift_ratio=calc_shift_ratio(x_shift,y_shift)
				training_data.append((int(is_correct_match),score,count,min(50,abs(angle_deg)),shift_ratio))

				if print_training_data:
					print '%d 1:%s 2:%s 3:%s 4:%s' % training_data[-1]

				decision_value=calc_classifier_decision_value(
										(score,count,min(50,abs(angle_deg)),shift_ratio),classifier_params)

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
		print 'Successes: %u/%u %.2f%% (hardcoded params                %s and threshold %+.3f)' % (
								nonzero_successes + correct_preditions_with_zero_score,
								tries,(nonzero_successes + correct_preditions_with_zero_score)*100.0/tries,
								' '.join(map(str,classifier_params[:-1])),classifier_params[-1])

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

elif len(positional_args) == 1:
	ImageKeypoints(positional_args[0]).show_img_with_keypoints(0)
else:
	image_fnames=positional_args
	images_with_keypoints=[ImageKeypoints(fname,True) for fname in image_fnames]

	output_fd=sys.stdout
	output_fname=keyword_args.get('--output-fname')
	if output_fname:
		output_fd=open(output_fname,'wt')

	print >>output_fd,'''<?xml version="1.0" encoding="UTF-8"?>
<pano>
    <version filemodel="2.0" application="Autopano Pro 4.4.1" id="9"/>
    <finalRender basename="%a" path="%p" fileType="jpg" fileCompression="5" fileDepth="8" interpolationMode="3" blendMode="2" outputPercent="100" overwrite="1" fileEmbedAll="0" removeAlpha="0" outputPanorama="1" outputLayers="0" outputPictures="0" multibandLevel="-2" alphaDiamond="0" exposureWeights="0" cutting="1" graphcutGhostFocal="0" bracketedGhost="0"/>
    <optim>
        <options automaticSteps="0" automaticSettings="0" focalHandling="-1" distoHandling="-1" offsetsHandling="-1" multipleVPHandling="0" yprScope="0" focalScope="2" distoScope="2" offsetScope="2" hScope="0" optLG="1" useGO="1" matrixLG="0" gridLG="0" optFinal1="1" optLens1="1" optLens2="1" stepGeomAnalysis="1" cleanPoints="0" cleanLinks="0" cbpMode="0" cbpThreshold="5" cbpLimit="50" cbpLinkThreshold="40" matLGMode="1" matLGRow="1" matLGStack="1" matLG360="0" matLGOverlapping="25" gaMode="0" calibratedRig="0"/>
    </optim>
    <colorCorrection eqMode="1" eqPerLayer="1" colorDtScaler="1"/>
    <exposureWeighting enabled="0" tone="0.5" dark="0.5" light="0.5"/>
    <projection fitMode="1" type="0">
        <params/>
    </projection>
    <images>
'''

	for fname,img in zip(image_fnames,images_with_keypoints):
		print >>output_fd,('<image><def filename="%s" focal35mm="%.3f" lensModel="0" ' + \
										'fisheyeRadius="0" fisheyeCoffX="0" fisheyeCoffY="0"/></image>') % \
									(fname,get_focal_length_35mm(fname) or 0)
		print >>output_fd,'<!-- %s %s keypoints -->' % (fname,'+'.join(map(str,img.channel_keypoints)))

	print >>output_fd,'''
    </images>
    <layers>
        <layer name="N_0" ouput="1">
            <images>
'''

	for idx in range(len(images_with_keypoints)):
		print >>output_fd,'                <image index="%d" preview="1" output="1"/>' % (idx,)

	print >>output_fd,'''
            </images>
        </layer>
    </layers>
    <stacks>
'''

	for idx in range(len(images_with_keypoints)):
		print >>output_fd,'        <stack>%d</stack>' % (idx,)

	print >>output_fd,'''
    </stacks>
    <matches>
'''
	print_matches_for_images(output_fd)

	print >>output_fd,'''
    </matches>
</pano>
'''

# img1.show_img_with_keypoints([matches[xy_pair[4]].queryIdx for xy_pair in representative_xy_pairs])
# if len(sys.argv) >= 1+3:
#	cv2.imwrite(sys.argv[3],img1.img)
