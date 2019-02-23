# -*- encoding: latin-1 -*-

import sys,math,operator,cPickle,decimal,numpy,cv2,exif,boto3,sky_detection
from quaternion import Quaternion

RESIZE_FACTOR=4
KEYPOINT_BLOCKS=5
IMG_HISTOGRAM_SIZE=10

S3_KEYPOINTS_BUCKET_NAME='image-keypoints'
DYNAMODB_TABLE_NAME='panorama-match-records'

aws_session=None

detector_patch_size=31
detector=cv2.ORB_create(nfeatures=1000,patchSize=detector_patch_size)

keypoint_matcher=cv2.FlannBasedMatcher({'algorithm': 6, 'table_number': 6, 'key_size': 12,
										'multi_probe_level': 1},{'checks': 50})
clahe=cv2.createCLAHE(clipLimit=40,tileGridSize=(16,16))

classifier_params=(0.02,0.68,-0.21,-0.4,-0.63,-10.542)

class ImageKeypoints:
	class Keypoints:
		def __init__(self,img=None,focal_length_pixels=None,x1=None,x2=None,y1=None,y2=None,mask=None):
			self.descriptors=None
			self.xys=[]

			if img is None:
				return

			kp,self.descriptors=detector.detectAndCompute(img[y1:y2,x1:x2],mask[y1:y2,x1:x2])

			if self.descriptors is not None:
				base_x=x1 - img.shape[1]/2.0
				base_y=y1 - img.shape[0]/2.0
				for keypoint in kp:
					self.xys.append((
								math.degrees(math.atan2(base_x + keypoint.pt[0],focal_length_pixels)),
								math.degrees(math.atan2(base_y + keypoint.pt[1],focal_length_pixels))))

		def __iadd__(self,kp):
			if kp.descriptors is not None:
				if self.descriptors is None:
					self.descriptors=kp.descriptors
				else:
					self.descriptors=numpy.concatenate((self.descriptors,kp.descriptors))

			self.xys.extend(kp.xys)
			return self

	def __init__(self,fname,deallocate_image=False,from_s3=False):
		if from_s3:
			self.load_from_s3(fname)
			return

		self.img=cv2.imread(fname)
		self.orig_img_shape=tuple(self.img.shape)[:2]

		self.focal_length_35mm=get_focal_length_35mm(fname)

		if RESIZE_FACTOR != 1:
			self.img=cv2.resize(self.img,(0,0),fx=1.0 / RESIZE_FACTOR,fy=1.0 / RESIZE_FACTOR)

		self.sky_mask=cv2.threshold(sky_detection.calc_image_skyness(self.img),
																		128,255,cv2.THRESH_BINARY_INV)[1]

		tags=exif.read_exif(fname)
		focal_length_mm=exif.exif_focal_length(tags)
		if focal_length_mm is None or focal_length_mm < 1e-6:
			focal_length_mm=20		# Assume 20mm lens by default

		sensor_size_mm=exif.exif_sensor_size_mm(tags)

		self.x_fov_deg=2 * math.degrees(math.atan2(sensor_size_mm[0] / 2.0,focal_length_mm))
		self.y_fov_deg=2 * math.degrees(math.atan2(sensor_size_mm[1] / 2.0,focal_length_mm))

		focal_length_ratio=focal_length_mm / float(sum(sensor_size_mm))
		self.focal_length_orig_pixels   =sum(self.orig_img_shape      ) * focal_length_ratio
		self.focal_length_resized_pixels=sum(tuple(self.img.shape)[:2]) * focal_length_ratio

		# self.img=cv2.Laplacian(self.img,cv2.CV_8U,ksize=5)	# somewhat works
		# self.img=cv2.Canny(self.img,10,20)

		self.channels=[]

		self.add_channel(self.img)

		b,g,r=cv2.split(self.img)
		self.add_channel(cv2.subtract(
							cv2.add(cv2.transform(b,numpy.array((0.5,))),
									128),
							cv2.transform(r,numpy.array((0.5,)))))

			##### Build self.coverage_areas[] #####

		total_keypoints=0
		keypoint_counts=[[0 for i in range(IMG_HISTOGRAM_SIZE)] for j in range(IMG_HISTOGRAM_SIZE)]
		x_bin_size=self.x_fov_deg / float(IMG_HISTOGRAM_SIZE)
		y_bin_size=self.y_fov_deg / float(IMG_HISTOGRAM_SIZE)
		for chan in self.channels:
			total_keypoints+=len(chan.xys)
			for x_deg,y_deg in chan.xys:
				keypoint_counts [max(0,min(IMG_HISTOGRAM_SIZE-1,int(math.floor(
														(x_deg + self.x_fov_deg/2.0) / x_bin_size))))] \
								[max(0,min(IMG_HISTOGRAM_SIZE-1,int(math.floor(
														(y_deg + self.y_fov_deg/2.0) / y_bin_size))))]+=1
		self.coverage_areas=[]
		for x_idx in range(IMG_HISTOGRAM_SIZE):
			x_deg=(x_idx + 0.5) * x_bin_size - self.x_fov_deg/2.0
			for y_idx in range(IMG_HISTOGRAM_SIZE):
				y_deg=(y_idx + 0.5) * y_bin_size - self.y_fov_deg/2.0
				if keypoint_counts[x_idx][y_idx]:
					self.coverage_areas.append((x_deg,y_deg,
												keypoint_counts[x_idx][y_idx] / float(total_keypoints)))

		self.channel_keypoints=tuple([len(chan.xys) for chan in self.channels])

		if deallocate_image:
			self.img=None

	def add_channel(self,img):
		global KEYPOINT_BLOCKS,detector_patch_size

		if len(img.shape) >= 3:
			img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

		img=clahe.apply(img)

		x_splits=tuple([i*img.shape[1]/KEYPOINT_BLOCKS for i in range(KEYPOINT_BLOCKS+1)])
		y_splits=tuple([i*img.shape[0]/KEYPOINT_BLOCKS for i in range(KEYPOINT_BLOCKS+1)])

		self.channels.append(ImageKeypoints.Keypoints())

		for x_idx in range(len(x_splits)-1):
			for y_idx in range(len(y_splits)-1):
				self.channels[-1]+=ImageKeypoints.Keypoints(img,self.focal_length_resized_pixels,
						x_splits[x_idx],min(img.shape[1],x_splits[x_idx+1] + 2*detector_patch_size),
						y_splits[y_idx],min(img.shape[0],y_splits[y_idx+1] + 2*detector_patch_size),
						self.sky_mask)

	def degrees_to_pixels(self,x_deg,y_deg):
		return (int(self.orig_img_shape[1] / 2.0 + math.tan(math.radians(x_deg)) * \
																			self.focal_length_orig_pixels),
				int(self.orig_img_shape[0] / 2.0 + math.tan(math.radians(y_deg)) * \
																			self.focal_length_orig_pixels))

	def calc_keypoints_coverage(self,img2,angle_sin,angle_cos,x_add_deg,y_add_deg):
		coverage_sum=0

		for area_x_deg,area_y_deg,keypoints_coverage in self.coverage_areas:
			x=area_x_deg * angle_cos - area_y_deg * angle_sin + x_add_deg
			if abs(x) > 0.5 * img2.x_fov_deg:
				continue
			y=area_x_deg * angle_sin + area_y_deg * angle_cos + y_add_deg
			if abs(y) > 0.5 * img2.y_fov_deg:
				continue

			coverage_sum+=keypoints_coverage

		return coverage_sum

	def show_img_with_keypoints(self,channel_idx,highlight_indexes=tuple()):
		for idx,xy_deg in enumerate(self.channels[channel_idx].xys):
			highlight=(idx in highlight_indexes)
			color=(255,0,0) if highlight else (0,255,0)
			cv2.circle(self.img,self.degrees_to_pixels(*xy_deg),
												(15 if highlight else 10) / RESIZE_FACTOR,color,-1)
		import matplotlib.pyplot
		matplotlib.pyplot.imshow(self.img)
		matplotlib.pyplot.show()

	def save_to_s3(self,fname):
		obj_to_pickle=[self.focal_length_35mm,self.focal_length_orig_pixels,self.x_fov_deg,self.y_fov_deg,
															self.orig_img_shape,self.coverage_areas] + \
						[(chan.descriptors,chan.xys) for chan in self.channels]
		pickle_result=cPickle.dumps(obj_to_pickle,-1)

		s3=aws_session.resource('s3')
		s3.Bucket(S3_KEYPOINTS_BUCKET_NAME).put_object(Key=fname,Body=pickle_result)

	def load_from_s3(self,fname):
		s3_bucket=aws_session.resource('s3').Bucket(S3_KEYPOINTS_BUCKET_NAME)
		obj_from_pickle=cPickle.loads(s3_bucket.Object(fname).get()['Body'].read())

		self.focal_length_35mm,self.focal_length_orig_pixels,self.x_fov_deg,self.y_fov_deg, \
												self.orig_img_shape,self.coverage_areas=obj_from_pickle[:6]
		self.channels=[]
		for pickled_channel in obj_from_pickle[6:]:
			chan=ImageKeypoints.Keypoints()
			chan.descriptors,chan.xys=pickled_channel
			self.channels.append(chan)

		self.channel_keypoints=tuple([len(chan.xys) for chan in self.channels])

def calc_shift_for_angle(img1,img2,matches,angle_deg):
	angle_sin=math.sin(math.radians(angle_deg))
	angle_cos=math.cos(math.radians(angle_deg))

	histogram_bin_degrees=0.7

	xy_deltas=[]
	histogram=dict()
	for distance,x1,y1,x2,y2 in matches[:1000]:
		xd=x1 - (x2 * angle_cos - y2 * angle_sin)
		yd=y1 - (x2 * angle_sin + y2 * angle_cos)

		# print 'ZZZ',xd,yd,distance

		idx=(int(math.floor(xd / histogram_bin_degrees)),int(math.floor(yd / histogram_bin_degrees)))
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

		img2_keypoints_coverage=max(0.1,img2.calc_keypoints_coverage(img1,angle_sin,angle_cos,
											idx[0] * histogram_bin_degrees,idx[1] * histogram_bin_degrees))

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

	xd=xd_sum / float(len(inliers))
	yd=yd_sum / float(len(inliers))

	best_count+=len(inliers)

	# print angle_deg,best_count,best_coverage,xd,yd

	return (best_count,best_coverage,inliers,xd,yd)

def calc_classifier_decision_value(inputs,params):
	decision_value=sum(value*weight for value,weight in zip(inputs,params))
	if len(params) > len(inputs):
		decision_value-=params[-1]
	return decision_value

def calc_shift_ratio(xd,yd):
	abs_shifts=(abs(xd),abs(yd))
	return min(abs_shifts) / float(max(1,max(abs_shifts)))

def find_matches(img1,img2):
	global keypoint_matcher

	matches=[]
	for chan1,chan2 in zip(img1.channels,img2.channels):
		for match_pair in keypoint_matcher.knnMatch(chan1.descriptors,chan2.descriptors,k=2):
			if len(match_pair) == 2:
				m,m2=match_pair
				if m.distance < 0.8 * m2.distance:
					matches.append((m.distance,	chan1.xys[m.queryIdx][0],chan1.xys[m.queryIdx][1],
												chan2.xys[m.trainIdx][0],chan2.xys[m.trainIdx][1]))

	matches.sort(key=operator.itemgetter(0))

	debug_str='%d matches' % len(matches)

	if not matches:
		return (debug_str,)

	debug_str+=', distances %.0f:%.0f' % (matches[0][0],matches[:30][-1][0])

	best_angle_deg=0
	best_score=0
	best_count=0
	best_coverage=1
	best_inliers=None	# In increasing matches[] index order
	best_xd=0
	best_yd=0

	if matches[0][0] < 30:
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

	debug_str+=', %+ddeg, score %d/%.2f=%d, shift %+.0fdeg,%+.0fdeg' % (best_angle_deg,best_count,
															best_coverage,best_score,best_xd,best_yd)
	if best_score <= 0 or not best_inliers:
		return (debug_str,)

	# src_pts=numpy.float32([(x1,y1) for distance,x1,y1,x2,y2 in matches[:1000]]).reshape(-1,1,2)
	# dst_pts=numpy.float32([(x2,y2) for distance,x1,y1,x2,y2 in matches[:1000]]).reshape(-1,1,2)
	# homography_matrix=cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0)[0]
	# print homography_matrix

	max_dist_square=((img1.x_fov_deg + img1.y_fov_deg) / 50.0) ** 2
	representative_xy_pairs=[]
	matched_points=[]
	for i in best_inliers:
		xy=matches[i][1:3]

		for xy_pair in representative_xy_pairs:
			if (xy[0] - xy_pair[0])**2 + (xy[1] - xy_pair[1])**2 < max_dist_square:
				break
		else:
			representative_xy_pairs.append(xy + matches[i][3:] + (i,))
			matched_points.append((img1.degrees_to_pixels(*xy) + img2.degrees_to_pixels(*matches[i][3:])))
			if len(representative_xy_pairs) >= 50:
				break

	# img1.show_img_with_keypoints([matches[xy_pair[4]].queryIdx for xy_pair in representative_xy_pairs])

	return (debug_str,matched_points,best_score,best_count,best_angle_deg,best_xd,best_yd)

def write_dynamodb_item(item):
	global DYNAMODB_TABLE_NAME,aws_session

	table=aws_session.resource('dynamodb').Table(DYNAMODB_TABLE_NAME)
	table.put_item(Item=item)

def process_match_and_write_to_dynamodb(processing_batch_key,s3_fname1,s3_fname2,
																		orig_fname1=None,orig_fname2=None):
	global S3_KEYPOINTS_BUCKET_NAME

	img1=ImageKeypoints(s3_fname1,False,S3_KEYPOINTS_BUCKET_NAME)
	img2=ImageKeypoints(s3_fname2,False,S3_KEYPOINTS_BUCKET_NAME)

	result=find_matches(img1,img2)

	item={'processing_batch_key': processing_batch_key,
			's3_filenames': s3_fname1 + '_' + s3_fname2,
			'debug_str': result[0],
			'img1_fname': orig_fname1 or s3_fname1,
			'img2_fname': orig_fname2 or s3_fname2,
			'img1_focal_length_35mm': decimal.Decimal(str(img1.focal_length_35mm)),
			'img2_focal_length_35mm': decimal.Decimal(str(img2.focal_length_35mm)),
			'img1_focal_length_pixels': decimal.Decimal(str(img1.focal_length_orig_pixels)),
			'img2_focal_length_pixels': decimal.Decimal(str(img2.focal_length_orig_pixels)),
			'img1_width': decimal.Decimal(str(img1.orig_img_shape[1])),
			'img1_height': decimal.Decimal(str(img1.orig_img_shape[0])),
			'img2_width': decimal.Decimal(str(img2.orig_img_shape[1])),
			'img2_height': decimal.Decimal(str(img2.orig_img_shape[0])),
			'img1_channel_keypoints': list(img1.channel_keypoints),
			'img2_channel_keypoints': list(img2.channel_keypoints)
			}

	if len(result) > 1:
		item['matched_points']=map(list,result[1])

		for idx,attr_name in enumerate(('score','count','angle_deg','xd','yd')):
			item[attr_name]=decimal.Decimal(str(result[2 + idx]))

	write_dynamodb_item(item)

def get_focal_length_35mm(fname):
	tags=exif.read_exif(fname)
	focal_length=exif.exif_focal_length(tags)
	if focal_length is None or focal_length < 1e-6:
		return None

	sensor_x_mm,sensor_y_mm=exif.exif_sensor_size_mm(tags)

	focal_multiplier=math.sqrt(36**2 + 24**2) / math.sqrt(sensor_x_mm**2 + sensor_y_mm**2)

	return focal_length * focal_multiplier

def init_aws_session(profile_name=None):
	global aws_session

	aws_session=boto3.Session(profile_name=profile_name)

def quaternion_from_match_angles(angle_deg,x_shift,y_shift):
	# Output quaternion coordinate system is such that X points ahead, Y left, Z up
	x=Quaternion.from_single_axis_angle_deg(0,-angle_deg)
	y=Quaternion.from_single_axis_angle_deg(1,-y_shift)
	z=Quaternion.from_single_axis_angle_deg(2,+x_shift)
	return y*z*x		#!!! Probably in correct order

def quaternion_from_kolor_file(yaw_rad,pitch_rad,roll_rad):
	# Output quaternion coordinate system is such that X points ahead, Y left, Z up
	x=Quaternion.from_single_axis_angle(0,+roll_rad)
	y=Quaternion.from_single_axis_angle(1,-pitch_rad)
	z=Quaternion.from_single_axis_angle(2,-yaw_rad)
	return x*y*z

def calc_triplet_scores(matches):
	# Input: matches[image_ids_pair]=(quaternion,match_metrics)

	decision_values=dict()
	for image_ids_pair,(q,match_metrics) in matches.items():
		score,count,angle_deg,xd,yd=match_metrics
		shift_ratio=calc_shift_ratio(xd,yd)
		decision_values[image_ids_pair]=calc_classifier_decision_value(
					(score,count,min(50,abs(angle_deg)),shift_ratio,10),classifier_params)	#!!! Tune this 10

	triplet_scores=dict()
	for image_ids_pair1,(q1,match_metrics1) in matches.items():
		image_id1=image_ids_pair1[0]
		image_id2=image_ids_pair1[1]
		if image_id1 > image_id2:
			continue
		for image_ids_pair2,(q2,match_metrics2) in matches.items():
			if not (image_id2 in image_ids_pair2 and image_id1 not in image_ids_pair2):
				continue

			reverse_pair2=(image_id2 == image_ids_pair2[1])
			image_id3=image_ids_pair2[0 if reverse_pair2 else 1]

			if image_id2 > image_id3:
				continue

			if reverse_pair2:
				q2=q2.conjugate()

			for reverse_pair3 in (False,True):
				image_ids_pair3=(image_id3,image_id1) if reverse_pair3 else (image_id1,image_id3)
				if image_ids_pair3 not in matches:
					continue

				q3,match_metrics3=matches[image_ids_pair3]

				if reverse_pair3:
					q3=q3.conjugate()

				error_deg=(q1 * q2).rotation_to_b(q3).total_rotation_angle_deg()

				pairs=(image_ids_pair1,image_ids_pair2,image_ids_pair3)
				for pair in pairs:
					triplet_decision_value=min([decision_values[p] for p in pairs if p != pair])
					if triplet_decision_value > 20:		#!!! This threshold should be found automatically
						if pair not in triplet_scores or triplet_decision_value > triplet_scores[pair][1]:
							triplet_scores[pair]=(error_deg,triplet_decision_value)

	return triplet_scores

def write_output_file_matches(output_fd,matches,nr_of_images):
	triplets_input=dict()

	for idx1,idx2,debug_str,output_string,match_metrics in matches:
		if output_string:
			score,count,angle_deg,xd,yd=match_metrics
			detected_match_rot=quaternion_from_match_angles(angle_deg,xd,yd)
			triplets_input[(idx1,idx2)]=(detected_match_rot,match_metrics)

	triplet_scores=calc_triplet_scores(triplets_input)

	link_stats=[[] for i in range(nr_of_images)]

	for idx1,idx2,debug_str,output_string,match_metrics in matches:
		print >>output_fd,'        <!-- image %d<-->%d: %s -->' % (idx1,idx2,debug_str)

		if not output_string:
			continue

		score,count,angle_deg,xd,yd=match_metrics
		triplet_score=triplet_scores.get((idx1,idx2),(30,-1000))[0]		#!!! Tune this

		shift_ratio=calc_shift_ratio(xd,yd)
		decision_value=calc_classifier_decision_value(
							(score,count,min(50,abs(angle_deg)),shift_ratio,triplet_score),classifier_params)
		if decision_value < 0:
			continue

		print >>output_fd,'        <match image1="%d" image2="%d">\n            <points>\n%s            </points>\n        </match>' % \
																				(idx1,idx2,output_string)
		link_stats[idx1].append(idx2)
		link_stats[idx2].append(idx1)

	print >>output_fd,'<!-- Link stats: -->'

	for idx,linked_images in enumerate(link_stats):
		print >>output_fd,'<!-- #%d links: %s -->' % (idx,' '.join(map(str,linked_images)))

def write_output_file_header(output_fd):
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

def write_output_file_image(output_fd,fname,focal_length_35mm,channel_keypoints,
													kolor_file_angles_rad=None,focal_length_pixels=None):
	print >>output_fd,'<image>'

	print >>output_fd,('     <def filename="%s" focal35mm="%.3f" lensModel="0" ' + \
										'fisheyeRadius="0" fisheyeCoffX="0" fisheyeCoffY="0"/>') % \
									(fname,focal_length_35mm or 0)
	if kolor_file_angles_rad is not None and focal_length_pixels is not None:
		print >>output_fd,'     <camera yaw="%.5f" pitch="%.5f" roll="%.5f" f="%.2f"/>' % \
														(kolor_file_angles_rad + (focal_length_pixels,))
	print >>output_fd,'</image>'

	print >>output_fd,'<!-- %s %s keypoints -->' % (fname,'+'.join(map(str,channel_keypoints)))

def write_output_file_midsection(output_fd,nr_of_images):
	print >>output_fd,'''
    </images>
    <layers>
        <layer name="N_0" ouput="1">
            <images>
'''

	for idx in range(nr_of_images):
		print >>output_fd,'                <image index="%d" preview="1" output="1"/>' % (idx,)

	print >>output_fd,'''
            </images>
        </layer>
    </layers>
    <stacks>
'''

	for idx in range(nr_of_images):
		print >>output_fd,'        <stack>%d</stack>' % (idx,)

	print >>output_fd,'''
    </stacks>
    <matches>
'''

def write_output_file_footer(output_fd):
	print >>output_fd,'''
    </matches>
</pano>
'''
