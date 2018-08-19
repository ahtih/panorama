#!/usr/bin/python

import sys,os,math,operator,time,calendar
import exif

images=[]

if len(sys.argv) <= 1:
	print 'Usage: find_panoramas.py IMAGE-FNAME ...'
	print
	exit(1)

for fname in sys.argv[1:]:
	tags=exif.read_exif(fname)

#	for k,v in tags.items():
#		value_str=str(v)
#		if len(value_str) > 100:
#			value_str='<%d chars>' % len(value_str)
#		print '%s: %s' % (k,value_str)

	timestamp=None
	for time_tag in ('Image DateTime','EXIF DateTimeOriginal','EXIF DateTimeDigitized'):
		value=tags.get(time_tag)
		if not value:
			continue
		timestamp=calendar.timegm(time.strptime(str(value),'%Y:%m:%d %H:%M:%S'))
		break

	fov_sq_deg=2506		# Assume 20mm focal length on APS-C sensor

	focal_length=exif.exif_focal_length(tags)
	if focal_length is not None and focal_length > 1e-6:
		sensor_x_mm,sensor_y_mm=exif.exif_sensor_size_mm(tags)

		x_degrees=math.degrees(2*math.atan(0.5*sensor_x_mm/focal_length))
		y_degrees=math.degrees(2*math.atan(0.5*sensor_y_mm/focal_length))

		fov_sq_deg=x_degrees * y_degrees

	images.append((timestamp,fname,tags,fov_sq_deg / float(360*180)))

images.sort(key=operator.itemgetter(0))

def delta_time(image_idx):
	global images

	if image_idx < 1:
		return 1e9

	timestamp=images[image_idx][0]
	last_timestamp=images[image_idx-1][0]

	if timestamp is None or last_timestamp is None:
		return None

	return timestamp - last_timestamp

panoramas=[]

cur_panorama_start_idx=0
cur_sphere_coverage_sum=0
for idx,(timestamp,fname,tags,sphere_coverage) in enumerate(images):
	print '#',idx,fname,delta_time(idx),sphere_coverage	#!!!

	if idx >= 1:
		if delta_time(idx) > 90 * min(1.0,3.0 / max(0.01,cur_sphere_coverage_sum)):
			panoramas.append((cur_panorama_start_idx,idx-1))
			cur_panorama_start_idx=idx
			cur_sphere_coverage_sum=0

	cur_sphere_coverage_sum+=sphere_coverage

panoramas.append((cur_panorama_start_idx,len(images)-1))

for panorama_idx in reversed(range(len(panoramas))):
	if panoramas[panorama_idx][1] <= panoramas[panorama_idx][0]:
		del panoramas[panorama_idx]

def calc_panorama_data(start_idx,end_idx):
	global images

	sphere_coverage=images[start_idx][3]
	max_time_delta=0
	for idx in range(start_idx+1,end_idx + 1):
		sphere_coverage+=images[idx][3]
		max_time_delta=max(max_time_delta,delta_time(idx))

	return (end_idx + 1 - start_idx,sphere_coverage,max_time_delta)

def split_panorama(start_idx,end_idx):
	max_time_delta=0
	split_idx=start_idx
	for idx in range(start_idx+1,end_idx + 1):
		if max_time_delta <= delta_time(idx):
			max_time_delta=delta_time(idx)
			split_idx=idx

	if split_idx - start_idx < 2 or end_idx+1-split_idx < 2:
		return None

	sphere_coverage1,max_time_delta1=calc_panorama_data(start_idx,split_idx-1)[1:]
	sphere_coverage2,max_time_delta2=calc_panorama_data(split_idx,end_idx)[1:]

	if min(sphere_coverage1,sphere_coverage2) < 1.7 or \
												max(max_time_delta1,max_time_delta2)*2 > max_time_delta:
		return None

	return ((start_idx,split_idx-1),(split_idx,end_idx))

panorama_idx=0
while panorama_idx < len(panoramas):
	start_idx,end_idx=panoramas[panorama_idx]
	nr_of_images,sphere_coverage,max_time_delta=calc_panorama_data(start_idx,end_idx)
	if max_time_delta >= 20 and sphere_coverage >= max(2.0,4.2 - max_time_delta/80.0):
		split_result=split_panorama(start_idx,end_idx)
		if split_result is not None:
			del panoramas[panorama_idx]
			panoramas[panorama_idx:panorama_idx]=split_result
			panorama_idx+=2
			continue
	panorama_idx+=1

non_panorama_image_indexes=set(range(len(images)))

for panorama_idx,(start_idx,end_idx) in enumerate(panoramas):
	nr_of_images,sphere_coverage,max_time_delta=calc_panorama_data(start_idx,end_idx)
	# print start_idx,end_idx,nr_of_images,max_time_delta,sphere_coverage,images[start_idx][1],images[end_idx][1]

	if sphere_coverage < 0.5:
		continue

	print 'echo %d..%d %d %.0fsec %.2f' % (start_idx,end_idx,nr_of_images,max_time_delta,sphere_coverage)

	print './panorama.py --output-fname=panorama-%s.pano %s' % (
							os.path.splitext(os.path.basename(images[start_idx][1]))[0].replace('IMG_',''),
							' '.join([images[idx][1] for idx in range(start_idx,end_idx+1)]))
	non_panorama_image_indexes-=frozenset(range(start_idx,end_idx+1))

if non_panorama_image_indexes:
	print
	print 'echo Probable non-panorama images:'
	for idx in sorted(non_panorama_image_indexes):
		print 'echo',images[idx][1]

#Image Model
#EXIF ExposureTime
#EXIF ISOSpeedRatings
#EXIF FNumber
#EXIF FocalLength: 11
