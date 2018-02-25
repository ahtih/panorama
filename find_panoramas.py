#!/usr/bin/python

import sys,math,operator,time,calendar,exifread

exif_resolution_units_mm={2: 25.4, 3: 10}

images=[]

def exif_to_float(exif_value):
	if exif_value is None:
		return None
	exif_value=str(exif_value)
	if '/' in exif_value:
		fields=map(float,exif_value.split('/'))
		return fields[0] / fields[1]
	else:
		return float(exif_value)

for fname in sys.argv[1:]:
	fd=open(fname,'rb')
	tags=exifread.process_file(fd,details=False)
	fd.close()

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

	focal_length=exif_to_float(tags.get('EXIF FocalLength'))
	if focal_length is not None and focal_length > 1e-6:
		sensor_x_mm=25.1	# Assume APS-C if no sensor size is specified
		sensor_y_mm=16.7
		resolution_unit_mm=exif_resolution_units_mm.get(
												int(str(tags.get('EXIF FocalPlaneResolutionUnit','0'))))
		x_res=exif_to_float(tags.get('EXIF FocalPlaneXResolution'))
		y_res=exif_to_float(tags.get('EXIF FocalPlaneYResolution'))
		x_size=exif_to_float(tags.get('EXIF ExifImageWidth'))
		y_size=exif_to_float(tags.get('EXIF ExifImageLength'))
		if resolution_unit_mm is not None and x_res is not None and y_res is not None and \
									x_size is not None and y_size is not None and x_res > 0 and y_res > 0:
			sensor_x_mm=resolution_unit_mm * x_size / x_res
			sensor_y_mm=resolution_unit_mm * y_size / y_res

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

for panorama_idx,(start_idx,end_idx) in enumerate(panoramas):
	nr_of_images,sphere_coverage,max_time_delta=calc_panorama_data(start_idx,end_idx)
	# print start_idx,end_idx,nr_of_images,max_time_delta,sphere_coverage,images[start_idx][1],images[end_idx][1]

	print 'echo %d..%d %d %.0fsec %.2f' % (start_idx,end_idx,nr_of_images,max_time_delta,sphere_coverage)
	print './panorama.py --output-fname=panorama-%d.pano %s' % (panorama_idx,
										' '.join([images[idx][1] for idx in range(start_idx,end_idx+1)]))

#Image Model
#EXIF ExposureTime
#EXIF ISOSpeedRatings
#EXIF FNumber
#EXIF FocalLength: 11
