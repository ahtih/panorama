import exifread

exif_resolution_units_mm={2: 25.4, 3: 10}

def read_exif(fname):
	fd=open(fname,'rb')
	tags=exifread.process_file(fd,details=False)
	fd.close()

	return tags

def exif_to_float(exif_value):
	if exif_value is None:
		return None
	exif_value=str(exif_value)
	if '/' in exif_value:
		fields=map(float,exif_value.split('/'))
		return fields[0] / fields[1]
	else:
		return float(exif_value)

def exif_focal_length(tags):
	return exif_to_float(tags.get('EXIF FocalLength'))

def exif_sensor_size_mm(tags):
	sensor_x_mm=25.1	# Assume APS-C if no sensor size is specified
	sensor_y_mm=16.7
	resolution_unit_mm=exif_resolution_units_mm.get(int(str(tags.get('EXIF FocalPlaneResolutionUnit','0'))))

	x_res=exif_to_float(tags.get('EXIF FocalPlaneXResolution'))
	y_res=exif_to_float(tags.get('EXIF FocalPlaneYResolution'))
	x_size=exif_to_float(tags.get('EXIF ExifImageWidth'))
	y_size=exif_to_float(tags.get('EXIF ExifImageLength'))

	if resolution_unit_mm is not None and x_res is not None and y_res is not None and \
									x_size is not None and y_size is not None and x_res > 0 and y_res > 0:
		sensor_x_mm=resolution_unit_mm * x_size / x_res
		sensor_y_mm=resolution_unit_mm * y_size / y_res

	return (sensor_x_mm,sensor_y_mm)
