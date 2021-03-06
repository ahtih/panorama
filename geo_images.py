#!/usr/bin/python3

import math,random,pickle

EARTH_RADIUS_METERS=6371e3

geo_images=dict()			# [id]=(lon_deg,lat_deg,center_azimuth_deg,owner_uid,sharing_mode)

def is_valid_image_id(id):
	return id and not set(id) - set('0123456789')

def load_geo_images_list(kml_path='.'):
	import fastkml,shapely

	def process_KML_feature(e):
		global geo_images

		if hasattr(e,'features'):
			for f in e.features():
				process_KML_feature(f)
			return

		if not hasattr(e,'geometry'):
			return

		if not isinstance(e.geometry,shapely.geometry.point.Point) or not is_valid_image_id(e.name):
			return

		if not hasattr(e,'description'):
			return

		if not e.description:
			return

		fields=e.description.split()
		if len(fields) < 1:
			return

		center_azimuth_deg=float(fields[0].partition('deg')[0])

		owner_uid=0
		sharing_mode=None

		if len(fields) >= 2:
			owner_uid=int(fields[1])
		if len(fields) >= 3:
			sharing_mode=fields[2]

		geo_images[e.name]=tuple(e.geometry.coords[0][:2]) + (center_azimuth_deg,owner_uid,sharing_mode)

	k=fastkml.KML()
	k.from_string(open(kml_path + '/images.kml','r').read().encode('utf-8'))
	process_KML_feature(k)

def load_from_pickle(pickled_data):
	global geo_images

	geo_images=pickle.loads(pickled_data)

def dot_product(v1,v2):
	return sum([a*b for a,b in zip(v1,v2)])

def vec_len(vec):
	return math.sqrt(sum([val*val for val in vec]))

def cross_product(v1,v2):
	return (v1[1]*v2[2] - v1[2]*v2[1],
			v1[2]*v2[0] - v1[0]*v2[2],
			v1[0]*v2[1] - v1[1]*v2[0])

def calc_geo_distance_deg(lon1,lat1,lon2,lat2):
	lon_weight=math.cos(math.radians(0.5*(lat1 + lat2)))
	return math.sqrt(((lon1 - lon2)*lon_weight) ** 2 + (lat1 - lat2) ** 2)

def calc_geo_distance_and_azimuth(lon1,lat1,lon2,lat2):
	global EARTH_RADIUS_METERS

	rel_lon2=lon2 - lon1

	loc1=(0,-math.cos(math.radians(lat1)),math.sin(math.radians(lat1)))
	loc2=(	+math.cos(math.radians(lat2))*math.sin(math.radians(rel_lon2)),
			-math.cos(math.radians(lat2))*math.cos(math.radians(rel_lon2)),
			+math.sin(math.radians(lat2)))

	longitude_plane_normal=(-1,0,0)

	cross_product_vec=cross_product(loc1,loc2)
	cross_product_vec_len=vec_len(cross_product_vec)

	distance_meters=EARTH_RADIUS_METERS * math.atan2(cross_product_vec_len,dot_product(loc1,loc2))

	if cross_product_vec_len < 1e-10:
		azimuth_deg=0
	else:
		azimuth_cos=dot_product(cross_product_vec,longitude_plane_normal) / float(cross_product_vec_len)
		azimuth_deg=math.degrees(math.acos(azimuth_cos)) if abs(azimuth_cos) < 0.999999 else \
																		(0 if azimuth_cos > 0 else math.pi)
		if dot_product(loc1,cross_product(longitude_plane_normal,cross_product_vec)) > 0:
			azimuth_deg*=-1

	return (distance_meters,azimuth_deg)

def select_next_image_links(cur_image_id,viewer_uid=-1):
		# Returns a dict: [id]=(distance_meters,azimuth_deg,center_azimuth_deg)

	global geo_images

	cur_lon,cur_lat=geo_images[cur_image_id][:2]

	selected_images=dict()		# [id]=(lon,lat,lon_weight,min_angle_square)

	for id,(lon,lat,center_azimuth_deg,owner_uid,sharing_mode) in random.sample(
										geo_images.items(),len(geo_images)):	#!!! In the future, sort by rating instead
		if (sharing_mode or 'private') != 'public' and viewer_uid != owner_uid:
			continue
		for other_lon,other_lat,lon_weight,min_angle_square in selected_images.values():
			if ((lon - other_lon)*lon_weight) ** 2 + (lat - other_lat) ** 2 < min_angle_square:
				break
		else:
			if id != cur_image_id:
				selected_images[id]=(lon,lat,math.cos(math.radians(lat)),
											(0.2 * calc_geo_distance_deg(cur_lon,cur_lat,lon,lat)) ** 2)

	results=dict()
	for id in selected_images:
		results[id]=calc_geo_distance_and_azimuth(*(geo_images[cur_image_id][:2] + geo_images[id][:2])) + \
					(geo_images[id][2],)

	return results

if __name__ == '__main__':
	import sys,os

	if len(sys.argv) < 1+2:
		print('Usage:')
		print()
		print('%s <viewable-images-dir> <output-pickle-file>' % (sys.argv[0],))
		exit(1)

	viewable_images_dir=sys.argv[1]
	load_geo_images_list(viewable_images_dir)

	public_images=0
	private_images=0

	for image_id in geo_images:
		dirname=viewable_images_dir + '/' + image_id
		if not os.access(dirname,os.F_OK):
			print('Warning: missing image directory',dirname)

		if geo_images[image_id][4] == 'public':
			public_images+=1
		else:
			private_images+=1

	print('%d public images, %d private images' % (public_images,private_images))

	pickle.dump(geo_images,open(sys.argv[2],'wb'),2)
	print('Pickle file written')
