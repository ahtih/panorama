#!/usr/bin/python
# -*- coding: latin-1

import sys,os,operator,math,numpy,kolor_xml_file,cv2

images=dict()	# [fname]=[image1_keypoints,other_images_projected_keypoints,image_matrix,matches,cur_error,diagonal_fov_deg]
				#							matches=((image2_fname,nr_of_points,image2_start_point_idx),...)

euler_angle_unit_matrices=(	numpy.array([
											[	[ 0, 0, 0],
												[ 0, 0,-1],
												[ 0, 1, 0]],
											[	[ 0, 0, 0],
												[ 0, 1, 0],
												[ 0, 0, 1]],
											[	[ 1, 0, 0],
												[ 0, 0, 0],
												[ 0, 0, 0]]
											]).transpose((1,2,0)),
							numpy.array([
											[	[ 0, 0, 1],
												[ 0, 0, 0],
												[-1, 0, 0]],
											[	[ 1, 0, 0],
												[ 0, 0, 0],
												[ 0, 0, 1]],
											[	[ 0, 0, 0],
												[ 0, 1, 0],
												[ 0, 0, 0]]
											]).transpose((1,2,0)),
							numpy.array([
											[	[ 0,-1, 0],
												[ 1, 0, 0],
												[ 0, 0, 0]],
											[	[ 1, 0, 0],
												[ 0, 1, 0],
												[ 0, 0, 0]],
											[	[ 0, 0, 0],
												[ 0, 0, 0],
												[ 0, 0, 1]]
											]).transpose((1,2,0))
										)

def normalise(v):
	return v / numpy.linalg.norm(v)

def acos_degrees(value):
	if value >= 1:
		return 0
	if value <= -1:
		return 180
	return math.degrees(math.acos(value))

def matrix_to_kolor_file_angles(m):
	# Returns values in radians
	#!!! Do we need to handle specially the singular matrix case where sqrt is 0?

	roll_rad =-math.atan2(-m[2,1],m[2,2])
	pitch_rad=-math.atan2(-m[2,0],math.sqrt(m[0,0]**2 + m[1,0]**2))
	yaw_rad  =-math.atan2( m[1,0],m[0,0])

	return (yaw_rad,pitch_rad,roll_rad)

def euler_angles_to_matrix(euler_angles_rad):
	global euler_angle_unit_matrices

	return numpy.asmatrix(numpy.linalg.multi_dot([
									euler_angle_unit_matrices[i].dot(
												numpy.array([math.sin(angle),math.cos(angle),1])) \
															for i,angle in enumerate(euler_angles_rad)]))

epsilon_angle_rad=0.3 / float(3000)
epsilon_rotation_matrices=(	euler_angles_to_matrix((epsilon_angle_rad,0,0)),
							euler_angles_to_matrix((0,epsilon_angle_rad,0)),
							euler_angles_to_matrix((0,0,epsilon_angle_rad)))
epsilon_2axes_rad=epsilon_angle_rad / math.sqrt(2.0)
alt_epsilon_rotation_matrices=(
							euler_angles_to_matrix((-epsilon_angle_rad,0,0)),
							euler_angles_to_matrix((0,-epsilon_angle_rad,0)),
							euler_angles_to_matrix((0,0,-epsilon_angle_rad)),

							euler_angles_to_matrix((epsilon_2axes_rad,epsilon_2axes_rad,0)),
							euler_angles_to_matrix((0,epsilon_2axes_rad,epsilon_2axes_rad)),
							euler_angles_to_matrix((epsilon_2axes_rad,0,epsilon_2axes_rad)),

							euler_angles_to_matrix((epsilon_2axes_rad,-epsilon_2axes_rad,0)),
							euler_angles_to_matrix((0,epsilon_2axes_rad,-epsilon_2axes_rad)),
							euler_angles_to_matrix((epsilon_2axes_rad,0,-epsilon_2axes_rad)),

							euler_angles_to_matrix((-epsilon_2axes_rad,epsilon_2axes_rad,0)),
							euler_angles_to_matrix((0,-epsilon_2axes_rad,epsilon_2axes_rad)),
							euler_angles_to_matrix((-epsilon_2axes_rad,0,epsilon_2axes_rad)),

							euler_angles_to_matrix((-epsilon_2axes_rad,-epsilon_2axes_rad,0)),
							euler_angles_to_matrix((0,-epsilon_2axes_rad,-epsilon_2axes_rad)),
							euler_angles_to_matrix((-epsilon_2axes_rad,0,-epsilon_2axes_rad)),
							)

def keypoint_pixels_to_vec3(x,y,focal_length_pixels,image_size):
	# Output vector coordinate system is such that X points ahead, Y left, Z up

	if x < 0 or x >= image_size[0] or y < 0 or y >= image_size[1]:
		print 'Invalid keypoint coordinates',x,y,focal_length_pixels,image_size

	v=numpy.array([focal_length_pixels,image_size[0]/2.0 - x,image_size[1]/2.0 - y],'f')
	return normalise(v)

def camera_forward_vec3(image_fname):
	global images

	return numpy.asarray(images[image_fname][2])[:,0]

def calc_image_pair_fitness(image1_keypoints,other_images_projected_keypoints,image1_matrix):
	# This takes ca 10us for 30 keypoint pairs

	image1_projected_keypoints=numpy.asarray((image1_matrix * image1_keypoints).T)

#	dot_product_sum=0
#	for kp_idx in range(len(other_images_projected_keypoints)):
#		dot_product_sum+=image1_projected_keypoints[kp_idx].dot(other_images_projected_keypoints[kp_idx])

	return len(other_images_projected_keypoints) - \
									numpy.sum(image1_projected_keypoints * other_images_projected_keypoints)

def calc_avg_error_deg():
	global images

	error_sum=0
	total_keypoints=0
	for image_record in images.values():
		image1_keypoints,other_images_projected_keypoints,image1_matrix,matches=image_record[:4]
		error_sum+=calc_image_pair_fitness(image1_keypoints,other_images_projected_keypoints,image1_matrix)
		total_keypoints+=image1_keypoints.shape[1]

	return acos_degrees(1 - error_sum / float(total_keypoints))

def print_keypoint_errors():
	global images

	for image_fname in images.keys():
		image1_keypoints,other_images_projected_keypoints,image1_matrix=images[image_fname][:3]
		image1_projected_keypoints=numpy.asarray((image1_matrix * image1_keypoints).T)

		cur_error=calc_image_pair_fitness(image1_keypoints,other_images_projected_keypoints,image1_matrix)
		print '   ',image_fname,cur_error

		for kp_idx in range(len(other_images_projected_keypoints)):
			dot_product=image1_projected_keypoints[kp_idx].dot(other_images_projected_keypoints[kp_idx])
			print '        ',kp_idx+1,dot_product,acos_degrees(dot_product),image1_projected_keypoints[kp_idx],other_images_projected_keypoints[kp_idx]

def add_keypoints(fnames_pair,keypoints_lists_pair):
	global images

	for idx in range(2):
		image1_fname=fnames_pair[  idx]
		image2_fname=fnames_pair[1-idx]
		image1_keypoints=numpy.asmatrix(keypoints_lists_pair[idx]).T

		nr_of_points=image1_keypoints.shape[1]
		images[image2_fname][3].append((image1_fname,nr_of_points,len(images[image1_fname][1])))

		images[image1_fname][0]=numpy.append(images[image1_fname][0],image1_keypoints,axis=1)
		images[image1_fname][1]=numpy.append(images[image1_fname][1],
												numpy.zeros(tuple(reversed(image1_keypoints.shape))),axis=0)

def set_keypoints_for_other_images(image_fname):
	global images

	img_record=images[image_fname]
	projected_keypoints=numpy.asarray((img_record[2] * img_record[0]).T)

	image1_start_point_idx=0
	for image2_fname,nr_of_points,image2_start_point_idx in img_record[3]:
		images[image2_fname][1][image2_start_point_idx:(image2_start_point_idx+nr_of_points),:]= \
						projected_keypoints[image1_start_point_idx:(image1_start_point_idx+nr_of_points),:]
		image1_start_point_idx+=nr_of_points

def try_optimise_direction(gradient_matrix,cur_error,image1_keypoints,other_images_projected_keypoints,
																		image1_matrix,max_rotation_rad):
	global epsilon_angle_rad

	iterations=0
	best_error=cur_error
	best_m=None
	best_rot_amount=None
	best_rot_rad=epsilon_angle_rad

	for i in range(15):
		iterations+=1

		m=image1_matrix * gradient_matrix
		e=calc_image_pair_fitness(image1_keypoints,other_images_projected_keypoints,m)

		if e >= best_error:
			break

		best_error=e
		best_m=m
		best_rot_amount=i
		best_rot_rad*=2

		if max_rotation_rad is not None and best_rot_rad >= max_rotation_rad:
			break

		gradient_matrix=gradient_matrix * gradient_matrix

	return (iterations,best_m,best_error,best_rot_rad,best_rot_amount)

def calc_optimisation_gradient(cur_error,image1_matrix,image1_keypoints,other_images_projected_keypoints,
																					axis_rotation_matrices):
	gradient=[]

	best_fitness_improvement=0
	best_direction_axis=None
	iterations=0

	for axis,axis_rotation_matrix in enumerate(axis_rotation_matrices):
		iterations+=1

		m=image1_matrix * axis_rotation_matrix
		fitness_improvement=cur_error - calc_image_pair_fitness(image1_keypoints,
																		other_images_projected_keypoints,m)
		gradient.append(fitness_improvement)

		if best_fitness_improvement < fitness_improvement:
			best_fitness_improvement= fitness_improvement
			best_direction_axis=axis

	return (iterations,gradient,best_direction_axis)

def optimise_image_rot(image_fname,max_rotation_rad=None):
	global epsilon_angle_rad,epsilon_rotation_matrices,alt_epsilon_rotation_matrices

	image1_keypoints,other_images_projected_keypoints,image1_matrix=images[image_fname][:3]

	cur_error=calc_image_pair_fitness(image1_keypoints,other_images_projected_keypoints,image1_matrix)

	iterations=0
	cumulative_rot_rad=0

	while max_rotation_rad is None or cumulative_rot_rad < max_rotation_rad:

		_iterations,gradient_direction,best_direction_axis=calc_optimisation_gradient(
												cur_error,image1_matrix,image1_keypoints,
												other_images_projected_keypoints,epsilon_rotation_matrices)
		iterations+=_iterations

		best_m=None

		if best_direction_axis is not None:
			gradient_direction=normalise(numpy.array(gradient_direction))

			_iterations,best_m,best_error,best_rot_rad,best_rot_amount=try_optimise_direction(
							euler_angles_to_matrix(gradient_direction * epsilon_angle_rad),
							cur_error,image1_keypoints,
							other_images_projected_keypoints,image1_matrix,
							max_rotation_rad-cumulative_rot_rad if max_rotation_rad is not None else None)
			iterations+=_iterations

			if best_m is None:
				# print '   ',gradient_direction

				_iterations,best_m,best_error,best_rot_rad,best_rot_amount=try_optimise_direction(
							epsilon_rotation_matrices[best_direction_axis],
							cur_error,image1_keypoints,
							other_images_projected_keypoints,image1_matrix,
							max_rotation_rad-cumulative_rot_rad if max_rotation_rad is not None else None)
				iterations+=_iterations

		if best_m is None:
			_iterations,gradient_direction,best_direction_axis=calc_optimisation_gradient(
											cur_error,image1_matrix,image1_keypoints,
											other_images_projected_keypoints,alt_epsilon_rotation_matrices)
			iterations+=_iterations

			if best_direction_axis is not None:
				_iterations,best_m,best_error,best_rot_rad,best_rot_amount=try_optimise_direction(
							alt_epsilon_rotation_matrices[best_direction_axis],
							cur_error,image1_keypoints,
							other_images_projected_keypoints,image1_matrix,
							max_rotation_rad-cumulative_rot_rad if max_rotation_rad is not None else None)
				iterations+=_iterations

		if best_m is None:
			break

		cur_error=best_error
		image1_matrix=best_m
		cumulative_rot_rad+=best_rot_rad

		# print '   ',best_rot_amount,cur_error,matrix_to_kolor_file_angles(image1_matrix),gradient_direction,numpy.linalg.det(image1_matrix)-1

	images[image_fname][2]=image1_matrix
	images[image_fname][4]=cur_error
	set_keypoints_for_other_images(image_fname)

	# print iterations,math.degrees(cumulative_rot_rad),len(other_images_projected_keypoints),image_fname
	return (iterations,cumulative_rot_rad)

def clear_images():
	global images

	images=dict()

def add_image(fname,initial_quaternion=None):
	global images

	images[fname]=[numpy.asmatrix(numpy.zeros((3,0))),
					numpy.zeros((0,3)),
					numpy.matrix(numpy.identity(3)) if initial_quaternion is None else q.to_matrix(),
					[],
					None,
					None]

def calc_diagonal_fov_deg(image_size,focal_length_pixels):
	diagonal_pixels=math.sqrt(image_size[0]**2 + image_size[1]**2)
	return math.degrees(2 * math.atan2(diagonal_pixels / 2,focal_length_pixels))

def add_image_pair_match(image_pair_fnames,image1_size,image2_size,
										image1_focal_length_pixels,image2_focal_length_pixels,matches):
	if not matches:
		return

	image1_keypoints=[]
	image2_keypoints=[]

	for x1,y1,x2,y2 in matches:
		image1_keypoints.append(keypoint_pixels_to_vec3(x1,y1,image1_focal_length_pixels,image1_size))
		image2_keypoints.append(keypoint_pixels_to_vec3(x2,y2,image2_focal_length_pixels,image2_size))

	add_keypoints(image_pair_fnames,(image1_keypoints,image2_keypoints))

	images[image_pair_fnames[0]][5]=calc_diagonal_fov_deg(image1_size,image1_focal_length_pixels)
	images[image_pair_fnames[1]][5]=calc_diagonal_fov_deg(image2_size,image2_focal_length_pixels)

def optimise_panorama(print_verbose=False):
	global images

	for image_fname in images.keys():
		set_keypoints_for_other_images(image_fname)

	total_iterations=0

	for i in range(3000):
		round_iterations=0
		round_rot_rad=0
		for image_fname in images.keys():
			iterations,rot_rad=optimise_image_rot(image_fname,math.radians(30))
			round_iterations+=iterations
			round_rot_rad+=rot_rad
		if print_verbose:
			print '   Round %u: %u iterations, %.1fdeg rotation' % \
														(i,round_iterations,math.degrees(round_rot_rad))
		if round_rot_rad == 0:
			break

		total_iterations+=round_iterations

	avg_error=sum(map(operator.itemgetter(4),images.values())) / \
									sum([image_record[0].shape[1] for image_record in images.values()])
	print '%u total iterations, avg error %.2fdeg' % (total_iterations,acos_degrees(1-avg_error))

def get_image_kolor_file_angles_rad(fname):
	global images

	m=images[fname][2]
	return matrix_to_kolor_file_angles(m)

def matched_image_fnames(image1_fname):
	global images

	return map(operator.itemgetter(0),images[image1_fname][3])

def remove_insignificant_matches():
	global images

	image_forward_vectors=dict()

	for image_fname in images.keys():
		image_forward_vectors[image_fname]=camera_forward_vec3(image_fname)

	match_angle_deg=dict()
	match_nr_of_points=dict()

	for image1_fname,image1_record in images.items():
		for image2_fname,nr_of_points,image2_start_point_idx in image1_record[3]:
			angle_deg=acos_degrees(image_forward_vectors[image1_fname].dot(
																	image_forward_vectors[image2_fname]))
			match_angle_deg[image1_fname,image2_fname]=angle_deg
			match_nr_of_points[image1_fname,image2_fname]=nr_of_points

	matches_to_remove=set()

	for (image1_fname,image2_fname),angle_deg in match_angle_deg.items():
		if image1_fname < image2_fname:
			continue
		diagonal_fov_deg=min(images[image1_fname][5],images[image2_fname][5])

		if angle_deg < 0.45 * diagonal_fov_deg:
			continue

		for image3_fname in frozenset(matched_image_fnames(image1_fname)).intersection(
																		matched_image_fnames(image2_fname)):
			image3_record=images[image3_fname]
			diagonal_fov_deg3=min(diagonal_fov_deg,image3_record[5])
			angle_deg3=max(	match_angle_deg[image1_fname,image3_fname],
							match_angle_deg[image2_fname,image3_fname])
			if angle_deg3 > 0.6 * angle_deg:
				continue
			if min(	match_nr_of_points[image1_fname,image3_fname],
					match_nr_of_points[image2_fname,image3_fname]) < 7:
				continue

			image3_error_deg=acos_degrees(1 - image3_record[4] / float(len(image3_record[1])))
			if image3_error_deg > 6:
				continue

			# print angle_deg3/angle_deg,image1_fname,image2_fname,angle_deg,angle_deg3,match_nr_of_points[image1_fname,image2_fname]

			matches_to_remove.add((image1_fname,image2_fname))
			matches_to_remove.add((image2_fname,image1_fname))
			break

	for image1_fname,delete_image2_fname in matches_to_remove:
		img_record=images[image1_fname]
		matches=img_record[3]

		kp_idx=0
		nr_of_kps=0

		for idx in range(len(matches)):
			nr_of_kps=matches[idx][1]
			if matches[idx][0] != delete_image2_fname:
				kp_idx+=nr_of_kps
				continue
			del matches[idx]

			img_record[0]=numpy.delete(img_record[0],numpy.s_[kp_idx:(kp_idx+nr_of_kps)],axis=1)
			img_record[1]=numpy.delete(img_record[1],numpy.s_[kp_idx:(kp_idx+nr_of_kps)],axis=0)

			break

		for img_record in images.values():
			for idx,match_record in enumerate(img_record[3]):
				if match_record[0] == image1_fname and match_record[2] > kp_idx:
					match_record=list(match_record)
					match_record[2]-=nr_of_kps
					img_record[3][idx]=tuple(match_record)

	return matches_to_remove

def optimise_panorama_and_remove_insignificant_matches(print_verbose=False):
	optimise_panorama(print_verbose)

	matches_to_remove=remove_insignificant_matches()

	if matches_to_remove:
		if print_verbose:
			print 'Removing %d insignificant matches' % (len(matches_to_remove)/2,)
		optimise_panorama(print_verbose)

	return matches_to_remove

if __name__ == '__main__':
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
		keypoint_vectors=numpy.stack([numpy.array([0.3,0.3,0.6]) for i in range(30)])
		keypoint_vectors_matrix=numpy.asmatrix(keypoint_vectors).T

		image1_matrix=numpy.asmatrix(numpy.array([	[ 5, 1 ,3], 
													[ 1, 1 ,1], 
													[ 1, 2 ,1]]))
		for i in range(1000*1000):
			error_sum=calc_image_pair_fitness(keypoint_vectors_matrix,keypoint_vectors,image1_matrix)

		print error_sum		# -40.2 = 30 * -1.34
	else:
		ignore_input_rotations=('--ignore-input-rotations' in keyword_args)
		no_optimise=('--no-optimise' in keyword_args)

		image_fnames_sequence=kolor_xml_file.read_kolor_xml_file(positional_args[0],True)

		image_sizes=dict()

		for image_pair_idx in kolor_xml_file.matches.keys():
			for image_fname in image_pair_idx:
				if image_fname not in image_sizes:
					if os.access(image_fname,os.R_OK):
						image_sizes[image_fname]=tuple(reversed(cv2.imread(image_fname).shape[:2]))

		clear_images()

		for image_fname,q in kolor_xml_file.image_quaternions.items():
			add_image(image_fname,None if ignore_input_rotations else q)

		for image_pair_idx,matches in kolor_xml_file.matches.items():
			# if len(positional_args) >= 1+2:
			#	if image_pair_idx != tuple(positional_args[1:]):
			#		continue

			focal_length_pixels=2598.69				#!!!!
			add_image_pair_match(image_pair_idx,
									image_sizes[image_pair_idx[0]],image_sizes[image_pair_idx[1]],
									focal_length_pixels,focal_length_pixels,
									matches)

		if no_optimise:
			for image_fname in images.keys():
				set_keypoints_for_other_images(image_fname)

			print 'Avg error %.2fdeg' % (calc_avg_error_deg(),)
			# print_keypoint_errors()

			exit(0)

		optimise_panorama_and_remove_insignificant_matches(True)

		for image_fname in image_fnames_sequence:
			m=images[image_fname][2]
			kolor_file_angles_rad=matrix_to_kolor_file_angles(m)
			# print '%s %+.2f %+.2f %+.2f' % ((os.path.basename(image_fname),) + tuple(map(math.degrees,kolor_file_angles_rad)))
			# print os.path.basename(image_fname),math.degrees(math.atan2(m[idx,0],m[idx,1])),math.degrees(math.atan2(m[idx,2],math.sqrt(m[idx,0]**2 + m[idx,1]**2)))

			print '        <image>'
			print ('            <def filename="%s" focal35mm="0" lensModel="0" ' + \
										'fisheyeRadius="0" fisheyeCoffX="0" fisheyeCoffY="0"/>') % \
												(image_fname,)
			print '            <camera yaw="%.5f" pitch="%.5f" roll="%.5f" f="%.2f"/>' % \
														(kolor_file_angles_rad + (focal_length_pixels,))
			print '        </image>'
