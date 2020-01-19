#!/usr/bin/python
# -*- encoding: latin-1 -*-

import sys,os.path,time,random,json,socket,multiprocessing,boto3,panorama,image_rotation_optimiser

def get_match_results(processing_batch_key):
	try:
		table=panorama.aws_session.resource('dynamodb').Table(panorama.DYNAMODB_TABLE_NAME)
		query_result=table.query(KeyConditionExpression=boto3.dynamodb.conditions.Key('processing_batch_key').
																				eq(processing_batch_key),
								ConsistentRead=False)
	except socket.error:
		return tuple()

	return query_result.get('Items',tuple())

def write_output_file(match_results,output_fname=None,print_verbose=False):
	images=[]		# (fname,focal_length_35mm,channel_keypoints,focal_length_pixels)
	fname_to_idx={}

	image_rotation_optimiser.clear_images()

	for match_result in match_results:
		for key_prefix in ('img1_','img2_'):
			fname=match_result.get(key_prefix + 'fname')
			if not fname or fname in fname_to_idx:
				continue
			fname_to_idx[fname]=len(images)
			images.append((fname,float(match_result[key_prefix + 'focal_length_35mm']),
								map(int,match_result[key_prefix + 'channel_keypoints']),
								float(match_result[key_prefix + 'focal_length_pixels'])))

			image_rotation_optimiser.add_image(fname)

	matches=[]
	optimiser_matches=[]

	for match_result in match_results:
		fname1=match_result.get('img1_fname')
		fname2=match_result.get('img2_fname')

		idx1=fname_to_idx.get(fname1)
		idx2=fname_to_idx.get(fname2)
		if idx1 is None or idx2 is None:
			continue

		matched_points=match_result.get('matched_points',tuple())

		if matched_points:
			output_str=''
			match_coords=[]

			for coords_decimal in matched_points:
				coords_int=map(int,coords_decimal)
				match_coords.append(coords_int)
				output_str+='                <point x1="%d" y1="%d" x2="%d" y2="%d"/>\n' % \
																					tuple(coords_int)

			match_metrics=(int(match_result['score']),int(match_result['count']),
											float(match_result['angle_deg']),
											float(match_result['xd']),float(match_result['yd']))
			optimiser_matches.append((
									(fname1,fname2),
									(int(match_result['img1_width']),int(match_result['img1_height'])),
									(int(match_result['img2_width']),int(match_result['img2_height'])),
									images[idx1][3],	# focal_length_pixels
									images[idx2][3],	# focal_length_pixels
									match_coords))

			matches.append([idx1,idx2,match_result.get('debug_str',''),output_str,match_metrics])
		else:
			optimiser_matches.append(None)
			matches.append([idx1,idx2,match_result.get('debug_str',''),'',None])

	trustworthy_indexes=set(panorama.filter_matches(matches))
	for idx in trustworthy_indexes:
		image_rotation_optimiser.add_image_pair_match(*optimiser_matches[idx])

	matches_to_remove=image_rotation_optimiser.optimise_panorama_and_remove_insignificant_matches(
																							print_verbose)
	for idx,(idx1,idx2,debug_str,output_string,match_metrics) in enumerate(matches):
		if (images[idx1][0],images[idx2][0]) in matches_to_remove:
			trustworthy_indexes.discard(idx)

	output_fd=sys.stdout
	if output_fname:
		output_fd=open(output_fname,'wt')

	panorama.write_output_file_header(output_fd)

	for fname,focal_length_35mm,channel_keypoints,focal_length_pixels in images:
		panorama.write_output_file_image(output_fd,fname,focal_length_35mm,channel_keypoints,
						image_rotation_optimiser.get_image_kolor_file_angles_rad(fname),focal_length_pixels)

	panorama.write_output_file_midsection(output_fd,len(images))

	panorama.write_output_file_matches(output_fd,matches,trustworthy_indexes,len(images))
	panorama.write_output_file_footer(output_fd)

def extract_keypoints_and_upload_to_s3(fname,s3_fname,print_verbose=False):
	if print_verbose:
		print 'Processing',fname

	try:
		img=panorama.ImageKeypoints(fname,True)
	except Exception as e:
		print 'Error:',fname,e
		raise

	# print '   ','+'.join(map(str,img.channel_keypoints))

	if print_verbose:
		print '   Uploading to S3',fname

	img.save_to_s3(s3_fname)

def add_images_to_worker_pool(worker_pool,output_fname,image_fnames,print_verbose=False):
		# Updates processing_batches

	global processing_batches

	processing_batch_key='%016x' % (random.randint(0,2**64-1),)
	print '%s: Creating processing batch %s with %u images' % (output_fname or '-',processing_batch_key,
																						len(image_fnames))
	s3_fnames=[]
	async_result_objects=[]

	for idx,fname in enumerate(image_fnames):
		s3_fname=processing_batch_key + '/' + str(idx) + '-' + os.path.basename(fname)
		s3_fnames.append(s3_fname)
		async_result_objects.append(worker_pool.apply_async(
										extract_keypoints_and_upload_to_s3,(fname,s3_fname,print_verbose)))

	processing_batches[processing_batch_key]= \
							(output_fname,tuple(image_fnames),tuple(s3_fnames),tuple(async_result_objects))

def print_usage_and_exit():
	print '%s AWS_PROFILE_NAME [--verbose] [--output-fname=PANO_FNAME] IMAGE_FNAME [...]' % (sys.argv[0],)
	print
	print '%s AWS_PROFILE_NAME [--verbose] [--panoramas-file=INPUT_FILELIST_FNAME]' % (sys.argv[0],)
	print
	print '%s AWS_PROFILE_NAME --output-batch=BATCH_KEY [--output-fname=PANO_FNAME]' % (sys.argv[0],)
	print
	exit(0)

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

if positional_args:
	panorama.init_aws_session(positional_args[0])
else:
	print_usage_and_exit()

if '--output-batch' in keyword_args:
	processing_batch_key=keyword_args['--output-batch']

	match_results=get_match_results(processing_batch_key)
	print 'Got %d pairwise image match results' % (len(match_results),)

	write_output_file(match_results,keyword_args.get('--output-fname'),True)
else:
	print_verbose=('--verbose' in keyword_args)

	lambda_client=panorama.aws_session.client('lambda')
	worker_pool=multiprocessing.Pool(8*2)

	processing_batches=dict()

	panorama_fnames=keyword_args.get('--panoramas-file')
	if panorama_fnames:
		if not isinstance(panorama_fnames,list):
			panorama_fnames=(panorama_fnames,)

		for fname in panorama_fnames:
			for line in open(fname,'r'):
				line=line.strip()
				if not line:
					continue
				fields=line.split()
				if len(fields) < 2:
					continue
				add_images_to_worker_pool(worker_pool,fields[0],fields[1:],print_verbose=print_verbose)
	else:
		image_fnames=positional_args[1:]
		if not image_fnames:
			print_usage_and_exit()
		add_images_to_worker_pool(worker_pool,keyword_args.get('--output-fname'),image_fnames,print_verbose=print_verbose)

	if print_verbose:
		print 'Worker tasks submitted'

	match_images_started=set()
	match_images_failed=set()
	batches_completed=set()

	while True:
		something_was_done=False

		if print_verbose:
			print len(processing_batches),len(match_images_started),len(match_images_failed), \
																					len(batches_completed)

		for processing_batch_key in set(processing_batches.keys()) - \
																match_images_started - match_images_failed:
			output_fname,image_fnames,s3_fnames,async_result_objects=processing_batches[processing_batch_key]
			if not min([a.ready() for a in async_result_objects]):
				continue

			if not min([a.successful() for a in async_result_objects]):
				print '%s: Errors occurred while processing images' % (output_fname or '-',)
				match_images_failed.add(processing_batch_key)
				continue

			if print_verbose:
				print '%s: Spawning match_images Lambda tasks' % (output_fname or '-',)

			something_was_done=True
			spawn_start_time=time.time()
			lambda_parameters={'function': 'spawn_match_images_tasks',
								'processing_batch_key': processing_batch_key,
								's3_fnames': s3_fnames,
								'orig_fnames': image_fnames}
			try:
				invoke_result=lambda_client.invoke(FunctionName='panorama',InvocationType='RequestResponse',
										Payload=json.dumps(lambda_parameters))
			except Exception as e:
				print '%s: Lambda invoke failed with: %s' % (output_fname or '-',e)
				match_images_failed.add(processing_batch_key)
				continue

			status_code=invoke_result.get('StatusCode')

			if print_verbose:
				print '%s: Lambda task spawning took %dsec, status %d' % (output_fname or '-',
															time.time() - spawn_start_time,status_code)

			if status_code is not None and status_code >= 200 and status_code <= 299:
				match_images_started.add(processing_batch_key)
			else:
				match_images_failed.add(processing_batch_key)
				print '%s: Lambda invoke failed with: %s' % (output_fname or '-',invoke_result)

		for processing_batch_key in match_images_started - batches_completed:
			s3_fnames=processing_batches[processing_batch_key][2]
			expected_nr_of_match_results=len(s3_fnames) * (len(s3_fnames)-1) / 2

			match_results=get_match_results(processing_batch_key)

			if len(match_results) < expected_nr_of_match_results:
				continue

			something_was_done=True

			first_error_text=None
			error_count=0
			non_error_results=[]

			for match_result in match_results:
				error_text=match_result.get('error')
				if error_text is None:
					non_error_results.append(match_result)
				else:
					error_count+=1
					if first_error_text is None:
						first_error_text=error_text

			output_fname=processing_batches[processing_batch_key][0]

			if error_count:
				print '%s: %d pairwise matches produced errors, first error is here:' % \
																	(output_fname or '-',error_count)
				print first_error_text

			if non_error_results:
				write_output_file(match_results,output_fname,print_verbose)
				print '%s: Completed' % (output_fname or '-',)

			batches_completed.add(processing_batch_key)

		if len(batches_completed) + len(match_images_failed) >= len(processing_batches):
			break

		if not something_was_done:
			time.sleep(3)

	worker_pool.close()
	worker_pool.join()
