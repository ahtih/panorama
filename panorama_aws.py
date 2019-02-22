#!/usr/bin/python
# -*- encoding: latin-1 -*-

import sys,os.path,time,random,json,socket,boto3,panorama,image_rotation_optimiser

def get_match_results(processing_batch_key):
	try:
		table=panorama.aws_session.resource('dynamodb').Table(panorama.DYNAMODB_TABLE_NAME)
		query_result=table.query(KeyConditionExpression=boto3.dynamodb.conditions.Key('processing_batch_key').
																				eq(processing_batch_key),
								ConsistentRead=False)
	except socket.error:
		return tuple()

	return query_result.get('Items',tuple())

def write_output_file(match_results,output_fname=None):
	images=[]		# (fname,focal_length_35mm,channel_keypoints)
	fname_to_idx={}

	image_rotation_optimiser.clear_images()

	for match_result in match_results:
		for key_prefix in ('img1_','img2_'):
			fname=match_result.get(key_prefix + 'fname')
			if not fname or fname in fname_to_idx:
				continue
			fname_to_idx[fname]=len(images)
			images.append((fname,float(match_result[key_prefix + 'focal_length_35mm']),
												map(int,match_result[key_prefix + 'channel_keypoints'])))

			image_rotation_optimiser.add_image(fname)

	focal_length_pixels=2844.49				#!!!!

	matches=[]
	for match_result in match_results:
		fname1=match_result.get('img1_fname')
		fname2=match_result.get('img2_fname')

		idx1=fname_to_idx.get(fname1)
		idx2=fname_to_idx.get(fname2)
		if idx1 is None or idx2 is None:
			continue

		matched_points=match_result.get('matched_points',tuple())

		output_str=''
		match_metrics=None
		match_coords=[]

		if matched_points:
			for coords_decimal in matched_points:
				coords_int=map(int,coords_decimal)
				match_coords.append(coords_int)
				output_str+='                <point x1="%d" y1="%d" x2="%d" y2="%d"/>\n' % \
																					tuple(coords_int)

			match_metrics=(int(match_result['score']),int(match_result['count']),
											float(match_result['angle_deg']),
											float(match_result['xd']),float(match_result['yd']))

		matches.append((idx1,idx2,match_result.get('debug_str',''),output_str,match_metrics))

		image_rotation_optimiser.add_image_pair_match(
									(fname1,fname2),
									(int(match_result['img1_width']),int(match_result['img1_height'])),
									(int(match_result['img2_width']),int(match_result['img2_height'])),
									focal_length_pixels,focal_length_pixels,
									match_coords)

	image_rotation_optimiser.optimise_panorama()

	output_fd=sys.stdout
	if output_fname:
		output_fd=open(output_fname,'wt')

	panorama.write_output_file_header(output_fd)

	for fname,focal_length_35mm,channel_keypoints in images:
		panorama.write_output_file_image(output_fd,fname,focal_length_35mm,channel_keypoints,
						image_rotation_optimiser.get_image_kolor_file_angles_rad(fname),focal_length_pixels)

	panorama.write_output_file_midsection(output_fd,len(images))

	panorama.write_output_file_matches(output_fd,matches,len(images))
	panorama.write_output_file_footer(output_fd)

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

panorama.init_aws_session(positional_args[0])

if '--output-batch' in keyword_args:
	processing_batch_key=keyword_args['--output-batch']

	match_results=get_match_results(processing_batch_key)
	print 'Got %d pairwise image match results' % (len(match_results),)

	write_output_file(match_results,keyword_args.get('--output-fname'))
else:
	image_fnames=positional_args[1:]

	processing_batch_key='%016x' % (random.randint(0,2**64-1),)
	print 'Creating processing batch %s with %u images' % (processing_batch_key,len(image_fnames))

	s3_fnames=[]

	for idx,fname in enumerate(image_fnames):
		print 'Processing',fname
		img=panorama.ImageKeypoints(fname,True)
		# print '   ','+'.join(map(str,img.channel_keypoints))

		s3_fname=processing_batch_key + '/' + str(idx) + '-' + os.path.basename(fname)
		img.save_to_s3(s3_fname)
		s3_fnames.append(s3_fname)

	print 'Spawning match_images Lambda tasks'

	lambda_client=panorama.aws_session.client('lambda')

	lambda_parameters={'function': 'spawn_match_images_tasks',
								'processing_batch_key': processing_batch_key,
								's3_fnames': s3_fnames,
								'orig_fnames': image_fnames}
	invoke_result=lambda_client.invoke(FunctionName='panorama',InvocationType='RequestResponse',
										Payload=json.dumps(lambda_parameters))
	status_code=invoke_result.get('StatusCode')

	if status_code < 200 or status_code > 299:
		print 'Lambda invoke failed with:',invoke_result
		exit(1)

	expected_nr_of_match_results=len(s3_fnames) * (len(s3_fnames)-1) / 2
	prev_printed_status=0

	prev_update_time=time.time()
	while True:
		match_results=get_match_results(processing_batch_key)
		if len(match_results) != prev_printed_status:
			print 'Completed %d of %d pairwise matches' % (len(match_results),expected_nr_of_match_results)
			prev_printed_status=len(match_results)
			prev_update_time=time.time()
		if len(match_results) >= expected_nr_of_match_results:
			break
		if time.time() > prev_update_time + 60:
			print 'Timeout - Lambda functions not progressing'
			exit(1)

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

	if error_count:
		print '%d pairwise matches produced errors, first error is here:' % (error_count,)
		print first_error_text

	write_output_file(match_results,keyword_args.get('--output-fname'))
