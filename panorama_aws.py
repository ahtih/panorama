#!/usr/bin/python
# -*- encoding: latin-1 -*-

import sys,os.path,random,json,socket,boto3,panorama

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
	images=[]
	fname_to_idx={}
	for match_result in match_results:
		for key_prefix in ('img1_','img2_'):
			fname=match_result.get(key_prefix + 'fname')
			if not fname or fname in fname_to_idx:
				continue
			fname_to_idx[fname]=len(images)
			images.append((fname,float(match_result[key_prefix + 'focal_length_35mm']),
													map(int,match_result[key_prefix + 'channel_keypoints'])))

	output_fd=sys.stdout
	if output_fname:
		output_fd=open(output_fname,'wt')

	panorama.write_output_file_header(output_fd,images)

	matches=[]
	for match_result in match_results:
		idx1=fname_to_idx.get(match_result.get('img1_fname'))
		idx2=fname_to_idx.get(match_result.get('img2_fname'))
		if idx1 is None or idx2 is None:
			continue

		matched_points=match_result.get('matched_points',tuple())

		output_str=''
		match_metrics=None

		if matched_points:
			for coords_decimal in matched_points:
				x1,y1,x2,y2=map(int,coords_decimal)
				output_str+='                <point x1="%d" y1="%d" x2="%d" y2="%d"/>\n' % (x1,y1,x2,y2)
			match_metrics=(int(match_result['score']),int(match_result['count']),
											float(match_result['angle_deg']),
											float(match_result['xd']),float(match_result['yd']))

		matches.append((idx1,idx2,match_result.get('debug_str',''),output_str,match_metrics))

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

	while True:
		match_results=get_match_results(processing_batch_key)
		if len(match_results) >= expected_nr_of_match_results:
			break
		if len(match_results) != prev_printed_status:
			print 'Completed %d of %d pairwise matches' % (len(match_results),expected_nr_of_match_results)
			prev_printed_status=len(match_results)

	write_output_file(match_results,keyword_args.get('--output-fname'))
