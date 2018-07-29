#!/usr/bin/python
# -*- encoding: latin-1 -*-

import sys,os.path,random,json,boto3,panorama

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
	table=panorama.aws_session.resource('dynamodb').Table(panorama.DYNAMODB_TABLE_NAME)
	query_result=table.query(KeyConditionExpression=boto3.dynamodb.conditions.Key('processing_batch_key').
																				eq(processing_batch_key),
								ConsistentRead=False)
	match_results=query_result.get('Items',tuple())
	print len(match_results)	#!!!

	images=[]
	fname_to_idx={}
	for match_result in match_results:
		for key_prefix in ('img1_','img2_'):
			fname=match_result.get(key_prefix + 's3_fname')
			if not fname or fname in fname_to_idx:
				continue
			fname_to_idx[fname]=len(images)
			images.append((fname,float(match_result[key_prefix + 'focal_length_35mm']),
													map(int,match_result[key_prefix + 'channel_keypoints'])))

	output_fname=keyword_args.get('--output-fname')
	output_fd=sys.stdout
	if output_fname:
		output_fd=open(output_fname,'wt')

	panorama.write_output_file_header(output_fd,images)

	link_stats=[[] for img in images]

	for match_result in match_results:
		idx1=fname_to_idx.get(match_result.get('img1_s3_fname'))
		idx2=fname_to_idx.get(match_result.get('img2_s3_fname'))
		if idx1 is None or idx2 is None:
			continue

		print >>output_fd,'        <!-- image %d<-->%d: %s -->' % (idx1,idx2,match_result.get('debug_str',''))

		output_str=''
		for coords_decimal in match_result.get('matched_points',tuple()):
			x1,y1,x2,y2=map(int,coords_decimal)
			output_str+='                <point x1="%d" y1="%d" x2="%d" y2="%d"/>\n' % (x1,y1,x2,y2)

		if output_str:
			print >>output_fd,'        <match image1="%d" image2="%d">\n            <points>\n%s            </points>\n        </match>' % \
																				(idx1,idx2,output_str)
			link_stats[idx1].append(idx2)
			link_stats[idx2].append(idx1)

	print >>output_fd,'<!-- Link stats: -->'

	for idx,linked_images in enumerate(link_stats):
		print >>output_fd,'<!-- #%d links: %s -->' % (idx,' '.join(map(str,sorted(linked_images))))

	panorama.write_output_file_footer(output_fd)

elif '--match-batch' in keyword_args:
	processing_batch_key=keyword_args['--match-batch']
	#!!!
else:
	image_fnames=positional_args[1:]

	processing_batch_key='%016x' % (random.randint(0,2**64-1),)
	print 'Creating processing batch %s with %u images' % (processing_batch_key,len(image_fnames))

	s3_fnames=[]

	for fname in image_fnames:
		print 'Processing',fname
		img=panorama.ImageKeypoints(fname,True)
		print '   ','+'.join(map(str,img.channel_keypoints))

		s3_fname=processing_batch_key + '/' + os.path.basename(fname)
		img.save_to_s3(s3_fname)
		s3_fnames.append(s3_fname)

	print 'Spawning match_images Lambda tasks'

	lambda_client=panorama.aws_session.client('lambda')

	lambda_parameters={'function': 'spawn_match_images_tasks',
								'processing_batch_key': processing_batch_key,
								's3_fnames': s3_fnames}
	invoke_result=lambda_client.invoke(FunctionName='panorama',InvocationType='RequestResponse',
										Payload=json.dumps(lambda_parameters))
	status_code=invoke_result.get('StatusCode')

	if status_code < 200 or status_code > 299:
		print 'Lambda invoke failed with:',invoke_result
