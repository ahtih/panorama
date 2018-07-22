#!/usr/bin/python
# -*- encoding: latin-1 -*-

#################### AWS Lambda design ###################
# 1. upload image files (JPG/PNG/etc) to S3
# 2. invoke synchronous Lambda via AWS API Gateway
#		* spawn_extract_image_keypoints_tasks
#		* arguments:
#			* S3 filenames
#			* processing batch key (arbitrary string)
#			* processing parameters
#		* this does an async Invoke for each image
#			* extract_image_keypoints (image file in S3 --> ImageKeypoints file in S3)
#			* ImageKeypoints file in S3:
#				* Object Key: processing batch key + '/' + S3 filename
# 3. periodically poll S3 for ImageKeypoints files using List Objects with processing batch key as prefix
#		* https://docs.aws.amazon.com/AmazonS3/latest/API/v2-RESTBucketGET.html
# 4. when all ImageKeypoints files are in S3, submit futher synchronous Lambda via AWS API Gateway:
#		* spawn_match_images_tasks
#		* this does an async Invoke for each image pair
#			* match_images (two ImageKeypoints objects in S3 --> match record in DynamoDB)
#			* match record in DynamoDB:
#				* partition key: processing batch key
#				* sort key: S3 filename + '_' + S3 filename
#				* attributes:
#					* debug_str: str
#					* img1_focal_length_35mm: float
#					* img2_focal_length_35mm: float
#					* matched_points:
#						x1, y1, x2, y2: int
# 5. periodically poll DynamoDB using Query with partition key until all match records are in DynamoDB
# 6. read all match records from DynamoDB, run classifier and generate XML file

import json,panorama

panorama.init_aws_session()

def match_images(event):
	panorama.process_match_and_write_to_dynamodb(event['processing_batch_key'],
																event['s3_fname1'],event['s3_fname2'])
	return 'OK'

def spawn_match_images_tasks(event):
	processing_batch_key=event['processing_batch_key']
	fnames=event['s3_fnames']

	lambda_client=panorama.aws_session.client('lambda')

	invoke_return_values=[]
	for idx1 in range(len(fnames)):
		for idx2 in range(idx1+1,len(fnames)):
			lambda_parameters={'function': 'match_images',
								'processing_batch_key': processing_batch_key,
								's3_fname1': fnames[idx1],
								's3_fname2': fnames[idx2]}
			invoke_ret_val=lambda_client.invoke(
											FunctionName='panorama',InvocationType='Event',
											Payload=json.dumps(lambda_parameters))
			invoke_return_values.append((invoke_ret_val.get('StatusCode'),
										invoke_ret_val.get('FunctionError')))

	return invoke_return_values

def lambda_handler(event,context):
	func=event['function']

	if func == 'match_images':
		return match_images(event)
	elif func == 'spawn_match_images_tasks':
		return spawn_match_images_tasks(event)
