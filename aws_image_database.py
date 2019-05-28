import json,cPickle,boto3
import geo_images

GEO_IMAGES_FNAME='geo-images.pickle'

def get_next_image_links(event,context):
	global GEO_IMAGES_FNAME

	s3_bucket=boto3.resource('s3').Bucket('panorama-geo-images')
	geo_images.load_from_pickle(s3_bucket.Object(GEO_IMAGES_FNAME).get()['Body'].read())

	next_image_links=geo_images.select_next_image_links(event['pathParameters']['image_id'])

	response={
		'statusCode': 200,
		'headers': {'Access-Control-Allow-Origin': '*'},
		'body': json.dumps(next_image_links)
		}

	return response
