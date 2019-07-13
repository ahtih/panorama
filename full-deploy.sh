#!/bin/sh

sls deploy &&
	echo 'Uploading geo-images.pickle...'
	aws --profile panorama-serverless-cli s3 cp viewable-images/geo-images.pickle s3://panorama-geo-images/ &&

	echo 'Getting Lambda endpoint ID...'
	echo -n "var rest_api_id='" > /dev/shm/rest-api-id.js
	aws --profile panorama-serverless-cli cloudformation describe-stack-resources --stack-name=panorama-dev --logical-resource-id=ApiGatewayRestApi |
			grep -w PhysicalResourceId |
			cut -d: -f2 |
			tr -d '", \n' >> /dev/shm/rest-api-id.js
	echo "';" >> /dev/shm/rest-api-id.js

	echo 'Uploading index.html...' &&
	for i in www.vr-eesti.ee www.vr-loodus.ee vr-eesti.ee vr-loodus.ee ; do
		echo "  $i" ;
		aws --profile panorama-serverless-cli s3 cp viewer-aframe/index.html s3://$i/ ;
		aws --profile panorama-serverless-cli s3 cp /dev/shm/rest-api-id.js s3://$i/ ;
		done
