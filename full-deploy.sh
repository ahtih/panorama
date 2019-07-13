#!/bin/sh

sls deploy &&
	echo 'Uploading geo-images.pickle...'
	aws --profile panorama-serverless-cli s3 cp viewable-images/geo-images.pickle s3://panorama-geo-images/ &&

	echo 'Uploading index.html...' &&
	for i in www.vr-eesti.ee www.vr-loodus.ee ; do
		echo "  $i" ;
		aws --profile panorama-serverless-cli s3 cp viewer-aframe/index.html s3://$i/ ;
		done
