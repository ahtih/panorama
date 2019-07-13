module.exports.resources=(serverless) => {
	var resources={};

	var domains=[	'www.vr-eesti.ee',
						'vr-eesti.ee',
					'www.vr-loodus.ee',
						'vr-loodus.ee',
					];

	for (var i=0; i < domains.length;i++) {
		var domain=domains[i];

		resources['WebAppS3Bucket' + domain.replace(/[-.]/g,'')]=
					{'Type': 'AWS::S3::Bucket',
					'Properties': {
					        'BucketName': domain,
							'AccessControl': 'PublicRead',
							'WebsiteConfiguration': {
				        		'IndexDocument': 'index.html',
				        		'ErrorDocument': 'index.html',
					        	'RoutingRules': [
									{ 'RedirectRule': {
											'HostName': 'panorama-final-images.s3-website-eu-west-1.amazonaws.com'
											},
										'RoutingRuleCondition': {
											'HttpErrorCodeReturnedEquals': '404'
											}
										},
									]
								}
							}
						};
		}

/* !!! We'd like to have that block, but with it we get
			"WebAppS3BucketPolicy - API: s3:PutBucketPolicy Access Denied" during "sls deploy".
	See https://stackoverflow.com/questions/47931342/how-to-change-s3-bucket-policies-with-cloudformation

    WebAppS3BucketPolicy:
      Type: AWS::S3::BucketPolicy
      Properties:
        Bucket:
          Ref: WebAppS3Bucket
        PolicyDocument:
          Statement:
            - Sid: AllowPublicRead
              Effect: Allow
              Principal: "*"
              Action:
              - s3:GetObject
              Resource: arn:aws:s3:::www.vr-eesti.ee/*
	*/

	return resources;
	}
