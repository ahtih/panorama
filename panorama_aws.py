#!/usr/bin/python
# -*- encoding: latin-1 -*-

import sys,os.path,random,panorama

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

if '--match-batch' in keyword_args:
	keyword_args['--match-batch']	#!!!
else:
	image_fnames=positional_args[1:]

	processing_batch_key='%016x' % (random.randint(0,2**64-1),)
	print 'Creating processing batch %s with %u images' % (processing_batch_key,len(image_fnames))

	for fname in image_fnames:
		print 'Processing',fname
		img=panorama.ImageKeypoints(fname,True)
		print '   ','+'.join(map(str,img.channel_keypoints))
		img.save_to_s3(processing_batch_key + '/' + os.path.basename(fname))
