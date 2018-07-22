#!/usr/bin/python
# -*- encoding: latin-1 -*-

import sys,operator,re,xml.sax.handler,xml.sax,gc,itertools,multiprocessing
import panorama,iterative_optimiser

max_procs=8	#!!!

def worker_func(args):
	try:
		global images_with_keypoints
		idx1,idx2=args

		debug_str,matched_points=panorama.find_matches(	images_with_keypoints[idx1],
														images_with_keypoints[idx2])
		output_str=''
		for x1,y1,x2,y2 in matched_points:
			output_str+='                <point x1="%d" y1="%d" x2="%d" y2="%d"/>\n' % (x1,y1,x2,y2)

		return args + (debug_str,output_str)
	except KeyboardInterrupt:
		return None

def print_matches_for_images(output_fd):
	global max_procs,images_with_keypoints

	link_stats=[[] for img in images_with_keypoints]

	gc.collect()
	worker_pool=multiprocessing.Pool(max_procs) if max_procs > 1 else None

	worker_args=[]
	for idx1,img1 in enumerate(images_with_keypoints):
		for idx2,img2 in enumerate(images_with_keypoints):
			if idx2 > idx1:
				worker_args.append((idx1,idx2))
	if worker_pool is not None:
		results=worker_pool.imap(worker_func,worker_args,5)		#!!! Try imap_unordered()
	else:
		results=itertools.imap(worker_func,worker_args)

	for idx1,idx2,debug_str,output_string in results:
		print >>output_fd,'        <!-- image %d<-->%d: %s -->' % (idx1,idx2,debug_str)

		if output_string:
			print >>output_fd,'        <match image1="%d" image2="%d">\n            <points>\n%s            </points>\n        </match>' % \
																				(idx1,idx2,output_string)
			link_stats[idx1].append(idx2)
			link_stats[idx2].append(idx1)

	print >>output_fd,'<!-- Link stats: -->'

	for idx,linked_images in enumerate(link_stats):
		print >>output_fd,'<!-- #%d links: %s -->' % (idx,' '.join(map(str,linked_images)))

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

if '--aws-profile' in keyword_args:
	panorama.init_aws_session(keyword_args['--aws-profile'])

if not positional_args:
	print '''Usage:
	%s IMAGE-FNAME
		Extract and show keypoints from a single image

	%s [--output-fname=PANO-XML-FNAME] IMAGE-FNAME IMAGE-FNAME [IMAGE-FNAME ...]
		Match a set of images to each other, and write the resulting AutoPano XML file to stdout or output file

	%s --testcase-fname=PANO-XML-FNAME ... [--print-training-data] [--nowarn] MATCHES-XML-FNAME ...
		Optimise the matches classifier (decision_value formula) against testcases, using raw matches files as input''' % \
		((sys.argv[0],) * 3)

	exit(1)

correct_matches=None
testcase_fnames=keyword_args.get('--testcase-fname')
if testcase_fnames:
	if not isinstance(testcase_fnames,list):
		testcase_fnames=(testcase_fnames,)

	correct_matches=dict()

	class kolor_xml_handler(xml.sax.handler.ContentHandler):
		def __init__(self):
			self.image_fnames=[]

		def startElement(self,name,attrs):
			global correct_matches

			if name == 'def':
				fname=attrs.get('filename')
				for fname2 in self.image_fnames:
					correct_matches[tuple(sorted((fname,fname2)))]=False
				self.image_fnames.append(fname)
			elif name == 'match':
				img1=self.image_fnames[int(attrs.get('image1'))]
				img2=self.image_fnames[int(attrs.get('image2'))]
				correct_matches[tuple(sorted((img1,img2)))]=True

	for testcase_fname in testcase_fnames:
		parser=xml.sax.make_parser()
		parser.setContentHandler(kolor_xml_handler())
		parser.parse(open(testcase_fname,'r'))

	print_training_data=('--print-training-data' in keyword_args)
	training_data=[]
	optimiser_params=(('score',iterative_optimiser.FloatParam(0,1)),
					('count',iterative_optimiser.FloatParam(0,1)),
					('angle_deg_limit50',iterative_optimiser.FloatParam(-1,0)),
					('shift_ratio',iterative_optimiser.FloatParam(-1,0)),
					)

	def optimiser_test_func(params):
		global training_data

		scores=[]
		total_correct_matches=0
		for e in training_data:
			scores.append((calc_classifier_decision_value(e[1:],params),e[0]))
			total_correct_matches+=int(bool(e[0]))

		scores.sort()

		best_correct_predictions=0
		best_threshold=0

		# print

		cumulative_correct_matches=0
		prev_score=None
		for idx,(score,is_correct_match) in enumerate(scores):
			correct_predictions_at_this_threshold=(idx - cumulative_correct_matches) + \
													(total_correct_matches - cumulative_correct_matches)
			# print idx,is_correct_match,cumulative_correct_matches,total_correct_matches,correct_predictions_at_this_threshold,score

			cumulative_correct_matches+=int(bool(is_correct_match))
			if correct_predictions_at_this_threshold > best_correct_predictions:
				best_correct_predictions=correct_predictions_at_this_threshold
				if prev_score is None:
					prev_score=score
				best_threshold=0.5 * (score + prev_score)
				prev_score=score

		return (best_correct_predictions,'%+.3f' % (best_threshold,))

	tries=0
	nonzero_tries,nonzero_successes=0,0
	correct_predictions_with_zero_score=0

	for matches_name in positional_args:
		image_fnames=[]
		for line in open(matches_name,'r'):
			if line.strip().endswith(' keypoints') or line.strip().endswith(' keypoints -->'):
				image_fnames.append(line.rpartition('/')[2].partition(' ')[0])
				continue

			m=re.search(r'<!-- image ([0-9]+)<-->([0-9]+): .* ([-+]*[0-9.]+)deg, score ([0-9.]+)/[0-9.]+=([0-9.]+), shift ([-+]*[0-9.]+)deg, *([-+]*[0-9.]+)deg',line)
			if not m:
				continue

			fields=m.groups()
			img_idx1=int(fields[0])
			img_idx2=int(fields[1])
			angle_deg=float(fields[2])
			count=float(fields[3])
			score=float(fields[4])
			x_shift=int(fields[5])
			y_shift=int(fields[6])

			fnames_pair=tuple(sorted((image_fnames[img_idx1],image_fnames[img_idx2])))
			is_correct_match=correct_matches.get(fnames_pair)
			if is_correct_match is None:
				if '--nowarn' not in keyword_args:
					print 'Warning: image pair %s %s not present in testcases' % fnames_pair
				continue

			if score > 0:
				shift_ratio=calc_shift_ratio(x_shift,y_shift)
				training_data.append((int(is_correct_match),score,count,min(50,abs(angle_deg)),shift_ratio))

				if print_training_data:
					print '%d 1:%s 2:%s 3:%s 4:%s' % training_data[-1]

				decision_value=calc_classifier_decision_value(
										(score,count,min(50,abs(angle_deg)),shift_ratio),classifier_params)

				predicted=(decision_value >= 0)
				nonzero_successes+=int(predicted == is_correct_match)
				nonzero_tries+=1
			else:
				predicted=False
				decision_value=-1.11111
				correct_predictions_with_zero_score+=int(not is_correct_match)

			tries+=1

			if not print_training_data and predicted != is_correct_match:
				print is_correct_match,decision_value,line.strip()

	if not print_training_data and tries:
		print 'Successes: %u/%u %.2f%% (nonzero links only)' % (nonzero_successes,nonzero_tries,
																	nonzero_successes*100.0/nonzero_tries)
		print 'Successes: %u/%u %.2f%% (hardcoded params                %s and threshold %+.3f)' % (
								nonzero_successes + correct_predictions_with_zero_score,
								tries,(nonzero_successes + correct_predictions_with_zero_score)*100.0/tries,
								' '.join(map(str,classifier_params[:-1])),classifier_params[-1])

		# print 'Optimising with the following parameters:', \
		#									' '.join(map(operator.itemgetter(0),optimiser_params))
		best_params=iterative_optimiser.optimise([p[1] for p in optimiser_params],
																			optimiser_test_func,30,False)
		iterative_optimiser_successes,threshold_str=optimiser_test_func(best_params)
		print 'Successes: %u/%u %.2f%% (iterative_optimiser with params %s and threshold %s)' % \
							(iterative_optimiser_successes + correct_predictions_with_zero_score,
							tries,
							(iterative_optimiser_successes + correct_predictions_with_zero_score) * \
																							100.0/tries,
							' '.join(map(str,best_params)),threshold_str)

elif len(positional_args) == 1:
	ikp=panorama.ImageKeypoints(positional_args[0])
	if panorama.aws_session:
		print 'Writing keypoints to S3'
		ikp.save_to_s3('img2.keypoints')	#!!!
	else:
		ikp.show_img_with_keypoints(0)
else:
	image_fnames=positional_args

	if '--dynamodb' in keyword_args:
		process_match_and_write_to_dynamodb(*image_fnames[:2])
		exit(0)

	images_with_keypoints=[panorama.ImageKeypoints(fname,True,(panorama.aws_session is not None)) \
																				for fname in image_fnames]
	output_fd=sys.stdout
	output_fname=keyword_args.get('--output-fname')
	if output_fname:
		output_fd=open(output_fname,'wt')

	print >>output_fd,'''<?xml version="1.0" encoding="UTF-8"?>
<pano>
    <version filemodel="2.0" application="Autopano Pro 4.4.1" id="9"/>
    <finalRender basename="%a" path="%p" fileType="jpg" fileCompression="5" fileDepth="8" interpolationMode="3" blendMode="2" outputPercent="100" overwrite="1" fileEmbedAll="0" removeAlpha="0" outputPanorama="1" outputLayers="0" outputPictures="0" multibandLevel="-2" alphaDiamond="0" exposureWeights="0" cutting="1" graphcutGhostFocal="0" bracketedGhost="0"/>
    <optim>
        <options automaticSteps="0" automaticSettings="0" focalHandling="-1" distoHandling="-1" offsetsHandling="-1" multipleVPHandling="0" yprScope="0" focalScope="2" distoScope="2" offsetScope="2" hScope="0" optLG="1" useGO="1" matrixLG="0" gridLG="0" optFinal1="1" optLens1="1" optLens2="1" stepGeomAnalysis="1" cleanPoints="0" cleanLinks="0" cbpMode="0" cbpThreshold="5" cbpLimit="50" cbpLinkThreshold="40" matLGMode="1" matLGRow="1" matLGStack="1" matLG360="0" matLGOverlapping="25" gaMode="0" calibratedRig="0"/>
    </optim>
    <colorCorrection eqMode="1" eqPerLayer="1" colorDtScaler="1"/>
    <exposureWeighting enabled="0" tone="0.5" dark="0.5" light="0.5"/>
    <projection fitMode="1" type="0">
        <params/>
    </projection>
    <images>
'''

	for fname,img in zip(image_fnames,images_with_keypoints):
		print >>output_fd,('<image><def filename="%s" focal35mm="%.3f" lensModel="0" ' + \
										'fisheyeRadius="0" fisheyeCoffX="0" fisheyeCoffY="0"/></image>') % \
									(fname,img.focal_length_35mm or 0)
		print >>output_fd,'<!-- %s %s keypoints -->' % (fname,'+'.join(map(str,img.channel_keypoints)))

	print >>output_fd,'''
    </images>
    <layers>
        <layer name="N_0" ouput="1">
            <images>
'''

	for idx in range(len(images_with_keypoints)):
		print >>output_fd,'                <image index="%d" preview="1" output="1"/>' % (idx,)

	print >>output_fd,'''
            </images>
        </layer>
    </layers>
    <stacks>
'''

	for idx in range(len(images_with_keypoints)):
		print >>output_fd,'        <stack>%d</stack>' % (idx,)

	print >>output_fd,'''
    </stacks>
    <matches>
'''
	print_matches_for_images(output_fd)

	print >>output_fd,'''
    </matches>
</pano>
'''
