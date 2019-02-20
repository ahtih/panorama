#!/usr/bin/python
# -*- encoding: latin-1 -*-

import sys,operator,re,os.path,gc,itertools,multiprocessing
import panorama,kolor_xml_file,iterative_optimiser

max_procs=8	#!!!

def worker_func(args):
	try:
		global images_with_keypoints
		idx1,idx2=args

		result=panorama.find_matches(	images_with_keypoints[idx1],
										images_with_keypoints[idx2])
		output_str=''
		match_metrics=tuple()
		if len(result) > 1:
			match_metrics=result[2:]
			for x1,y1,x2,y2 in result[1]:
				output_str+='                <point x1="%d" y1="%d" x2="%d" y2="%d"/>\n' % (x1,y1,x2,y2)

		return args + (result[0],output_str,match_metrics)
	except KeyboardInterrupt:
		return None

def print_matches_for_images(output_fd):
	global max_procs,images_with_keypoints

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

	panorama.write_output_file_matches(output_fd,list(results),len(images_with_keypoints))

def read_lowlevel_matches_from_file(fname):
	image_fnames=[]
	for line in open(fname,'r'):
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

		yield (fnames_pair,line.strip(),angle_deg,count,score,x_shift,y_shift)

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

testcase_fnames=keyword_args.get('--testcase-fname')
if testcase_fnames:
	if not isinstance(testcase_fnames,list):
		testcase_fnames=(testcase_fnames,)

	for testcase_fname in testcase_fnames:
		kolor_xml_file.read_kolor_xml_file(testcase_fname,False,True)

	print_training_data=('--print-training-data' in keyword_args)
	training_data=[]
	optimiser_params=(('score',iterative_optimiser.FloatParam(0,1)),
					('count',iterative_optimiser.FloatParam(0,1)),
					('angle_deg_limit50',iterative_optimiser.FloatParam(-1,0)),
					('shift_ratio',iterative_optimiser.FloatParam(-1,0)),
					('triplets_error_deg',iterative_optimiser.FloatParam(-1,0)),
					)

	def optimiser_test_func(params):
		global training_data

		scores=[]
		total_correct_matches=0
		for e in training_data:
			scores.append((panorama.calc_classifier_decision_value(e[1:],params),e[0]))
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
		matches=dict()
		triplets_input=dict()
		for fnames_pair,line,angle_deg,count,score,x_shift,y_shift in \
															read_lowlevel_matches_from_file(matches_name):
			testcase_match_points=kolor_xml_file.matches.get(tuple(sorted(fnames_pair)))
			if testcase_match_points is None:
				if '--nowarn' not in keyword_args:
					print 'Warning: image pair %s %s not present in testcases' % tuple(fnames_pair)
				continue

			if score > 0:
				correct_match_rot=kolor_xml_file.image_quaternions[fnames_pair[0]].rotation_to_b(
														kolor_xml_file.image_quaternions[fnames_pair[1]])
				detected_match_rot=panorama.quaternion_from_match_angles(angle_deg,x_shift,y_shift)
				rotation_error_deg=correct_match_rot.rotation_to_b(detected_match_rot). \
																				total_rotation_angle_deg()
				if rotation_error_deg > 15 and rotation_error_deg < 40:
					continue	# Unclear if match rotation is the same as in testcase - skip this match

				tries+=1
				match_metrics=(score,count,angle_deg,x_shift,y_shift)
				matches[fnames_pair]=[(rotation_error_deg < 25),line,detected_match_rot,rotation_error_deg,
																							match_metrics]
				triplets_input[fnames_pair]=(detected_match_rot,match_metrics)
			else:
				tries+=1
				correct_predictions_with_zero_score+=int(not testcase_match_points)
				if not print_training_data and testcase_match_points:
					print True,-1.11111,line

		triplet_scores=panorama.calc_triplet_scores(triplets_input)

		for fnames_pair,(is_correct_match,line,q,rotation_error_deg,match_metrics) in matches.items():
			score,count,angle_deg,xd,yd=match_metrics
			shift_ratio=panorama.calc_shift_ratio(xd,yd)
			triplet_score=triplet_scores.get(fnames_pair,(30,-1000))[0]		#!!! Tune this
			classifier_input=(score,count,min(50,abs(angle_deg)),shift_ratio,triplet_score)

			training_data.append((int(is_correct_match),) + classifier_input)

			if print_training_data:
				print '%d %d %d %.2f %.4f %.2f' % training_data[-1]

			decision_value=panorama.calc_classifier_decision_value(classifier_input,
																				panorama.classifier_params)
			predicted=(decision_value >= 0)
			nonzero_successes+=int(predicted == is_correct_match)
			nonzero_tries+=1

			if not print_training_data and predicted != is_correct_match:
				print '%s %.2f %.3f %s %s %.0f %s' % (is_correct_match,rotation_error_deg,decision_value,
											fnames_pair[0],fnames_pair[1],triplet_score,classifier_input)

	if not print_training_data and tries:
		print 'Successes: %u/%u %.2f%% (nonzero links only)' % (nonzero_successes,nonzero_tries,
																	nonzero_successes*100.0/nonzero_tries)
		print 'Successes: %u/%u %.2f%% (hardcoded params                %s and threshold %+.3f)' % (
						nonzero_successes + correct_predictions_with_zero_score,
						tries,(nonzero_successes + correct_predictions_with_zero_score)*100.0/tries,
						' '.join(map(str,panorama.classifier_params[:-1])),panorama.classifier_params[-1])

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
	if '--dynamodb' in keyword_args:
		panorama.process_match_and_write_to_dynamodb(*positional_args)
		exit(0)

	image_fnames=positional_args

	images_with_keypoints=[panorama.ImageKeypoints(fname,True,(panorama.aws_session is not None)) \
																				for fname in image_fnames]
	output_fname=keyword_args.get('--output-fname')
	output_fd=sys.stdout
	if output_fname:
		output_fd=open(output_fname,'wt')

	panorama.write_output_file_header(output_fd,
									[(fname,img.focal_length_35mm,img.channel_keypoints) \
												for fname,img in zip(image_fnames,images_with_keypoints)]
									)
	print_matches_for_images(output_fd)
	panorama.write_output_file_footer(output_fd)
