import random,math

MAX_CHANGE_COEFF=6

global_best_fitness=None	# Read and written by other modules

class FloatParam:
	def __init__(self,min_value,max_value,initial_value=None,nr_of_allowed_values=30):
		self.min_value=min_value
		self.max_value=max_value
		if initial_value is None:
			initial_value=0.5 * (min_value + max_value)
		self.initial_value=initial_value

		values_range=self.max_value - self.min_value
		if values_range > 1e-6:
			self.digits_after_comma=int(math.ceil(-math.log10(values_range / float(nr_of_allowed_values))))
		else:
			self.digits_after_comma=3

		self.change_coeff_multiplier=(self.max_value - self.min_value) / float(2*MAX_CHANGE_COEFF)
		if self.max_value > self.min_value:
			self.min_change_coeff=(0.1 ** self.digits_after_comma) / self.change_coeff_multiplier

		self.format_string='%%.%df' % (max(0,self.digits_after_comma))

	def change_value(self,old_value,change_coeff,change_direction):
		if self.max_value <= self.min_value:
			return None

		value=old_value + change_direction * change_coeff * self.change_coeff_multiplier
		if value < self.min_value or value > self.max_value:
			return None

		return round(value,self.digits_after_comma)

	def random_value(self):
		return round(random.uniform(self.min_value,self.max_value),self.digits_after_comma)

class IntParam:
	def __init__(self,min_value,max_value,initial_value=None):
		self.min_value=min_value
		self.max_value=max_value
		if initial_value is None:
			initial_value=int(0.5 * (min_value + max_value) + 0.5)
		self.initial_value=initial_value
		self.format_string='%d'

	def change_value(self,old_value,change_coeff,change_direction):
		if self.max_value <= self.min_value:
			return None
		change_coeff=int(min(max(1,(self.max_value-self.min_value) / 3.0),change_coeff) + 0.5)
		if not change_coeff:
			return None

		value=min(self.max_value,max(self.min_value,old_value + change_coeff * change_direction))
		if value != old_value:
			return value
		return None

	def random_value(self):
		return random.randint(self.min_value,self.max_value)

def optimise_to_local_optimum(param_descs,test_func,params,tried_hashes,print_prefix=''):
	recalc_best_error=True
	change_coeff=3.0
	no_change_run_length=0
	best_fitness=None
	best_result_str=''

	min_change_coeff=0.01
	for desc in param_descs:
		if hasattr(desc,'min_change_coeff'):
			min_change_coeff=min(min_change_coeff,desc.min_change_coeff)

	change_coeff_format_string='%%.%df' % (max(0,int(math.ceil(-math.log10(min_change_coeff)))),)

	while change_coeff >= min_change_coeff:
		if print_prefix is not None:
			print print_prefix + change_coeff_format_string % change_coeff + ': ',

		if recalc_best_error:
			kwargs={}
			code_object=test_func.__code__
			if 'recalc_best_error' in code_object.co_varnames[:code_object.co_argcount]:
				kwargs['recalc_best_error']=True
			prev_best_fitness=best_fitness
			best_fitness,extra_print_string=test_func(params,**kwargs)

			best_result_str=' '.join([desc.format_string % (p,) for p,desc in zip(params,param_descs)]) + \
													(' fitness %.3f %s' % (best_fitness,extra_print_string))
			if print_prefix is not None:
				print best_result_str
				if prev_best_fitness > best_fitness*1.0001:
					print 'OPTIMISATION ANOMALY: BEST_FITNESS GOT WORSE'
		else:
			if print_prefix is not None:
				print

		changed_params=None
		recalc_best_error=False

		for param_idx,desc in enumerate(param_descs):
			for change_direction in (-1,+1):
				new_params=list(params)
				new_params[param_idx]=desc.change_value(new_params[param_idx],change_coeff,change_direction)
				if new_params[param_idx] is None:
					continue

				params_hash=hash(tuple(new_params))
				if params_hash in tried_hashes:
					continue
				tried_hashes.add(params_hash)

				fitness=test_func(new_params)[0]

				if best_fitness < fitness:
					best_fitness=fitness
					changed_params=list(new_params)

		if changed_params is None:
			no_change_run_length+=1
			change_coeff*=0.5 if no_change_run_length >= 2 else 0.7
		else:
			no_change_run_length=0

		if no_change_run_length >= 2:
			for param_idx1,desc1 in enumerate(param_descs):
				if not param_idx1:
					continue
				for param_idx2,desc2 in enumerate(param_descs[:param_idx1]):
					new_params=list(params)
					new_params[param_idx1]=desc1.random_value()
					new_params[param_idx2]=desc2.random_value()

					params_hash=hash(tuple(new_params))
					if params_hash in tried_hashes:
						continue
					tried_hashes.add(params_hash)

					fitness=test_func(new_params)[0]

					if best_fitness*1.0001 < fitness:
						best_fitness=fitness
						changed_params=list(new_params)
						change_coeff=max(3,change_coeff)

		if changed_params is not None:
			params=changed_params
			change_coeff=min(MAX_CHANGE_COEFF,change_coeff*1.5)
			recalc_best_error=True

	return (best_fitness,best_result_str,params)

def optimise(param_descs,test_func,nr_of_starting_points_to_try=1,print_debug=True):
	# If nr_of_starting_points_to_try is True, optimises indefinitely

	global global_best_fitness

	params=[desc.initial_value for desc in param_descs]
	tried_hashes=set()

	fitness,best_result_str,best_params=optimise_to_local_optimum(param_descs,test_func,params,tried_hashes,
																			'' if print_debug else None)
	best_params=list(best_params)
	if global_best_fitness is None or fitness > global_best_fitness:
		global_best_fitness=fitness

	starting_points_tried=1
	while (nr_of_starting_points_to_try is True or starting_points_tried < nr_of_starting_points_to_try):
		if print_debug:
			print 'Starting from another initial location'

		for i in range(3):
			param_idx=random.randint(0,len(param_descs)-1)
			params[param_idx]=param_descs[param_idx].random_value()

		fitness,result_str,_params=optimise_to_local_optimum(param_descs,test_func,params,
									tried_hashes,'%.3f ' % (global_best_fitness,) if print_debug else None)
		if fitness > global_best_fitness:
			global_best_fitness=fitness
			best_result_str=result_str
			best_params=list(_params)

		starting_points_tried+=1
		if print_debug:
			print 'Best result so far:',best_result_str

	return best_params
