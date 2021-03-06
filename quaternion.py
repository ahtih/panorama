import math,numpy

def cross_product_vec3(a,b):
	return (a[1]*b[2] - a[2]*b[1],a[2]*b[0] - a[0]*b[2],a[0]*b[1] - a[1]*b[0])

def dot_product_vec3(a,b):
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def vec3_len(v):
	return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

class Quaternion(list):
	def __init__(self,w,q1,q2,q3):
		self[:]=(w,q1,q2,q3)

	@staticmethod
	def from_single_axis_angle(axis_idx,angle):
			# Rotation is clockwise when viewed from positive side of its axis

		half_angle=angle * 0.5
		xyz=[0,0,0]
		xyz[axis_idx]=-math.sin(half_angle)
		return Quaternion(math.cos(half_angle),*xyz)

	@staticmethod
	def from_single_axis_angle_deg(axis_idx,angle_deg):
			# Rotation is clockwise when viewed from positive side of its axis
		return Quaternion.from_single_axis_angle(axis_idx,math.radians(angle_deg))

	@staticmethod
	def from_two_unit_vectors(a,b):
		# Returns quaternion for rotation from a to b

		axis=cross_product_vec3(a,b)
		angle_sin=vec3_len(axis)

		if angle_sin < 1e-20:
			return Quaternion(1,0,0,0)

		return Quaternion(1 + dot_product_vec3(a,b),*axis).normalised()

	def __mul__(self,r):
			# q1 * q2 rotates q2 by q1 in extrinsic (world) frame, which is
			#	the same as rotating q1 by q2 in intrinsic (q1) frame

		return Quaternion(	r[0]*self[0] - r[1]*self[1] - r[2]*self[2] - r[3]*self[3],
							r[0]*self[1] + r[1]*self[0] - r[2]*self[3] + r[3]*self[2],
							r[0]*self[2] + r[2]*self[0] - r[3]*self[1] + r[1]*self[3],
							r[0]*self[3] + r[3]*self[0] - r[1]*self[2] + r[2]*self[1])

	def normalised(self):
		l=math.sqrt(self[0]**2 + self[1]**2 + self[2]**2 + self[3]**2)
		if l < 1e-20:
			return self
		return Quaternion(self[0] / l,self[1] / l,self[2] / l,self[3] / l)

	def conjugate(self):
		return Quaternion(self[0],-self[1],-self[2],-self[3])

	def rotation_to_b(self,b):
		return self.conjugate() * b

	def total_rotation_angle(self):
		halfangle=0
		w=self[0]

		if abs(w) > 0.99999:
			x=1 - abs(w)
			if x < 1e-20:
				halfangle=0
			else:
				halfangle=math.exp(0.5 * math.log(x) + 0.3468)
				if w < 0:
					halfangle=math.pi - halfangle
		else:
			halfangle=math.acos(w)

		if halfangle*2 > math.pi:
			halfangle=math.pi - halfangle

		return 2 * halfangle

	def total_rotation_angle_deg(self):
		return math.degrees(self.total_rotation_angle())

	def z_basis_vector(self):
		x=2*(self[1]*self[3] + self[0]*self[2])
		y=2*(self[2]*self[3] - self[0]*self[1])
		z=1 - 2*(self[1]*self[1] + self[2]*self[2])
		return (x,y,z)

	def to_matrix(self):
		# Returns a post-multiplication active rotation matrix

		m=numpy.zeros((3,3))

		q_squares=[v*v for v in self]

		m[0,0]=1 - 2*(q_squares[2] + q_squares[3])
		m[1,1]=1 - 2*(q_squares[1] + q_squares[3])
		m[2,2]=1 - 2*(q_squares[1] + q_squares[2])

		a=self[1] * self[2]
		b=self[0] * self[3]
		m[0,1]=2*(a + b)
		m[1,0]=2*(a - b)

		a=self[1] * self[3]
		b=self[0] * self[2]
		m[0,2]=2*(a - b)
		m[2,0]=2*(a + b)

		a=self[2] * self[3]
		b=self[0] * self[1]
		m[1,2]=2*(a + b)
		m[2,1]=2*(a - b)

		return m

if __name__ == '__main__':
	import sys,panorama

	if len(sys.argv) >= 1+3*3:
		q1=quaternion_from_panorama_angles(*map(float,sys.argv[1:4]))
		q2=quaternion_from_panorama_angles(*map(float,sys.argv[4:7]))
		q3=quaternion_from_panorama_angles(*map(float,sys.argv[7:]))

		print q1.total_rotation_angle_deg()
		print q2.total_rotation_angle_deg()
		print q3.total_rotation_angle_deg()

		print
		qq=q1 * q2
		print qq.total_rotation_angle_deg()

		print
		qqq=qq.rotation_to_b(q3)
		print qqq.total_rotation_angle_deg()
	else:
		matches=dict()
		for fnames_pair,line,angle_deg,count,score,x_shift,y_shift in \
													panorama.read_matches_from_xml_file(sys.argv[1]):
			if score < 0:
				continue
			shift_ratio=panorama.calc_shift_ratio(x_shift,y_shift)
			decision_value=panorama.calc_classifier_decision_value(
							(score,count,min(50,abs(angle_deg)),shift_ratio),panorama.classifier_params)
			if decision_value < 0:
				continue
			#if abs(angle_deg) < 20 or abs(x_shift) < 20 or abs(y_shift) < 20:		#!!!
			#	continue
			matches[fnames_pair]=[quaternion_from_panorama_angles(angle_deg,x_shift,y_shift),0,0,
													decision_value,angle_deg,count,score,x_shift,y_shift]

		for fnames_pair1,match1 in matches.items():
			fname1=fnames_pair1[0]
			fname2=fnames_pair1[1]
			if fname1 > fname2:
				continue
			for fnames_pair2,match2 in matches.items():
				if fname2 in fnames_pair2 and fname1 not in fnames_pair2:
					reverse_pair2=(fname2 == fnames_pair2[1])
					fname3=fnames_pair2[0 if reverse_pair2 else 1]

					if fname2 > fname3:
						continue

					for reverse_pair3 in (False,True):
						match3=matches.get((fname3,fname1) if reverse_pair3 else (fname1,fname3))

						if not match3:
							continue

						q1=match1[0]

						q2=match2[0]
						if reverse_pair2:
							q2=q2.conjugate()

						q3=match3[0]
						if reverse_pair3:
							q3=q3.conjugate()

						error_deg=(q1 * q2).rotation_to_b(q3).total_rotation_angle_deg()

						# print fname1,fname2,fname3,error_deg

						for match in (match1,match2,match3):
							match[1]+=error_deg
							match[2]+=1

		for match in matches.values():
			if not match[2]:
				continue
			print match[1] / match[2],match[3]
