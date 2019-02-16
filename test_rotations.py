#!/usr/bin/python
# -*- encoding: latin-1 -*-

import math,panorama,quaternion,image_rotation_optimiser

is_test_ok=True

def test_euler_angles(euler_angles_deg):
	global is_test_ok

	m=panorama.quaternion_from_kolor_file(*map(math.radians,euler_angles_deg)).to_matrix()
	output_euler_angles_deg=map(math.degrees,image_rotation_optimiser.matrix_to_kolor_file_angles(m))
	total_error=sum([abs(a-b) for a,b in zip(euler_angles_deg,output_euler_angles_deg)])
	print '%6.3f   %3.0f %3.0f %3.0f     %7.2f %7.2f %7.2f' % \
								((total_error,) + tuple(euler_angles_deg) + tuple(output_euler_angles_deg))
	if total_error > 0.1:
		is_test_ok=False

test_euler_angles((0,0,0))

print

for idx in range(3):
	euler_angles_deg=[0,0,0]
	euler_angles_deg[idx]=30
	test_euler_angles(euler_angles_deg)

print

for rot_amount_deg in (10,30,70,-30,-5):
	for i in range(2*2*2):
		if not i:
			continue

		if not (i & (i-1)):
			continue
		euler_angles_deg=[0,0,0]
		for idx in range(3):
			if i & (1 << idx):
				euler_angles_deg[idx]=rot_amount_deg
		test_euler_angles(euler_angles_deg)

	print

print 'TESTS',('OK' if is_test_ok else 'FAILED')
