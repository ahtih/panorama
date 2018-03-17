#!/usr/bin/python

# Program for viewing a 360 photosphere in a virtual reality headset

import os,time,traceback,numpy,openvr,openvr.gl_renderer,glfw,openvr.glframework.glfw_app,textwrap

from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader,compileProgram
from PIL import Image

MAX_IMG_SIZE=8192	# 16384

class SphericalPanorama(object):
	def __init__(self,image):
		self.image=image
		self.shader=None
		self.vao=None
		self.display_gl_time=0

	def set_texture_from_image(self):
		glBindTexture(GL_TEXTURE_2D,self.texture_handle)
		glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR)
		glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR)
		glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_REPEAT)
		glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_MIRRORED_REPEAT)
		glTexImage2D(GL_TEXTURE_2D,
					 0,
					 GL_RGB8,
					 self.image.shape[1], # width
					 self.image.shape[0], # height
					 0,
					 GL_RGB,
					 GL_UNSIGNED_BYTE,
					 self.image)
		glBindTexture(GL_TEXTURE_2D,0)

	def init_gl(self):
		self.vao=glGenVertexArrays(1)
		glBindVertexArray(self.vao)

		# Set up photosphere image texture for OpenGL
		self.texture_handle=glGenTextures(1)

		self.set_texture_from_image()

		# Set up shaders for rendering
		vertex_shader = compileShader(textwrap.dedent(
				"""#version 450 core
				#line 46
				
				layout(location = 1) uniform mat4 projection = mat4(1);
				layout(location = 2) uniform mat4 model_view = mat4(1);

				out vec3 viewDir;
				
				// projected screen quad
				const vec4 SCREEN_QUAD[4] = vec4[4](
					vec4(-1, -1, 1, 1),
					vec4( 1, -1, 1, 1),
					vec4( 1,  1, 1, 1),
					vec4(-1,  1, 1, 1));
				
				const int TRIANGLE_STRIP_INDICES[4]=int[4](0,1,3,2);
				
				void main() 
				{
					int vertexIndex = TRIANGLE_STRIP_INDICES[gl_VertexID];
					gl_Position = vec4(SCREEN_QUAD[vertexIndex]);
					mat4 xyzFromNdc = inverse(projection * model_view);
					vec4 campos = xyzFromNdc * vec4(0, 0, 0, 1);
					vec4 vpos = xyzFromNdc * SCREEN_QUAD[vertexIndex];
					viewDir = vpos.xyz/vpos.w - campos.xyz/campos.w;
				}
				"""),
				GL_VERTEX_SHADER)
		fragment_shader = compileShader(textwrap.dedent(
				"""\
				#version 450 core
				#line 85
		
				layout(binding = 0) uniform sampler2D image;
				
				in vec3 viewDir;
		
				out vec4 pixelColor;
				
				const float PI = 3.1415926535897932384626433832795;
				
				void main() 
				{
					vec3 d = viewDir;
					float longitude = 0.5 * atan(d.z, d.x) / PI + 0.5; // range [0-1]
					float r = length(d.xz);
					float latitude = -atan(d.y, r) / PI + 0.5; // range [0-1]
					
					pixelColor=texture(image,vec2(longitude,latitude));
				}
				"""),
				GL_FRAGMENT_SHADER)
		self.shader = compileProgram(vertex_shader, fragment_shader)

	def display_gl(self,modelview,projection):
		self.display_gl_time-=time.time()
		glBindVertexArray(self.vao)
		glBindTexture(GL_TEXTURE_2D,self.texture_handle)
		glUseProgram(self.shader)
		glUniformMatrix4fv(1,1,False,projection)
		glUniformMatrix4fv(2,1,False,modelview)
		glDrawArrays(GL_TRIANGLE_STRIP,0,4)
		self.display_gl_time+=time.time()

	def dispose_gl(self):
		if self.vao:
			glDeleteTextures([self.texture_handle])
			if self.shader is not None:
				glDeleteProgram(self.shader)
			glDeleteVertexArrays(1, [self.vao])

def load_image(fname):
	# Open equirectangular photosphere

	if fname.endswith('.npy'):
		return numpy.load(fname,'r',False)

	im=Image.open(fname)

	if max(im.size) > MAX_IMG_SIZE:
		coeff=MAX_IMG_SIZE / float(max(im.size))
		new_size=[]
		for value in im.size:
			new_size.append(min(MAX_IMG_SIZE,int(value*coeff)))

		print 'Resizing image from %dx%d to %dx%d' % (tuple(im.size) + tuple(new_size))
		im=im.resize(new_size,Image.ANTIALIAS)

	return numpy.array(im)

if __name__ == "__main__":
	import sys

	images=[load_image(fname) for fname in sys.argv[1:]]
	cur_image_idx=0

	# numpy.save('test',images[0])

	actor=SphericalPanorama(images[cur_image_idx])
	renderer=openvr.gl_renderer.OpenVrGlRenderer(actor)

	with openvr.glframework.glfw_app.GlfwApp(renderer,'Photosphere') as app:
		# app.run_loop()

		app.init_gl()
		# renderer.compositor.setExplicitTimingMode(
		#							openvr.VRCompositorTimingMode_Explicit_RuntimePerformsPostPresentHandoff)
		frames_displayed=0
		last_print_time=time.time()
		display_gl_time=0
		getposes_time=0

		poses_t=openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount	#!!!!
		poses=poses_t()

		render_times_ms=[]
		ev=openvr.VREvent_t()
		while not glfw.window_should_close(app.window):

			frame_start_time=time.time()
			# app.render_scene()

			# app.render_scene() replacement:
			cur_frame_timings=[]
			app.init_gl()
			glfw.make_context_current(app.window)

			result=renderer.compositor.waitGetPoses(renderer.poses,len(renderer.poses),None,0)
			if result != 0:
				print 'compositor error'
				exit(0)
			cur_frame_timings.append(int((time.time() - frame_start_time) * 1000))

			app.renderer.render_scene()
			cur_frame_timings.append(int((time.time() - frame_start_time) * 1000))
			# glFlush()
			# renderer.compositor.postPresentHandoff()
			glfw.poll_events()
			cur_frame_timings.append(int((time.time() - frame_start_time) * 1000))

			render_times_ms.append(cur_frame_timings)

			# getposes_time-=time.time()
			# print renderer.compositor.waitGetPoses(poses,len(poses),None,0)
			# print(poses[openvr.k_unTrackedDeviceIndex_Hmd].mDeviceToAbsoluteTracking)
			# getposes_time+=time.time()

			if renderer.vr_system.pollNextEvent(ev):
				if ev.eventType == openvr.VREvent_ButtonPress:
					print 'button pressed'

					result,controller_state=renderer.vr_system.getControllerState(ev.trackedDeviceIndex)
					if result:
						if controller_state.ulButtonPressed & (1 << openvr.k_EButton_ApplicationMenu):
							print 'Exiting by controller button press'
							break
						if controller_state.ulButtonPressed & (1 << openvr.k_EButton_SteamVR_Touchpad):
							cur_image_idx+=(-1 if controller_state.rAxis[0].x < 0 else +1)
							cur_image_idx%=len(images)
							actor.image=images[cur_image_idx]
							actor.set_texture_from_image()
							print 'actor.set_texture_from_image() call done'

			frames_displayed+=1
			if (frames_displayed % 50) == 0 and frames_displayed:
				# print getposes_time		#!!!!
				cur_time=time.time()

				time_passed=cur_time - last_print_time
				print 'Image #%d, %d frames displayed, %.0ffps, display_gl() takes %.0f%% of time' % \
													(cur_image_idx,frames_displayed,50 / float(time_passed),
													actor.display_gl_time * 100 / float(time_passed))
				# print ' '.join(','.join(map(str,tim)) for tim in render_times_ms)

				last_print_time=cur_time
				actor.display_gl_time=0
				getposes_time=0
				render_times_ms=[]

		print 'Exiting'
