#!/usr/bin/python
# -*- coding: latin-1

# Program for viewing a 360 photosphere in a VR headset

import os,math,time,traceback
import numpy,openvr,openvr.gl_renderer,glfw,openvr.glframework.glfw_app,textwrap
import OpenGL.arrays.vbo
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader,compileProgram
from PIL import Image,ImageDraw,ImageFont
import geo_images

MAX_QUALITY_CODE=13		# 8192 pixels x size
VIEWABLE_IMAGES_PATH='viewable-images'

next_image_links=dict()		# [id]=(distance_meters,world_azimuth_deg,center_azimuth_deg)

class SphericalPanorama(object):
	def __init__(self,image,center_azimuth_deg=0):
		self.image=image
		self.center_azimuth_deg=center_azimuth_deg
		self.shader=None
		self.vao=None
		self.display_gl_time=0

	def update(self):
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

		az=math.radians(self.center_azimuth_deg + 180)
		s=math.sin(az)
		c=math.cos(az)
		self.azimuth_rotation_matrix=numpy.matrix((	( c, 0,-s, 0),
													( 0, 1, 0, 0),
													( s, 0, c, 0),
													( 0, 0, 0, 1),
													))

	def init_gl(self):
		self.vao=glGenVertexArrays(1)
		glBindVertexArray(self.vao)

		# Set up photosphere image texture for OpenGL
		self.texture_handle=glGenTextures(1)

		self.update()

		# Set up shaders for rendering
		vertex_shader=compileShader(textwrap.dedent(
				"""#version 450 core
				#line 46

				layout(location=1) uniform mat4 projection=mat4(1);
				layout(location=2) uniform mat4 model_view=mat4(1);

				out vec3 viewDir;

				// projected screen quad
				const vec4 SCREEN_QUAD[4]=vec4[4](
					vec4(-1, -1, 1, 1),
					vec4( 1, -1, 1, 1),
					vec4( 1,  1, 1, 1),
					vec4(-1,  1, 1, 1));

				const int TRIANGLE_STRIP_INDICES[4]=int[4](0,1,3,2);

				void main() 
				{
					int vertexIndex=TRIANGLE_STRIP_INDICES[gl_VertexID];
					gl_Position=vec4(SCREEN_QUAD[vertexIndex]);
					mat4 xyzFromNdc=inverse(projection * model_view);
					vec4 campos=xyzFromNdc * vec4(0,0,0,1);
					vec4 vpos=xyzFromNdc * SCREEN_QUAD[vertexIndex];
					viewDir=vpos.xyz/vpos.w - campos.xyz/campos.w;
				}
				"""),
				GL_VERTEX_SHADER)
		fragment_shader=compileShader(textwrap.dedent(
				"""#version 450 core
				#line 85

				layout(binding=0) uniform sampler2D image;
				in vec3 viewDir;
				out vec4 pixelColor;

				const float PI=3.1415926535897932384626433832795;

				void main() 
				{
					vec3 d=viewDir;
					float longitude=0.5 * atan(d.z, d.x) / PI + 0.5; // range [0-1]
					float r=length(d.xz);
					float latitude=-atan(d.y, r) / PI + 0.5; // range [0-1]
					
					pixelColor=texture(image,vec2(longitude,latitude));
				}
				"""),
				GL_FRAGMENT_SHADER)
		self.shader=compileProgram(vertex_shader,fragment_shader)

	def display_gl(self,modelview,projection):
		self.display_gl_time-=time.time()

		m=numpy.asarray(numpy.matmul(modelview.T,self.azimuth_rotation_matrix).T)

		glBindVertexArray(self.vao)
		glBindTexture(GL_TEXTURE_2D,self.texture_handle)
		glUseProgram(self.shader)
		glUniformMatrix4fv(1,1,False,projection)
		glUniformMatrix4fv(2,1,False,m)
		glDrawArrays(GL_TRIANGLE_STRIP,0,4)
		glBindVertexArray(0)

		self.display_gl_time+=time.time()

	def dispose_gl(self):
		if self.vao is not None:
			if self.texture_handle is not None:
				glDeleteTextures([self.texture_handle])
			if self.shader is not None:
				glDeleteProgram(self.shader)
			glDeleteVertexArrays(1,[self.vao])

class NextImageLinksActor(object):
	def __init__(self,next_image_links=dict()):
		self.next_image_links=next_image_links
		self.shader=None
		self.vao=None
		self.vbo=None
		self.last_modelview_matrix=None
		self.texture_images_list=[]

		try:
			self.font=ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',48)
		except IOError:
			self.font=None

		self.next_image_sprite=numpy.array(Image.open('orange-donut.png'))
		self.north_sprite=numpy.array(Image.open('north.png'))

	def calc_texture_coords(self,sprite_image_numpy,x_fraction,y_fraction):
		base_y=0
		for img in self.texture_images_list:
			if img.__array_interface__['data'] == sprite_image_numpy.__array_interface__['data']:
				return (x_fraction * img.shape[1] / float(self.texture_image_shape[1]),
						(base_y + 0.5 + (1-y_fraction) * (img.shape[0]-1)) / \
																float(self.texture_image_shape[0]))
			base_y+=img.shape[0]

		return None

	def calc_sprite_point(self,world_azimuth_deg,elevation_deg):
		az=-math.radians(world_azimuth_deg)
		# az=0 means panorama image horizontal wrap point (image x coordinate=0)
		elevation=math.radians(elevation_deg)
		return (-math.cos(az) * math.cos(elevation),
				math.sin(elevation),
				math.sin(az) * math.cos(elevation),
				1)

	def calc_sprite_vertexes(self,sprite_image_numpy,world_azimuth_deg,elevation_deg,y_size_deg):
		x_size_deg=y_size_deg * sprite_image_numpy.shape[1] / float(sprite_image_numpy.shape[0])

		vertexes=[]
		for offset_x,offset_y in (	(0,0),(0,1),(1,1),	# Triangle 1
									(1,1),(0,0),(1,0)):	# Triangle 2
			vertexes.append(self.calc_sprite_point(world_azimuth_deg + (offset_x-0.5)*x_size_deg,
													elevation_deg + (offset_y-0.5)*y_size_deg) + \
							self.calc_texture_coords(sprite_image_numpy,offset_x,offset_y))
		return vertexes

	@staticmethod
	def next_image_link_elevation_deg(distance_meters):
		return -35 + 3.5*math.log(distance_meters)

	def add_textual_sprite(self,text):
		im=Image.new('RGBA',(1,1),(0,0,0,0))
		draw=ImageDraw.Draw(im)
		img_size=draw.textsize(text,font=self.font)

		im=Image.new('RGBA',img_size,(0,0,0,0))
		draw=ImageDraw.Draw(im)
		draw.text((0,0),text,font=self.font)
		del draw

		img_numpy=numpy.array(im)

		self.texture_images_list.append(img_numpy)
		return img_numpy

	def update(self):
		self.texture_images_list=[]
		self.texture_images_list.append(self.next_image_sprite)
		self.texture_images_list.append(self.north_sprite)

		next_image_text_sprites=dict()
		for id,(distance_meters,world_azimuth_deg,center_azimuth_deg) in self.next_image_links.items():
			if distance_meters < 1000:
				text='%.0fm' % (distance_meters,)
			elif distance_meters < 3000:
				text='%.1fkm' % (distance_meters/1000.0,)
			else:
				text='%.0fkm' % (distance_meters/1000.0,)

			next_image_text_sprites[id]=self.add_textual_sprite(text)

			# Create combined texture image from sprites

		combined_texture_width=0
		for img_numpy in self.texture_images_list:
			combined_texture_width=max(combined_texture_width,img_numpy.shape[1])

		texture_image=numpy.concatenate([numpy.pad(img_numpy,
												((0,0),(0,combined_texture_width-img_numpy.shape[1]),(0,0)),
												'constant') \
														for img_numpy in self.texture_images_list],0)
		self.texture_image_shape=texture_image.shape

		glBindTexture(GL_TEXTURE_2D,self.texture_handle)
		glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR)
		glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR)
		glTexImage2D(GL_TEXTURE_2D,
					 0,
					 GL_RGBA8,
					 texture_image.shape[1], # width
					 texture_image.shape[0], # height
					 0,
					 GL_RGBA,
					 GL_UNSIGNED_BYTE,
					 texture_image)
		glBindTexture(GL_TEXTURE_2D,0)

		sprite_vertexes=self.calc_sprite_vertexes(self.north_sprite,0,25,5)

		for id,(distance_meters,world_azimuth_deg,center_azimuth_deg) in self.next_image_links.items():
			size_deg=max(0.7,8 - 0.5*math.log(distance_meters))
			elevation_deg=self.next_image_link_elevation_deg(distance_meters)
			sprite_vertexes+=self.calc_sprite_vertexes(self.next_image_sprite,world_azimuth_deg,
																				elevation_deg,size_deg)
			text_size_deg=max(1,0.5 * size_deg)
			sprite_vertexes+=self.calc_sprite_vertexes(next_image_text_sprites[id],world_azimuth_deg,
											elevation_deg - (size_deg + text_size_deg)/2.0,text_size_deg)

		self.vbo.set_array(numpy.array(sprite_vertexes,'f'))

	def select_active_next_image_link(self):
		if self.last_modelview_matrix is None:
			return None

		best_image_id=None
		best_dist=0.25**2
		for id,(distance_meters,world_azimuth_deg,center_azimuth_deg) in self.next_image_links.items():
			v=numpy.array(self.calc_sprite_point(world_azimuth_deg,
													self.next_image_link_elevation_deg(distance_meters)))
			v_in_camera_frame=numpy.matmul(v,self.last_modelview_matrix)
			if v_in_camera_frame[2] < 0:
				dist=v_in_camera_frame[0]**2 + v_in_camera_frame[1]**2
				if best_dist > dist:
					best_dist=dist
					best_image_id=id

		return best_image_id

	def init_gl(self):
		self.vao=glGenVertexArrays(1)
		glBindVertexArray(self.vao)
		self.vbo=OpenGL.arrays.vbo.VBO(numpy.array([],'f'))

		self.texture_handle=glGenTextures(1)

		self.update()

		vertex_shader=compileShader(textwrap.dedent(
				"""#version 450 core

				layout(location=1) uniform mat4 projection=mat4(1);
				layout(location=2) uniform mat4 model_view=mat4(1);

				layout(location=0) in vec4 v;
				layout(location=1) in vec2 in_textCoord;
				out vec2 textCoord;

				void main() 
				{
					gl_Position=(projection * model_view) * v;
					textCoord=in_textCoord;
				}
				"""),
				GL_VERTEX_SHADER)
		fragment_shader=compileShader(textwrap.dedent(
				"""#version 450 core

				layout(binding=0) uniform sampler2D image;
				in vec2 textCoord;
				out vec4 pixelColor;

				void main() 
				{
					vec4 color=texture(image,textCoord);
					if (color.a < 0.3)
						discard;
					pixelColor=color;
				}
				"""),
				GL_FRAGMENT_SHADER)
		self.shader=compileProgram(vertex_shader,fragment_shader)

	def display_gl(self,modelview,projection):
		m=modelview.copy()
		m[3][0]=0	# Eliminate any translation in matrix (moving the camera position)
		m[3][1]=0
		m[3][2]=0
		m[0][3]=0
		m[1][3]=0
		m[2][3]=0
		m[3][3]=1

		self.last_modelview_matrix=m

		glBindVertexArray(self.vao)
		self.vbo.bind()

		glEnableVertexAttribArray(0)
		glVertexAttribPointer(0,4,GL_FLOAT,GL_FALSE,(4+2)*4,self.vbo + 0)

		glEnableVertexAttribArray(1)
		glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,(4+2)*4,self.vbo + 4*4)

		glBindTexture(GL_TEXTURE_2D,self.texture_handle)
		glUseProgram(self.shader)
		glUniformMatrix4fv(1,1,False,projection)
		glUniformMatrix4fv(2,1,False,m)
		glDrawArrays(GL_TRIANGLES,0,len(self.vbo))
		self.vbo.unbind()
		glBindVertexArray(0)

	def dispose_gl(self):
		if self.vao is not None:
			glDeleteVertexArrays(1,[self.vao])
			if self.texture_handle is not None:
				glDeleteTextures([self.texture_handle])
		if self.shader is not None:
			glDeleteProgram(self.shader)

def load_image(fname_or_id):
	# Open equirectangular photosphere
	global MAX_QUALITY_CODE

	if isinstance(fname_or_id,int) or \
						(geo_images.is_valid_image_id(fname_or_id) and not os.access(fname_or_id,os.F_OK)):
		for quality_code in range(MAX_QUALITY_CODE,10-1,-1):
			viewable_images_fname=VIEWABLE_IMAGES_PATH + '/' + str(fname_or_id) + '/' + \
																				str(quality_code) + '.jpg'
			if os.access(viewable_images_fname,os.F_OK):
				fname_or_id=viewable_images_fname
				break
		else:
			fname_or_id=str(fname_or_id)

	print 'Loading',fname_or_id

	if fname_or_id.endswith('.npy'):
		return numpy.load(fname_or_id,'r',False)

	im=Image.open(fname_or_id)

	max_img_size=2**MAX_QUALITY_CODE
	if max(im.size) > max_img_size:
		coeff=max_img_size / float(max(im.size))
		new_size=[]
		for value in im.size:
			new_size.append(min(max_img_size,int(value*coeff + 0.5)))

		print 'Resizing image from %dx%d to %dx%d' % (tuple(im.size) + tuple(new_size))
		im=im.resize(new_size,Image.ANTIALIAS)

	return numpy.array(im)

def go_to_image(id):
	global cur_image_id,cmdline_fnames,next_image_links

	cur_image_id=id

	img=load_image(cmdline_fnames[id] if cmdline_fnames else id)

	if not cmdline_fnames:
		next_image_links=geo_images.select_next_image_links(cur_image_id)

	return (img,next_image_links,geo_images.geo_images.get(cur_image_id,(None,None,0))[2])

if __name__ == "__main__":
	import sys

	cmdline_fnames=sys.argv[1:]
	if not cmdline_fnames:
		geo_images.load_geo_images_list(VIEWABLE_IMAGES_PATH)
		cur_image_id=max(geo_images.geo_images.keys())
	else:
		cur_image_id=0

	center_azimuth_manual_changes=dict()

	img,next_image_links,center_azimuth_deg=go_to_image(cur_image_id)
	# numpy.save('test',img)

	panorama_actor=SphericalPanorama(img,center_azimuth_deg)
	next_image_links_actor=NextImageLinksActor(next_image_links)
	renderer=openvr.gl_renderer.OpenVrGlRenderer((panorama_actor,next_image_links_actor))

	with openvr.glframework.glfw_app.GlfwApp(renderer,'Photosphere') as app:
		# app.run_loop()

		app.init_gl()
		# renderer.compositor.setExplicitTimingMode(
		#							openvr.VRCompositorTimingMode_Explicit_RuntimePerformsPostPresentHandoff)
		frames_displayed=0
		last_print_time=time.time()
		display_gl_time=0
		getposes_time=0

		poses_t=openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount	#!!!
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
					# print 'button pressed'

					result,controller_state=renderer.vr_system.getControllerState(ev.trackedDeviceIndex)
					if result:
						touchpad_button_dir=None
						if controller_state.ulButtonPressed & (1 << openvr.k_EButton_SteamVR_Touchpad):
							touchpad_button_dir=(-1 if controller_state.rAxis[0].x < 0 else +1)

						if controller_state.ulButtonPressed & (1 << openvr.k_EButton_ApplicationMenu):
							print 'Exiting by controller button press'
							renderer.compositor.compositorQuit()
							break

						if controller_state.ulButtonPressed & (1 << openvr.k_EButton_Grip):
							if touchpad_button_dir:
								panorama_actor.center_azimuth_deg+=touchpad_button_dir * 3
								panorama_actor.center_azimuth_deg%=360
								center_azimuth_manual_changes[cur_image_id]=panorama_actor.center_azimuth_deg
								print '   Manual change of center_azimuth_deg to %.0fdeg' % \
																	(panorama_actor.center_azimuth_deg,)
								panorama_actor.update()
								next_image_links_actor.update()
						elif touchpad_button_dir:
							if cmdline_fnames:
								cur_image_id+=touchpad_button_dir
								cur_image_id%=len(cmdline_fnames)
							else:
								sorted_image_ids=sorted(geo_images.geo_images.keys())
								idx=sorted_image_ids.index(cur_image_id)
								cur_image_id=sorted_image_ids[(idx + touchpad_button_dir) % \
																					len(sorted_image_ids)]
							panorama_actor.image,next_image_links_actor.next_image_links, \
											panorama_actor.center_azimuth_deg=go_to_image(cur_image_id)
							panorama_actor.update()
							next_image_links_actor.update()
							# print 'update() call done'
						elif controller_state.ulButtonPressed & (1 << openvr.k_EButton_SteamVR_Trigger):
							next_image_id=next_image_links_actor.select_active_next_image_link()
							if next_image_id is not None:
								cur_image_id=next_image_id
								panorama_actor.image,next_image_links_actor.next_image_links, \
											panorama_actor.center_azimuth_deg=go_to_image(cur_image_id)
								panorama_actor.update()
								next_image_links_actor.update()
								# print 'update() call done'
						else:
							print 'some other button pressed',controller_state.ulButtonPressed
							# print renderer.compositor.forceReconnectProcess()
							# print 'forceReconnectProcess() done'

			frames_displayed+=1
			print_interval_frames=200
			if (frames_displayed % print_interval_frames) == 0 and frames_displayed:
				# print getposes_time
				cur_time=time.time()

				time_passed=cur_time - last_print_time
				print 'Image #%s, %d frames displayed, %.0ffps, display_gl() takes %.0f%% of time' % \
												(cur_image_id,frames_displayed,
												print_interval_frames / float(time_passed),
												panorama_actor.display_gl_time * 100 / float(time_passed))
				# print ' '.join(','.join(map(str,tim)) for tim in render_times_ms)

				last_print_time=cur_time
				panorama_actor.display_gl_time=0
				getposes_time=0
				render_times_ms=[]

		print 'Exiting'

		if center_azimuth_manual_changes:
			print 'Manual center_azimuth_deg changes summary:'
			for id,center_azimuth_deg in center_azimuth_manual_changes.items():
				print 'ID #%s %.0fdeg' % (id,center_azimuth_deg)
