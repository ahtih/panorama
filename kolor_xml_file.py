import os,xml.sax.handler,xml.sax
import panorama

matches=dict()
image_quaternions=dict()

def read_kolor_xml_file(fname,reset_globals=False,set_alternative_fnames=False):
	# Updates globals matches and image_quaternions
	# Returns list of image full filenames in Kolor file order

	global matches,image_quaternions

	if reset_globals:
		matches=dict()
		image_quaternions=dict()

	class kolor_xml_handler(xml.sax.handler.ContentHandler):
		def __init__(self):
			self.image_fnames=[]
			self.cur_image_quaternion=None
			self.cur_image_fname=None

		@staticmethod
		def calc_filename_indexes(img_fnames):
			fname_lists=[img_fnames]
			if set_alternative_fnames:
				fname_lists.append(map(os.path.basename,img_fnames))

			return tuple([tuple(sorted(fnames)) for fnames in fname_lists])

		def startElement(self,name,attrs):
			global matches

			if name == 'image':
				self.cur_image_quaternion=None
			elif name == 'def':
				self.cur_image_fname=attrs.get('filename')
				for fname2 in self.image_fnames:
					for idx in self.calc_filename_indexes((self.cur_image_fname,fname2)):
						matches[idx]=False
				self.image_fnames.append(self.cur_image_fname)
			elif name == 'camera':
				self.cur_image_quaternion=panorama.quaternion_from_kolor_file(
											*[float(attrs.get(name,0)) for name in ('yaw','pitch','roll')])
			elif name == 'match':
				img_fnames=[self.image_fnames[int(attrs.get(attr))] for attr in ('image1','image2')]
				self.match_indexes=self.calc_filename_indexes(img_fnames)
				self.cur_match_points=[]
			elif name == 'point':
				self.cur_match_points.append(tuple(
											[float(attrs.get(name)) for name in ('x1','y1','x2','y2')]))

		def endElement(self,name):
			global image_quaternions

			if name == 'image':
				if self.cur_image_fname is not None:
					fname=self.image_fnames[-1]
					image_quaternions[fname]=self.cur_image_quaternion
					if set_alternative_fnames:
						image_quaternions[os.path.basename(fname)]=self.cur_image_quaternion
					self.cur_image_quaternion=None
					self.cur_image_fname=None
			elif name == 'match':
				for idx in self.match_indexes:
					matches[idx]=tuple(self.cur_match_points)

	handler=kolor_xml_handler()

	parser=xml.sax.make_parser()
	parser.setContentHandler(handler)
	parser.parse(open(fname,'r'))

	return handler.image_fnames
