import os,xml.sax.handler,xml.sax
import panorama

matches=dict()
image_quaternions=dict()

def read_kolor_xml_file(fname,reset_globals=False):
	# Updates globals matches and image_quaternions
	# Returns list of image file basenames in Kolor file order

	global matches,image_quaternions

	if reset_globals:
		matches=dict()
		image_quaternions=dict()

	class kolor_xml_handler(xml.sax.handler.ContentHandler):
		def __init__(self):
			self.image_fnames=[]
			self.cur_image_quaternion=None

		@staticmethod
		def calc_filename_indexes(img_fnames):
			return (tuple(sorted(img_fnames)),
					tuple(sorted(map(os.path.basename,img_fnames))))

		def startElement(self,name,attrs):
			global matches

			if name == 'image':
				self.cur_image_quaternion=None
			elif name == 'def':
				fname=attrs.get('filename')
				for fname2 in self.image_fnames:
					for idx in self.calc_filename_indexes((fname,fname2)):
						matches[idx]=False
				self.image_fnames.append(fname)
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
				fname=self.image_fnames[-1]
				image_quaternions[fname]=self.cur_image_quaternion
				image_quaternions[os.path.basename(fname)]=self.cur_image_quaternion
				self.cur_image_quaternion=None
			elif name == 'match':
				for idx in self.match_indexes:
					matches[idx]=tuple(self.cur_match_points)

	handler=kolor_xml_handler()

	parser=xml.sax.make_parser()
	parser.setContentHandler(handler)
	parser.parse(open(fname,'r'))

	return map(os.path.basename,handler.image_fnames)
