import facebook
import logging
import sys
import time


from os import listdir,remove,rename
from os.path import isfile,join,abspath,dirname,exists
from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer

logging.basicConfig(level=logging.ERROR)
# app access token
access_token ='EAAaZAORXHG3gBACotLEZCJY7qPsh3kz2PXr50Ya70weik2rbu4ALMVVYgLZCt8lfGXMPZAZCF2KNK8F3NwTLsaZBTP6tmSUJPRPVemPy3wKCfhioF6GcGEsENy0dZAkh1QMK0x8fNuywT1f44fnJXoa'
graph = facebook.GraphAPI(access_token=access_token)

folder_path = sys.argv[1]

class MyEventHandler(PatternMatchingEventHandler):
	"""docstring for MyEventHandler"""
	patterns = ["*.png"] 																							#image format
	def __init__(self,observer):
		super(MyEventHandler, self).__init__()
		self.observer = observer
		self.imgFiles = []

	def on_created(self, event):
		if not event.is_directory:
			print "created"
			self.post_image(event)
			# self.post_to_facebook(event)
			# self.imgFiles.append(event.src_path)
			

	def on_deleted(self, event):
		print event.src_path, " deleted"
		# self.imgFiles

	def on_moved(self,event):
		print event.src_path," moved"

	def post_image(self,event):
		pages = graph.get_object('me/accounts')
		for page in pages['data']:
			if page['id'] == '1730678097162289' : 																   #bgp page id
				graph_page = facebook.GraphAPI(page['access_token'])
				graph_page.put_photo(image=open(event.src_path, 'rb'), album_path="1874827756080655" + "/photos")  #album photobooth id
				print "image uploaded"
				remove(event.src_path);
				print "image deleted"
	
def main(argv=None):
	path = argv[0]

	observer = Observer()
	event_handler = MyEventHandler(observer)

	observer.schedule(event_handler, path, recursive=False)
	observer.start()
	try:
		while True:
			time.sleep(1)
	except KeyboardInterrupt:
		observer.stop()
	
	observer.join()

	return 0

if __name__ == "__main__":
	main([folder_path])	


