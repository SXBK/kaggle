#coding:utf-8
import logging
import sys

class logger:
	def __init__(self):
		self.logger = logging.getLogger("Main")
		formatter = logging.Formatter('%(name)s %(asctime)s %(levelname)-8s %(message)s', '%a, %d %b %Y %H:%M:%S',)  
		file_handler = logging.FileHandler("log/mylog.log")  
		file_handler.setFormatter(formatter)  
		stream_handler = logging.StreamHandler(sys.stderr)  
		self.logger.addHandler(file_handler)  
		self.logger.addHandler(stream_handler)  
		self.logger.setLevel(logging.DEBUG)

if __name__ == '__main__':
	logger = logger()
	logger.logger.error("ok?")
