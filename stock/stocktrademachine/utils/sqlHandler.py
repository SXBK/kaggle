#coding:utf-8
import MySQLdb
import logging

class sqlHandler:
	def exceQuery(self, queryString):
		count=self.cur.execute(queryString)
		return count, self.cur.fetchall()

	def closeConn(self):
		self.cur.close()
		self.conn.close()
	def getCurrentCur(self):
		"""Direct
		get cur of conn, not good.
		"""
		return self.cur
	

class mysqlHandler(sqlHandler):
	def __init__(self, config):
		self.logger=logging.getLogger("Main.SQL")
		self.logger.debug(config)
		try:
			self.conn=MySQLdb.connect(host=config["addr"], user=config["user"], passwd=config["pwd"], db=config["dbname"], port=int(config["port"]), unix_socket=config["us"], charset="utf8")
			self.cur=self.conn.cursor()
		except MySQLdb.Error, e:
			self.logger.error("Mysql Error %d: %s" % (e.args[0], e.args[1]))
			
	def reloadConfig(self,config):
		try:
			self.conn=MySQLdb.connect(host=config["addr"], user=config["user"], passwd=config["pwd"], db=config["dbname"], port=int(config["port"]), unix_socket=config["us"], charset="utf8")
			self.cur=self.conn.cursor()
		except MySQLdb.Error, e:
			self.logger.error("Mysql Error %d: %s" % (e.args[0], e.args[1]))

	def commitConn(self):
		try:
			self.conn.commit()
		except MySQLdb.Error, e:
			self.logger.error("Mysql Error %d: %s" % (e.args[0], e.args[1]))
