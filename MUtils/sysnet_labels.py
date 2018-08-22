import numpy as np

class Label:
	def __init__(self, filePath):
		self.fil = open(filePath, "r")
		self.tr_map = np.array(self.fil.read().splitlines())
		self.getLabels()
	
	def getLabels(self):
		self.sys_map = []
		self.hmn_map = []
		for i in range(len(self.tr_map)):
			self.sys_map.append(self.tr_map[i].split()[0])
			self.hmn_map.append(self.tr_map[i].split(',')[0].split(" ", 1)[1])
	
	def getLabel(self, index):
		return self.sys_map[index]
	
	def getHumanLabel(self, index):
		return self.hmn_map[index]

	def isTop1(self, index, Label2, pred_list):
		if self.sys_map[index] == Label2.getLabel(pred_list[1000]-1):
			return True
		else:
			return False

	def isTop5(self, index, Label2, pred_list):
		for i in range(5):
			if self.sys_map[index] == Label2.getLabel(pred_list[1000-i]-1):
				return True
		return False
