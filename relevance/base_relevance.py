from abc import ABC,abstractmethod


class BaseRelevance(ABC):
	@abstractmethod
	def compute_relevance(self,conversation:str,judgment:str)->float:
		pass