
# Requires the notebook for new file creation to be run beforehand!

import pandas as pd

class FastBipartiteGraph(type = "comments and votes"):
	"""This class creates a bipartite graph in which the nodes are users on one hand
	and articles on the other hand. An edge between a user and an article can be defined
	as either a comment from the user under this article, or a vote from the user on a
	comment under this article, or either of these conditions.
	This is specified by the type parameter.


	"""
	if type == 'comments':
		graph = pd.read_csv('./data/')

	if type == 'votes':


	if type == 'comments and votes':