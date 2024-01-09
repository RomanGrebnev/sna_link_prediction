
# Requires the notebook for new file creation to be run beforehand!

import pandas as pd

class FastBipartiteGraph:
	"""This class creates a bipartite graph in which the nodes are users on one hand
	and articles on the other hand. An edge between a user and an article can be defined
	as either a comment from the user under this article, or a vote from the user on a
	comment under this article, or either of these conditions.
	This is specified by the type parameter.

	"""
	def __init__(self, type = "comments and votes",
				 user_info = False,
				 article_info = False,
				 posting_info = False,
				 vote_info = False):
		self.graph = pd.read_csv('./data/User_article_graph.csv')

		if type == 'comments':  # keep only the links from postings
			self.graph = self.graph[self.graph['link'] == 'posting']

		elif type == 'votes':  # keep only the links from votes
			self.graph = self.graph[self.graph['link'] == 'vote']

		elif type == 'comments and votes':  # keep the graph as it is
			pass

		else:
			raise ValueError("argument type must be either 'comments', 'votes' or 'comments and votes'")

		if user_info:
			self.users_info = pd.read_csv('./data/Users_info.csv')

		if article_info:
			self.articles_info = pd.read_csv('./data/Articles_info.csv')

		if posting_info:
			self.postings_info = pd.read_csv('./data/Postings_info.csv')
			# TODO: aggregate posting info to have one row  of variables per edge in the graph

		if vote_info:
			self.votes_info = pd.read_csv('./data/Votes_info.csv')
			# TODO: aggregate vote info to have one row  of variables per edge in the graph


class FastArticleGraph:
	"""This class creates a projection of the user-article bipartite graph onto the set
	of articles, resulting in a graph where the nodes are all articles, and one article
	is linked to another if they have links with a common user in the bipartite graph."""

	def __init__(self, type = "comments and votes",
				 user_info = False,
				 article_info = False,
				 posting_info = False,
				 vote_info = False):
		self.graph = FastBipartiteGraph(type = type,
										user_info=user_info,
										article_info=article_info,
										posting_info=posting_info,
										vote_info=vote_info).graph

		# Projection of the bipartite graph onto the set of articles


