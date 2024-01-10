
# Requires the notebook for new file creation to be run beforehand!

import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite

class FastBipartiteGraph:
	"""This class creates a bipartite graph in which the nodes are users on one hand
	and articles on the other hand. An edge between a user and an article can be defined
	as either a comment from the user under this article, or a vote from the user on a
	comment under this article, or either of these conditions.
	This is specified by the type parameter.

	"""
	def __init__(self, type = "comments and votes",
				 format = 'dataframe',
				 user_info = False,
				 article_info = False,
				 posting_info = False,
				 vote_info = False):
		self.graph = pd.read_csv('./data/User_article_graph.csv')
		# A pandas dataframe with columns:
		# ID_CommunityIdentity | ID_Posting | ID_Article | link

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

		if format == 'dataframe': # keep the graph as a dataframe
			pass

		elif format == 'graph': # transform the dataframe to a networkx bipartite graph
			graph = nx.Graph()
			graph.add_nodes_from(self.graph['ID_CommunityIdentity'].unique(), bipartite=0)
			graph.add_nodes_from(self.graph['ID_Article'].unique(), bipartite=1)
			edges = list(zip(self.graph['ID_CommunityIdentity'], self.graph['ID_Article']))
			graph.add_edges_from(edges)
			# TODO: add node info and edge info to the graph
			self.graph = graph # replace the dataframe with the graph

		else:
			raise ValueError("argument format must be either 'dataframe' or 'graph'")

class FastArticleGraph:
	"""This class creates a projection of the user-article bipartite graph onto the set
	of articles, resulting in a graph where the nodes are all articles, and one article
	is linked to another if they have links with a common user in the bipartite graph."""

	def __init__(self, type = "comments and votes",
				 format = 'dataframe',
				 user_info = False,
				 article_info = False,
				 posting_info = False,
				 vote_info = False):

		self.bipartite_graph = FastBipartiteGraph(type = type,
										format = format,
										user_info=user_info,
										article_info=article_info,
										posting_info=posting_info,
										vote_info=vote_info).graph

		# Projection of the bipartite graph onto the set of articles
		if format == 'dataframe':
			pass # TODO: create the projection of the bipartite graph onto the set of articles

		elif format == 'graph':
			articles = {n for n, d in self.bipartite_graph.nodes(data=True) if d['bipartite']==1}
			self.graph = bipartite.projected_graph(self.bipartite_graph, articles)

