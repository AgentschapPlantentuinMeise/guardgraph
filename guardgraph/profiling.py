"""Profiling module to get general statistics on the
interactions network

References:
- https://neo4j.com/blog/data-profiling-holistic-view-neo4j/
"""
from functools import cached_property
import pandas as pd
import numpy as np

class Profiler(object):
    def __init__(self, ig):
        """Collects profiling methods

        Args:
          ig: InteractionGraph object
        """
        self.ig = ig

    @cached_property
    def labels(self):
        return pd.DataFrame(
            self.ig.run_query('CALL db.labels()')
        ).label

    @cached_property
    def relationships(self):
        return pd.DataFrame(
            self.ig.run_query('CALL db.relationshipTypes()')
        ).relationshipType

    @cached_property
    def label_counts(self):
        return pd.Series(
            [
                self.ig.run_query(f'MATCH (n:`{l}`) RETURN COUNT(n)')[0][
                    'COUNT(n)'
                ] for l in self.labels
            ], index=self.labels
        )

    @cached_property
    def label_prob(self):
        return self.label_counts/self.label_counts.sum()

    @cached_property
    def relationship_counts(self):
        return pd.Series(
            [
                self.ig.run_query(f'MATCH ()-[r:`{r}`]->() RETURN COUNT(r)')[0][
                    'COUNT(r)'
                ] for r in self.relationships
            ], index=self.relationships
        )

    @cached_property
    def relationship_total(self):
        return self.relationship_counts.sum()
    
    @cached_property
    def relationship_prob(self):
        return self.relationship_counts/self.relationship_total

    @cached_property
    def dyadic_connection_counts(self):
        "DataFrame with total connection codes ~ (:`column_name`)-[r]->(:`row_name`)"
        return pd.DataFrame({
            l1: [
                self.ig.run_query(
                    f'MATCH (n:`{l1}`)-[r]->(m:`{l2}`) RETURN COUNT(r)'
                )[0]['COUNT(r)'] for l2 in self.labels
            ] for l1 in self.labels
        }, index=self.labels)

    def dyadic_connection_prob(self, relationship, norm4labelprob=False, top_labels=None):
        if top_labels:
            labels = self.label_counts.sort_values(
                ascending=False
            ).index[:top_labels]
        else:
            labels = self.label_counts.index
        dyadic_conn = pd.DataFrame({
            l1: [
                self.ig.run_query(
                    f'MATCH (n:`{l1}`)-[r:{relationship}]->(m:`{l2}`) RETURN COUNT(r)'
                )[0]['COUNT(r)'] for l2 in labels
            ] for l1 in labels
        }, index=labels)
        #dyadic_conn_other = self.dyadic_connection_counts - dyadic_conn
        dyadic_conn_node_trends_rel_specific = (
            dyadic_conn/(dyadic_conn.T)
        )
        #dyadic_conn_node_trends = (
        #    self.dyadic_connection_counts/(self.dyadic_connection_counts.T)
        #)
        if norm4labelprob:
            label_counts = self.label_counts[dyadic_conn.index]
            label_prob = label_counts/label_counts.sum()
            norm_prob = np.multiply(
                *np.meshgrid(label_prob,label_prob)
            )
            return dyadic_conn/(
                self.relationship_counts[relationship]*norm_prob
            )
        else:
            return dyadic_conn/self.relationship_counts[relationship]

    def triadic_connection_prob(self, labels, relationship1=None, relationship2=None, rel1_forward=True, rel2_forward=True):
        relationship1 = (
            ('-[r1' if rel1_forward else '<-[r1')+
            (f':{relationship1}' if relationship1 else '')+
            (']->' if rel1_forward else ']-')
        )
        relationship2 = (
            ('-[r2' if rel2_forward else '<-[r2')+
            (f':{relationship2}' if relationship2 else '')+
            (']->' if rel2_forward else ']-')
        )
        triadic_conn = pd.DataFrame({
            (l1,l2): [
                self.ig.run_query(
                    f'MATCH p=(n:`{l1}`){relationship1}(m:`{l2}`){relationship2}(o:`{l3}`) RETURN COUNT(p)'
                )[0]['COUNT(p)'] for l3 in labels
            ] for l1 in labels for l2 in labels
        }, index=labels)
        return triadic_conn

    def degree_analysis(self, relationship, plotfile=None, max_degree=50):
        result = self.ig.run_query(f'MATCH (n)-[:{relationship}]-() WITH DISTINCT n RETURN apoc.node.degree(n, "{relationship}") AS degree')
        result_forward = self.ig.run_query(f'MATCH (n)-[:{relationship}]->() WITH DISTINCT n RETURN apoc.node.degree(n, "{relationship}>") AS degree')
        result_backward = self.ig.run_query(f'MATCH (n)<-[:{relationship}]-() WITH DISTINCT n RETURN apoc.node.degree(n, "<{relationship}") AS degree')
        result_backfor = self.ig.run_query(f'MATCH (n)<-[:{relationship}]-() WITH DISTINCT n RETURN apoc.node.degree(n, "{relationship}>") AS degree')
        degree_spread = pd.DataFrame(result).degree.value_counts().sort_index()
        degree_sprfor = pd.DataFrame(result_forward).degree.value_counts().sort_index()
        degree_sprbac = pd.DataFrame(result_backward).degree.value_counts().sort_index()
        degree_sprbf = pd.DataFrame(result_backfor).degree.value_counts().sort_index()
        if plotfile:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10,4), sharey=True)
            axes[0].bar(
                degree_spread.index[:max_degree], degree_spread[:max_degree]
            )
            axes[1].bar(
                degree_sprfor.index[:max_degree], degree_sprfor[:max_degree]
            )
            axes[2].bar(
                degree_sprbac.index[:max_degree], degree_sprbac[:max_degree]
            )
            axes[3].bar(
                # just top10 to check quality for strictly directional rel
                degree_sprbf.index[:10], degree_sprbf[:10]
            )
            fig.savefig(plotfile)
        return (degree_spread, degree_sprfor, degree_sprbac, degree_sprbf)

"""
Meeting notes with inria

Report globi stats:
- number of species
- number and type of relationships
gbif occurence -> local interaction network
co occurence merged tetrapods

global network --> metaweb for tetrapods

trait database, trait of plants and how they interact
correlations for causal inference

unbias this data -> slide

full network embeddings
dataset of networks

STARTING
MODEL for a

https://www.authorea.com/doi/full/10.22541/au.168010211.10402875
https://www.kaggle.com/competitions/geolifeclef-2023-lifeclef-2023-x-fgvc10/
"""
