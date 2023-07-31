from golem.core.dag.verification_rules import has_no_cycle, has_no_self_cycled_nodes
from golem.core.dag.convert import graph_structure_as_nx_graph


class Rule():

    def has_no_duplicates(self, graph):
        _, labels = graph_structure_as_nx_graph(graph)
        if len(labels.values()) != len(set(labels.values())):
            raise ValueError('Custom graph has duplicates')
        return True    

    def bn_rules(self):
        return [has_no_self_cycled_nodes, has_no_cycle, self.has_no_duplicates]