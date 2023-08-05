class Node:
    def __init__(self, name, parents, cpt):
        self.name = name
        self.parents = parents
        self.cpt = cpt

    def probability(self, state, parent_states):
        if not self.parents:
            return self.cpt

        parent_values = tuple(parent_states[parent] for parent in self.parents)
        return self.cpt[parent_values][state]


class BayesianNetwork:
    def __init__(self, nodes):
        self.nodes = nodes

    def get_node_by_name(self, name):
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def infer(self, query, evidence):
        query_node = self.get_node_by_name(query)
        evidence_nodes = {node.name: state for node, state in evidence.items()}

        hidden_nodes = [node for node in self.nodes if node.name != query and node.name not in evidence_nodes]

        results = {}
        for query_state in [True, False]:
            extended_evidence = evidence_nodes.copy()
            extended_evidence[query] = query_state

            probability_true = 1.0
            probability_false = 1.0

            for node in hidden_nodes:
                probability_true *= node.probability(True, extended_evidence)
                probability_false *= node.probability(False, extended_evidence)

            total_probability = probability_true + probability_false
            results[query_state] = probability_true / total_probability

        return results


# Example usage:
if __name__ == "__main__":
    # Define the conditional probability tables (CPTs) for each node
    cpt_weather = {
        (): {True: 0.7, False: 0.3},
    }
    cpt_rain = {
        (True,): {True: 0.8, False: 0.2},
        (False,): {True: 0.2, False: 0.8},
    }
    cpt_umbrella = {
        (True,): {True: 0.9, False: 0.1},
        (False,): {True: 0.2, False: 0.8},
    }

    # Create the nodes
    weather_node = Node("Weather", [], cpt_weather)
    rain_node = Node("Rain", ["Weather"], cpt_rain)
    umbrella_node = Node("Umbrella", ["Rain"], cpt_umbrella)

    # Create the Bayesian Network
    bn = BayesianNetwork([weather_node, rain_node, umbrella_node])

    # Perform inference
    query_result = bn.infer("Rain", {"Umbrella": True, "Weather": True})
    print("Probability of Rain given Umbrella=True and Weather=True:")
    print("Rain=True:", query_result[True])
    print("Rain=False:", query_result[False])
