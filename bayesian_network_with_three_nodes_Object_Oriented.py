class BayesianNetwork:
    def __init__(self):
        self.nodes = {}
    
    def add_node(self, name, parents, prob_table):
        self.nodes[name] = {
            'parents': parents,
            'prob_table': prob_table,
        }
    
    def calculate_probability(self, node_name, values, given={}):
        node = self.nodes[node_name]
        parents = node['parents']
        
        if not parents:
            return node['prob_table']
        
        given_key = tuple(given[parent] for parent in parents)
        probability = node['prob_table'].get(given_key, None)
        
        if probability is None:
            given_key = tuple(None for _ in parents)  # Try without conditioning on parents
            probability = node['prob_table'].get(given_key, None)
            
        if probability is None:
            raise ValueError(f"Insufficient information for node '{node_name}'")
        
        return probability[values]
    
    def infer(self, query, given={}):
        query_node, query_value = query.split("=")
        query_value = int(query_value)
        
        numerator = self.calculate_probability(query_node, query_value, given)
        denominator = 0.0
        
        for value in range(2):
            given[query_node] = value
            probability = self.calculate_probability(query_node, value, given)
            denominator += probability
            
        return numerator / denominator

def main():
    # Create a Bayesian Network
    bn = BayesianNetwork()
    
    # Define nodes with their parents and probability tables
    bn.add_node('A', [], {(): [0.5]})
    bn.add_node('B', ['A'], {(0,): [0.3], (1,): [0.7]})
    bn.add_node('C', ['A'], {(0,): [0.8], (1,): [0.2]})
    
    # Perform inference
    query = 'B=1'
    result = bn.infer(query)
    print(f"P(B=1): {result:.2f}")

if __name__ == "__main__":
    main()
