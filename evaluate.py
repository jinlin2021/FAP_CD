import pickle
import numpy as np
import pandas as pd
import os


def diversity(sample_node):
    grid_diversity= {}
    for grid_id, atom_sample in sample_node.items():
        classes = np.argmax(atom_sample, axis=-1)
      
        unique_classes = np.unique(classes)
        unique_classes = unique_classes[unique_classes != 6]
        diversity = len(unique_classes) / 13
        
        grid_diversity[grid_id] = diversity


    average_diversity = np.mean(list(grid_diversity.values()))

    return average_diversity, grid_diversity



# elderly_efficiency

def calculate_elderly_efficiency(node_sample, edge_sample, house_node_index, travel_time_threshold=15):
    elderly_indices = [3, 4]  
    grid_efficiency = {}
    n =0
    for grid_id, nodes in node_sample.items():
        edges = np.squeeze(edge_sample[grid_id]) 
        house_indices = house_node_index[grid_id]  
        nodes[:,6] = -1e9
        nodes = np.argmax(nodes, axis=-1)
        elderly_mask = np.isin(nodes[:], elderly_indices)
        elderly_nodes_indices = np.where(elderly_mask)[0]
    
        adjacency_to_houses = edges[elderly_nodes_indices, :house_indices]

        covered_by_any_elderly = np.any(adjacency_to_houses == 1, axis=1)
        num_covered = np.sum(covered_by_any_elderly)

        total_elderly = elderly_nodes_indices.size

        coverage_ratio = num_covered / total_elderly if total_elderly > 0 else 0
        grid_efficiency[grid_id] = coverage_ratio
        n+=1


    average_efficiency = np.sum(list(grid_efficiency.values())) /n

    return average_efficiency, grid_efficiency


def calculate_facility_efficiency(node_sample, edge_sample, house_node_index):
    grid_efficiency = {}
    facility_indices = [13, 9, 7, 5]
   
    for grid_id, nodes in node_sample.items():
        edges = np.squeeze(edge_sample[grid_id])  
        house_indices = house_node_index[grid_id]  

        efficiency_scores = [] 

        for indices in facility_indices:
            nodes[:,6] = -1e9
            nodes_type = np.argmax(nodes, axis=-1)
            facility_mask = np.isin(nodes_type, indices)
            facility_nodes_indices = np.where(facility_mask)[0]
            if not facility_nodes_indices.size or not house_indices:
                efficiency_scores.append(0)
                continue
            adjacency_to_houses = edges[facility_nodes_indices, :house_indices]
            covered_by_any_facility = np.any(adjacency_to_houses == 1, axis=1)
            num_covered = np.sum(covered_by_any_facility)
            total_facility = facility_nodes_indices.size
            coverage_ratio = num_covered / total_facility if total_facility > 0 else 0
            efficiency_scores.append(coverage_ratio)
            
        grid_efficiency[grid_id] = np.mean(efficiency_scores)
    average_efficiency = np.mean(list(grid_efficiency.values()))

    return average_efficiency, grid_efficiency

# calculate_accessability
def calculate_accessability(node_sample, edge_sample, house_node_index, population_data):
    
    facility_indices = [i for i in range(14) if i != 6]
    grid_efficiency = {}
    grid_access = {}
    for grid_id, nodes in node_sample.items():
        edges = np.squeeze(edge_sample[grid_id]) 
        num_houses = house_node_index[grid_id]  
        population = population_data[grid_id] 
        residential_indices = np.arange(num_houses)
        nodes[:,6] = -1e9
        nodes_type = np.argmax(nodes, axis=-1)
        not_residential_mask = ~np.isin(np.arange(len(nodes_type)), residential_indices)
        facility_mask = np.isin(nodes_type, facility_indices) & not_residential_mask
        facility_nodes_indices = np.where(facility_mask)[0]
        adjacency_to_houses = edges[facility_nodes_indices, :][:, residential_indices]
        covered_by_any_facility = np.any(adjacency_to_houses == 1, axis=1)
        coverage_count = np.sum(covered_by_any_facility)
        if population > 0:
            coverage_ratio_p = np.sum(coverage_count) / population
            coverage_ratio = np.sum(coverage_count) / (edges.shape[0]-num_houses)
        else:
            coverage_ratio = 0
        grid_efficiency[grid_id] = coverage_ratio_p
        grid_access[grid_id] = coverage_ratio
        
    average_accessability_p = np.mean(list(grid_efficiency.values()))
    average_accessability = np.mean(list(grid_access.values()))
    return average_accessability_p, average_accessability, grid_efficiency, grid_access

# calculate_gini_coefficient based on the accessability, elderly_efficiency, facility_efficiency
def calculate_gini_coefficient(node_sample, edge_sample, house_node_index, population_data):
    _, _, accessibilities,grid_access = calculate_accessability(node_sample, edge_sample, house_node_index, population_data)
    average_efficiency, elderly_efficiency = calculate_elderly_efficiency(sample_node, sample_edge, house_node_indices)
    
    average_efficiency, facility_efficiency = calculate_facility_efficiency(node_sample, edge_sample, house_node_index)
    df = pd.DataFrame([accessibilities, elderly_efficiency, facility_efficiency])
    values = np.array(list(df.mean()))
    sorted_values = np.sort(values)
    n = len(sorted_values)
    cumulative_values_sum = np.cumsum(sorted_values)
    sum_values = cumulative_values_sum[-1]
    gini_index = (n + 1 - 2 * np.sum(cumulative_values_sum) / sum_values) / n
    
    return gini_index



def metrics(sample_node, sample_edge, house_node_indices, population_data):
    Diversity = diversity(sample_node)
    Life_Service = calculate_facility_efficiency(sample_node, sample_edge, house_node_indices)
    Elderly_Care = calculate_elderly_efficiency(sample_node, sample_edge, house_node_indices)
    Accessibility =  calculate_accessability(sample_node, sample_edge, house_node_indices, population_data)
    Gini = calculate_gini_coefficient(sample_node, sample_edge, house_node_indices, population_data)
    return Diversity, Life_Service, Elderly_Care, Accessibility, Gini


if  '__main__':

    data_dir = "./data/"

    house_node_indices = pickle.load(open(os.path.join(data_dir, 'house_counts.pkl'), 'rb')) 
    population_data = pickle.load(open(os.path.join(data_dir, 'population.pkl'), 'rb'))

    sample_dir = "..."
    sample_node = pickle.load(open(os.path.join(sample_dir, 'sample_node.pkl'),'rb'))
    sample_edge = pickle.load(open(os.path.join(sample_dir, 'sample_edge.pkl'), 'rb'))


    for key, value in sample_edge.items():
        transformed = np.where(value > 0, 1, 0)
        sample_edge[key] = transformed


    Diversity, Life_Service, Elderly_Care, Accessibility, Gini = metrics(sample_node, sample_edge, house_node_indices, population_data)
    print("Diversity:", Diversity)
    print("Life_Service:", Life_Service)
    print("Elderly_Care:", Elderly_Care)
    print("Accessibility:", Accessibility)
    print("Gini:", Gini)
