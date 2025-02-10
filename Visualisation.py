import numpy as np
import pyvista as pv

def remap_node_field_for_vis(graph, field):
    """
    graph - nx.Graph
    field - np.array
    assuming field corresponds to graph.nodes ordering
    """
    graph_node_mapping = {k:v for k, v in zip(graph.nodes, range(graph.number_of_nodes()))}
    nodes = np.unique(np.array(graph.edges()).flatten())
    new_field = []
    for node in nodes:
        new_field.append(field[graph_node_mapping[node]])
    return new_field


def generate_visualisation_arrays(coords, edges):
    """coords - coordinate array for glaboal node numbers
    edges, node connections for global node numbers"""
    nodes = np.unique(edges.flatten())
    nodes.sort()
    new_coords = coords[nodes]
    new_map = {p: c for c, p in enumerate(nodes)}
    remapped_connections = []
    for element in edges:
        temp = []
        for node in element:
            temp.append(new_map[node])
        remapped_connections.append(temp)
    remapped_connections = np.array(remapped_connections)
    padding = np.empty(remapped_connections.shape[0], int) * 2
    padding[:] = 2
    connections_with_padding = np.vstack((padding, remapped_connections.T)).T
    return new_coords, connections_with_padding

def visualise_graph_and_field(graph, coords, field, field_name='radii', title='', need_remap = True):
    if title == '':
        title = f'{field_name} visualisation'
    if need_remap:
        field_remap = remap_node_field_for_vis(graph, field)
    else:
        field_remap = field
    vis_coords, vis_connections = generate_visualisation_arrays(coords, np.array(graph.edges()))
    plotter = pv.Plotter()
    plotter.add_title(title)
    pod = pv.PolyData(vis_coords, lines=vis_connections, n_lines=vis_connections.shape[0])
    pod[field_name] = field_remap
    pod_tube = pod.tube(scalars=field_name, radius=1.0, radius_factor = 30)
    plotter.add_mesh(pod_tube, render_lines_as_tubes=True, show_scalar_bar=False)
    plotter.add_scalar_bar(field_name, position_x=0.25)
    plotter.camera_position = 'xz'
    plotter.show()