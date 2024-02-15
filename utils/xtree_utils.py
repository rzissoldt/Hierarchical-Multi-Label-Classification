import json
from collections import namedtuple, deque
from types import SimpleNamespace
import os, re
import time

class xTree:
    def __init__(self, id, pref_label_de=None,pref_label_en=None, node_type=None, wikidata_mapping=None):
        self.id = id
        self.pref_label_de = pref_label_de
        self.pref_label_en = pref_label_en
        self.node_type = node_type
        self.wikidata_mapping = wikidata_mapping
        self.image_count = 0
        self.summed_image_count = 0
        self.nodes = []

        
    def add_node(self, node):
        self.nodes.append(node)

    def __repr__(self):
        return f"xTree({self.id}): {self.nodes}"   
    
    
    def __eq__(self,object):
        if isinstance(object, xTree):
            return self.id == object.id
        return False 
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)

class xTreeEncoder(json.JSONEncoder):
    def default(self, o):
        return o.__dict__

def find_all_paths(root):
    paths = []

    def _dfs(node, path):
        path.append(node.id)
        paths.append(path[:])  # Append a copy of the path
        for child in node.nodes:
            _dfs(child, path[:])  # Start a new path at each child node

    _dfs(root, [])
    return paths
    

def count_leaf_nodes(root):
    """
    Counts the number of leaf nodes in the tree.
    """
    # If it's a leaf node, return 1
    if root.nodes is None or len(root.nodes) == 0:
        return 1
    
    # Recursively count leaf nodes for each child
    return sum(count_leaf_nodes(child) for child in root.nodes)

def filter_tree(root, threshold):
    """
    Recursively removes nodes from the tree if their image_count or summed_image_count is below the specified threshold.
    Returns a new tree structure without modifying the original one.
    """
    # Create a copy of the tree to avoid modifying the original
    tree_copy = SimpleNamespace(**vars(root))

    # Check for leaf nodes and filter based on image_count
    if not hasattr(tree_copy, 'nodes'):
        if getattr(tree_copy, 'image_count', 0) < threshold:
            return None
    else:
        # Check for internal nodes and filter based on summed_image_count
        if getattr(tree_copy, 'summed_image_count', 0) < threshold:
            return None

    # Recursively filter child nodes
    filtered_children = [filter_tree(child, threshold) for child in getattr(tree_copy, 'nodes', [])]

    # Remove None values (nodes that were below the threshold)
    filtered_children = [child for child in filtered_children if child is not None]

    # Update the tree copy with the filtered children
    setattr(tree_copy, 'nodes', filtered_children)

    return tree_copy

def get_id_path(root,search_id, path=None):
    if path is None:
        path = []

    path.append(root.id)

    if root.id == search_id:
        return path

    for child in root.nodes:
        child_path = get_id_path(child, search_id, path.copy())
        if child_path:
            return child_path

    return None

def is_path_in_tree(root, path):
        """
        Check if the given ID path is part of the tree.

        Parameters:
        - path (list): An array of IDs representing the path.

        Returns:
        - bool: True if the path is found in the tree, False otherwise.
        """
        if not path:
            return False  # Empty path is not part of the tree

        current_node = root
        for node_id in path:
            # Check if the current node has a child with the given ID
            node_found = False
            for child_node in current_node.nodes:
                if child_node.id == node_id:
                    current_node = child_node
                    node_found = True
                    break

            if not node_found:
                return False  # ID not found in the current node's children

        return True  # Path is found in the tree
    

def get_leaf_count(node):
    # Base case: if the node is a leaf, return 1
    if not node.nodes:
        return 1
    
    # Recursive case: sum up the leaf counts of all children
    leaf_count = 0
    for child in node.nodes:
        leaf_count += get_leaf_count(child)
    
    return leaf_count

def generate_dicts_per_level(root):
    if not root:
        return []

    dicts_per_level = []
    queue = deque([(root, '')])  # Include an empty string as the initial prefix

    while queue:
        level_dict = {}
        for index in range(len(queue)):
            node, parent_prefix = queue.popleft()
            node_id_with_prefix = f"{parent_prefix}_{node.id}" if parent_prefix else node.id
            level_dict[node_id_with_prefix] = index
            for child in node.nodes:
                queue.append((child, node_id_with_prefix))  # Pass the current node's ID as the prefix for its children

        dicts_per_level.append(level_dict)

    return dicts_per_level[1:]


def get_deepest_path(root):
    def dfs(node, path):
        nonlocal max_path

        if not node:
            return

        path.append(node.id)

        if not node.nodes:
            if len(path) > len(max_path):
                max_path = path.copy()
        else:
            for child in node.nodes:
                dfs(child, path.copy())

    max_path = []
    dfs(root, [])

    return max_path
def id_path_to_string(root,id_path):
    temp_str = ''
    for id in id_path:
        node = search_node(root,id)
        temp_str += node.pref_label_de + ','
        
    return temp_str

def get_id_paths(root, search_id, path=None):
    if path is None:
        path = []

    path.append(root.id)

    result_paths = []

    if root.id == search_id:
        # Append a copy of the current path to the result_paths list
        result_paths.append(path.copy())

    for child in root.nodes:
        # Recursively search for the ID in the child's subtree
        child_paths = get_id_paths(child, search_id, path.copy())
        if child_paths:
            result_paths.extend(child_paths)

    return result_paths

def search_node(root, search_id):
    if not root:
        return None

    # Use a stack for DFS or a queue for BFS
    stack = [root]

    while stack:
        current_node = stack.pop()  # For DFS, use stack.pop(), for BFS, use queue.pop(0)
        if current_node.id == search_id:
            return current_node

        # Add children to the stack or queue
        stack.extend(current_node.nodes)

    # If the target_id is not found
    return None

def search_nodes(root, search_id):
    if not root:
        return []

    result_nodes = []
    stack = [root]

    while stack:
        current_node = stack.pop()
        if current_node.id == search_id:
            result_nodes.append(current_node)

        stack.extend(current_node.nodes)

    return result_nodes

def search_and_increment_image_count(node, search_id):
    if node.id == search_id:
        # Increment image_count attribute if the node is found
        node.image_count += 1
        return True

    # Recursively search in the children of the current node
    for child in node.nodes:
        if search_and_increment_image_count(child, search_id):
            return True

    # Node not found in the current subtree
    return False


def load_xtree_json(file_path):
    with open(file_path,'r') as infile:
        root = json.load(infile, object_hook=lambda d: SimpleNamespace(**d))
        
    return root

def search_item_label(wiki_data_uri):
    pattern = r'Q[0-9]*'
    x = re.search(pattern,wiki_data_uri)
    if x is not None:
        return x.group()
    return None

def save_xtree_json(file_path, root):
    with open(file_path,'w') as outfile:
        json_object = json.dumps(root, cls=xTreeEncoder)
        outfile.write(json_object)

def traverse_and_apply(root,method):
    if root.wikidata_mapping is not None:
        root.image_count = method(root.wikidata_mapping)
        #print(root.image_count)
    for child in root.nodes:
        traverse_and_apply(child,method)
        
    return None

def create_xtree_folder_structure(root,dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)   
    for child in root.nodes:
        create_xtree_folder_structure(child,dir+'/'+child.id)
    return

def eval_summed_image_count(root):
    if not root.nodes:
        return root.image_count

    # Recursively calculate summed_image_count for children
    total_image_count = root.image_count
    for child in root.nodes:
        total_image_count += eval_summed_image_count(child)

    # Update the parent's summed_image_count
    root.summed_image_count = total_image_count
    return total_image_count


def clear_null_image_uri_dicts(root,dir):
    if os.path.isfile(os.path.join(dir,root.id+".json")):
        with open(os.path.join(dir,root.id+".json"), "r") as infile:
            #print(os.path.join(dir,root.id+".json"))
            json_object = json.load(infile)
        if json_object is None:  
            os.remove(os.path.join(dir,root.id+".json"))
    for child in root.nodes:
        clear_null_image_uri_dicts(child,dir+'/'+child.id)
    return

def clear_wikimedia_image_uri_dicts(root,dir):
    if os.path.isfile(os.path.join(dir,root.id+"_wikimedia.json")):
        os.remove(os.path.join(dir,root.id+"_wikimedia.json"))
    for child in root.nodes:
        clear_wikimedia_image_uri_dicts(child,dir+'/'+child.id)
    return

    

def has_one_child_zero_image_count(root):
    for child in root.nodes:
        if child.image_count == 0:
            return True
    return False

def sum_image_count_of_nodes_with_non_zero_nodes(root,dir):         
    if not root:
        return 0
    if not root.nodes:
        file_path = os.path.join(dir,root.id+".json")
        if os.path.isfile(file_path):
            with open(file_path, "r") as infile:
                json_object = json.load(infile)
                if json_object is not None:  
                    image_count = json_object['image_count']
                    root.image_count = image_count
                    return image_count
    if has_one_child_zero_image_count(root):
        return 0
    image_count = 0
    for child in root.nodes:
        image_count += sum_image_count_of_nodes_with_non_zero_nodes(child,dir+'/'+child.id)
    file_path = os.path.join(dir,root.id+".json")
    
    if os.path.isfile(file_path):
        with open(file_path, "r") as infile:
            json_object = json.load(infile)
            if json_object is not None:
                image_count += json_object['image_count']
    root.image_count = image_count           
    return image_count

def sum_image_count_of_nodes(root,dir):
    if not root:
        return 0
    if not root.nodes:
        file_path = os.path.join(dir,root.id+".json")
        if os.path.isfile(file_path):
            with open(file_path, "r") as infile:
                json_object = json.load(infile)
                if json_object is not None:  
                    image_count = json_object['image_count']
                    root.image_count = image_count
                    return image_count

    image_count = 0
    for child in root.nodes:
        image_count += sum_image_count_of_nodes(child,dir)
    file_path = os.path.join(dir,root.id+".json")
    
    if os.path.isfile(file_path):
        with open(file_path, "r") as infile:
            json_object = json.load(infile)
            if json_object is not None:
                image_count += json_object['image_count']
    root.image_count = image_count           
    return image_count

def get_label_with_missing_wikidata_mapping(root,missing_mapping_list):
    
    for child in root.nodes:
        if child.wikidata_mapping is None:
            missing_mapping_list.append({
                'id':child.id,
                'term_de': child.pref_label_de,
                'term_en': child.pref_label_en
            })
        get_label_with_missing_wikidata_mapping(child,missing_mapping_list)

def get_image_counts_at_nth_layer(root, n):
    image_counts = []
    if root is None:
        return image_counts

    def traverse(node, level):
        if level == n:
            image_counts.append((node.id,node.image_count))
        else:
            for child in node.nodes:
                traverse(child, level + 1)

    traverse(root, 1)
    return image_counts
  


def filter_tree_with_threshold(root_file_path,threshold):
    root = load_xtree_json(root_file_path)
    eval_summed_image_count(root)
    
    filtered_root = filter_tree(root, threshold)
    filename_without_extension, _ = os.path.splitext(os.path.basename(root_file_path))
    
    save_xtree_json('./trees/' +filename_without_extension+'_with_image_count_threshold_{0}.json'.format(threshold),filtered_root)
    print(count_leaf_nodes(root))
    print(count_leaf_nodes(filtered_root))
    print(len(get_deepest_path(root)))
    print(len(get_deepest_path(filtered_root)))
    
def filter_tree_with_thresholds(root_file_path,thresholds):
    for threshold in thresholds:
        filter_tree_with_threshold(root_file_path=root_file_path,threshold=threshold)
