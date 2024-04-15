import itertools
paths = [["wk001", "wk333", "wk504"], ["wk002", "wk707"], ["wk001", "wk555"], ["wk001", "wk333", "wk505","wk909"]]

def is_subset(path1, path2):
    # Check if path1 is a subset of path2
    return set(path1).issubset(set(path2))

redundant_paths = []

for i, path1 in enumerate(paths):
    for j, path2 in enumerate(paths):
        if i != j:
            if set(path1) == set(path2):
               break
            if is_subset(path1, path2):
                redundant_paths.append(path1)
                break  # Break after finding the first instance of a subset

# Remove redundant paths
unique_paths = [path for path in paths if path not in redundant_paths]
unique_paths.sort()
unique_paths=list(unique_paths for unique_paths,_ in itertools.groupby(unique_paths))
print("Original paths:", paths)
print("Unique paths:", unique_paths)
