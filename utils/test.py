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



def longest_common_prefix(strs):
    if not strs:
        return ""
    min_str = min(strs)
    max_str = max(strs)
    for i, step in enumerate(min_str):
        if step != max_str[i]:
            return min_str[:i]
    return min_str

longest_match = ""

for i in range(len(paths)):
    for j in range(i + 1, len(paths)):
        match = longest_common_prefix(paths[i] + paths[j])
        if len(match) > len(longest_match):
            longest_match = match

print("Longest common prefix:", longest_match)
data = {
    'Hierarchy-Layer-1': [11, 0, 0],
    'Hierarchy-Layer-2': [1, 20, 0],
    'Hierarchy-Layer-3': [0, 0, 1],
    'Hierarchy-Layer-4': [0, 2, 1],
    'Hierarchy-Layer-5': [0, 46, 0],
    'Hierarchy-Layer-6': [0, 6, 13],
    'Hierarchy-Layer-7': [0, 7, 3],
    'Hierarchy-Layer-8': [0, 0, 0],
    'Hierarchy-Layer-9': [0, 0, 0]
}

# Transpose the data
transposed_data = list(zip(*data.values()))

# Calculate the sum of each column
column_sums = [sum(column) for column in transposed_data]

# Sort columns by their sums in descending order
sorted_columns = sorted(enumerate(column_sums), key=lambda x: x[1], reverse=True)

# Permutate data values based on sorted column indices
permutated_data = {key: [data[key][index] for index, _ in sorted_columns] for key in data.keys()}

print("Permutated data:",permutated_data)

