from typing import List, Any


def bucket_sort(arr: List[Any]) -> List[Any]:
    if not arr:
        return []

    max_val = max(arr)
    min_val = min(arr)
    
    # If all elements are the same, return the array as is.
    if max_val == min_val:
        return arr

    num_buckets = len(arr)
    if num_buckets == 0: 
        return []

    # Calculate the size of each bucket.
    if isinstance(arr[0], float):
        bucket_range = (max_val - min_val + 1e-9) / num_buckets
    else:
        bucket_range = (max_val - min_val + 1) / num_buckets
    
    if bucket_range == 0: 
        bucket_range = 1

    # Create empty buckets
    buckets = [[] for _ in range(num_buckets)]

    # Distribute elements into buckets
    for num in arr:
        index = int((num - min_val) / bucket_range)
        
        if index >= num_buckets:
            index = num_buckets - 1
        
        buckets[index].append(num)

    # Sort each bucket and concatenate the results
    sorted_arr = []
    for bucket in buckets:
        # Sort each bucket individually. For this, we can use another sorting algorithm.
        # Python's Timsort is efficient.
        bucket.sort()
        sorted_arr.extend(bucket)

    return sorted_arr
