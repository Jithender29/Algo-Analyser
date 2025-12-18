from typing import List, Any


def counting_sort(arr: List[Any]) -> List[Any]:
    """
    Sorts a list of non-negative integers using the Counting Sort algorithm.

    Args:
        arr: The list of non-negative integers to be sorted.

    Returns:
        A new list containing the sorted integers.
    """
    if not arr:
        return []

    # 1. Find the maximum element in the input array
    # This determines the range of values we need to count.
    max_val = max(arr)

    # 2. Initialize a count array with zeros.
    # The size of the count array will be (max_val + 1) to accommodate
    # all numbers from 0 to max_val.
    count = [0] * (max_val + 1)

    # 3. Populate the count array.
    # For each element in the input array, increment its corresponding
    # count in the count array.
    for num in arr:
        count[num] += 1

    # 4. Modify the count array to store the actual position of each element
    # in the output array.
    # Each element at index `i` in the count array will now store the
    # sum of counts up to `i`, indicating the last position of `i` in the
    # sorted array.
    for i in range(1, len(count)):
        count[i] += count[i - 1]

    # 5. Create the output array.
    # Initialize an output array of the same size as the input array with zeros.
    output = [0] * len(arr)

    # 6. Build the output array.
    # Iterate through the input array in reverse order to maintain stability
    # (i.e., elements with the same value retain their relative order).
    # For each element `num` in `arr`:
    #   - Decrement `count[num]` to get its correct sorted position.
    #   - Place `num` at that position in the `output` array.
    for num in reversed(arr):
        output[count[num] - 1] = num
        count[num] -= 1

    return output
