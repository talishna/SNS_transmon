import numpy as np

# Step 1: Generate a random 2D array with numbers from 1 to 100
rows, cols = 5, 5  # Define the size of the array
random_array = np.random.randint(1, 101, size=(rows, cols))

print("Original 2D Array:")
print(random_array)

# Step 2: Find numbers divisible by both 3 and 4
divisible_by_3_and_4 = np.vectorize(lambda x: x if x % 3 == 0 and x % 4 == 0 else np.nan)(random_array)

print("\nNumbers divisible by both 3 and 4:")
print(divisible_by_3_and_4)




# Step 1: Generate a 2D array with random numbers from 1 to 100
rows, cols = 5, 5  # Define the shape of the array
array = np.random.randint(1, 101, size=(rows, cols))
print("Original Array:\n", array)


divisible = (array % 3 == 0) & (array % 4 == 0)
result1 = np.where(divisible, array, None)
print("Numbers Divisible by 3 and 4:\n", result1)

arr = np.array([1, 2, 3, 4, 5])
result2 = np.where(arr > 3)
print(result2)
print(arr[result2])

condition = arr < 3
print(condition)
result3 = np.where(condition, arr, None)
print(result3)

print(12//10)


def lengthOfLongestSubstring(s):
    """
    :type s: str
    :rtype: int
    """
    string_set = set()
    left = 0
    max_len = 0
    for right in range(len(s)):
        while s[right] in string_set:
            string_set.remove(s[left])
            left += 1
        string_set.add(s[right])
        max_len = max(max_len, right - left + 1)
    return max_len

print(lengthOfLongestSubstring("abcabcbb"))
print(lengthOfLongestSubstring("bbbbb"))
print(lengthOfLongestSubstring("pwwkew"))
print(lengthOfLongestSubstring("abcdfa"))


def merge(nums1, m, nums2, n):
    """
    :type nums1: List[int]
    :type m: int
    :type nums2: List[int]
    :type n: int
    :rtype: None Do not return anything, modify nums1 in-place instead.
    """
    nums = nums1
    nums1 = nums1[:m]
    nums2 = nums2
    nums1 = nums1[0:m]
    p2 = 0
    p1 = 0
    while p1 < len(nums1) and p2 < len(nums2):
        if nums1[p1] < nums2[p2]:
            nums[p2 + p1] = nums1[p1]
            p1 += 1
        else:  # either the same or nums1 bigger
            nums[p2 + p1] = nums2[p2]
            p2 += 1
            nums[p2 + p1] = nums1[p1]
            p1 += 1
    if p1 < len(nums1):
        nums[(p2 + p1):] = nums1[p1:]
    if p2 < len(nums2):
        nums[(p2 + p1):] = nums2[p2:]
    return nums

def merge(nums1, m, nums2, n):
    """
    :type nums1: List[int]
    :type m: int
    :type nums2: List[int]
    :type n: int
    :rtype: None Do not return anything, modify nums1 in-place instead.
    """
    i, j, k = 0, 0, 0  # indexs of temp, nums2, nums1
    temp = nums1[:m]
    while i < m and j < n:
        if temp[i] < nums2[j]:
            nums1[k] = temp[i]
            i += 1
        else:
            nums1[k] = nums2[j]
            j+=1
        k += 1
    while i < m:
        nums1[k] = temp[i]
        i += 1
        k += 1
    while j < n:
        nums1[k] = nums2[j]
        j += 1
        k += 1

merge(nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3)


def removeElement(nums, val):
    """
    :type nums: List[int]
    :type val: int
    :rtype: int
    """
    n = 0  # num of vals
    l = len(nums)  # length
    for i in range(len(nums)):
        if nums[i-n] == val:
            nums[i-n:(l - 1)] = nums[(i-n + 1):l]
            nums[-1] = None
            n += 1
    return l - n

def removeElement(nums, val):
    """
    :type nums: List[int]
    :type val: int
    :rtype: int
    """
    # Pointer to place the next non-val element
    next_non_val = 0

    # Iterate through the list
    for i in range(len(nums)):
        if nums[i] != val:
            # Place the non-val element at the next_non_val position
            nums[next_non_val] = nums[i]
            next_non_val += 1

    # The length of the array without the val elements
    return next_non_val

print(removeElement([0,1,2,2,3,0,4,2], 2))





matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


print(matrix)

# Slicing a subarray
print(matrix[:2, 1:])  # Output: [[2 3]
                       #          [5 6]]

# Boolean indexing
print(matrix[matrix > 5])  # Output: [6 7 8 9]

print(np.where(matrix > 5, matrix, None))


a = np.array([0, 0, 0])
b = np.array([[1], [2], [3]])
print(a)
print(b)
print(a.shape)
print(b.shape)
result = a + b
print(result)

first_term = a[0]
print(first_term)
first_term += 2
print(a)
c = a.reshape((-1, 1))
print(c)
print(a)

e = np.arange(27).reshape((3,3,3))
mask = e>5
print(mask)
print(e[e>5])
mask2 = np.logical_and(mask, e<10)
print(np.where(mask2,e,0))


random = np.random.randint(0,10,(4,4))

array1 = np.array([[1, 2], [3, 4]])
array2 = np.array([[5, 6]]).T
f = np.concatenate((array1, array2), axis=1)

binary_num = 0b000
print(type(binary_num))

import numpy as np


def flip_bit_from_left(state, position):
    """
    Flips the bit at the specified position from the left in the given state.

    Parameters:
        state (np.ndarray): The current state of the qubits represented as a NumPy array of 0s and 1s.
        position (int): The position of the qubit to flip, counting from the left (0 is the leftmost bit).
        num_bits (int): The total number of bits in the system.

    Returns:
        np.ndarray: The new state after flipping the bit.
    """
    if position < 0 or position >= state.size:
        raise ValueError("Position must be within the range of the number of bits.")

    # Flip the bit at the specified position
    state[position] = 1 - state[position]

    return state


# Example usage
current_state = np.array([0, 1, 0])  # State |010⟩
print(current_state.size)
position_to_flip_from_left = 1  # 0 is the leftmost bit
new_state = flip_bit_from_left(current_state, position_to_flip_from_left)
print(f"New state: {new_state}")


t = np.linspace(0, 1, 1000, endpoint=False)  # Time vector
signal = np.sin(2 * np.pi * 5 * t)

# Compute the FFT of the signal
fft_result = np.fft.fft(signal)
reverse = np.fft.ifft(fft_result)
# Check if the inverse FFT result is equal to the original signal
print(np.allclose(reverse, signal))




# Create a masked array
data = np.array([1, 2, 3, -1, 5])
mask = np.array([False, False, False, True, False])
masked_array = np.ma.array(data, mask=mask)
print(masked_array)


# Create a 2D NumPy array
array = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])
array1 = np.array([1, 2, 3, 4])
result = np.delete(array1, [0, 1])
print(result)
result = np.delete(array, 1, axis=0)
print(result)

# # Delete the second row (index 1)
# result = np.delete(array, 1, axis=0)
# print(result)
#
# # Delete the second and third columns (indices 1 and 2)
# result = np.delete(array, slice(1, 3), axis=1)
# print(result)

def remove_ith(arr, n, m, i):
    mask = np.arange(n, m+1, i)
    return np.delete(arr, mask)

array = np.array([2, 6, 8, 3, 1, 3, 6, 8, 9, 2, 1, 2, 3, 4, 5, 6, 8, 9, 4, 3, 5, 6, 3, 4, 6, 3, 1, 2, 7])
result = remove_ith(array, 0, 10, 2)
print(result)


def one_element_removed(x,y):
    count = 0
    while x[count] == y[count]:
        count += 1
    print(x[count])


x = np.array([1,2,3,4,5])
y = np.array([1,2,4,5])

one_element_removed(x, y)

a = np.array([10, 20, 30])
b = np.arange(0,3,1).T.reshape(3,1)
print(a+b)
