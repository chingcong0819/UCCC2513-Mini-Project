#!/usr/bin/env python
# coding: utf-8

# ## Exercises
# 1. Find the range of values for each of the following data types:
#     * `uint8`
#     * `int8`
#     * `uint32`
#     * `int32`
# 

# In[1]:


import numpy as np

data_types = [np.uint8, np.int8, np.uint32, np.int32]
ranges = []

for dtype in data_types:
    info = np.iinfo(dtype)
    ranges.append(f"{dtype.__name__}: {info.min} to {info.max}")

print("Ranges of data types:")
for range in ranges:
    print(range)


# 2. Try to write a simple custom function to determine whether a given integer is odd or even number.

# In[2]:


def is_odd(num):
    """
    Checks if a number is odd.

    Args:
        num (int): The number to check.

    Returns:
        bool: True if the number is odd, False otherwise.
    """

    return num % 2 != 0

number = 17
if is_odd(number):
    print(f"{number} is odd.")
else:
    print(f"{number} is even.")


# 3. Write a simple example code to show that Numpy is more efficient in numerical computation of large arrays of data than equivalent Python list.

# In[4]:


import numpy as np
import time

# Define the size of the arrays/lists
size = 10**6

# Create two large Python lists
list1 = list(range(size))
list2 = list(range(size))

# Create two large NumPy arrays
array1 = np.arange(size)
array2 = np.arange(size)

# Measure the time taken for element-wise addition using Python lists
start_time = time.time()
result_list = [x + y for x, y in zip(list1, list2)]
end_time = time.time()
python_time = end_time - start_time

print(f"Time taken using Python lists: {python_time:.5f} seconds")

# Measure the time taken for element-wise addition using NumPy arrays
start_time = time.time()
result_array = array1 + array2
end_time = time.time()
numpy_time = end_time - start_time

print(f"Time taken using NumPy arrays: {numpy_time:.5f} seconds")


# 4. Run the following codes:
# ```python
#     # create a 1D array
#     my_arr = np.arange(10)
#     print("Initial my_arr: ", my_arr)
#     arr_slice = my_arr
#     print("Initial arr_slice: ", arr_slice)
# 
#     # change the first element of arr_slice
#     arr_slice[0] = 55
# 
#     print("my_arr: ", my_arr)
#     print("arr_slice: ", arr_slice)
# ```
# 
#     What do you notice? Propose a way to reassign `arr_slice` with new value       **without modifying** `my_arr`.
# 

# In[6]:


my_arr = np.arange(10)
print("Initial my_arr: ", my_arr)
arr_slice = my_arr
print("Initial arr_slice: ", arr_slice)

# change the first element of arr_slice
arr_slice[0] = 88

print("my_arr: ", my_arr)
print("arr_slice: ", arr_slice)


# 5. Create an image as shown as the following with the help of Numpy and matplotlib modules. You can arbitrarily set the dimension of the image and white circular spot at the middle.
# 

# In[11]:


import numpy as np
import matplotlib.pyplot as plt

# Define image dimensions
image_size = 256  # Adjust this as needed

# Define spot radius (adjust this as needed)
spot_radius = 45

# Create a zero-filled NumPy array for the image
image = np.zeros((image_size, image_size))

# Calculate distance from center
center_x = image_size // 2
center_y = center_x

# Create mesh grids for x and y coordinates
y, x = np.meshgrid(np.arange(image_size), np.arange(image_size))
distance = np.sqrt(((x - center_x) ** 2) + ((y - center_y) ** 2))

# Set spot values to white (255) within the spot radius
image[distance <= spot_radius] = 255

# Plot the image
plt.imshow(image, cmap='gray')  
plt.show()


# In[ ]:




