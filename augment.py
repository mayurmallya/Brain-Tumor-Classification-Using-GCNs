import numpy as np
import random, itertools

def generate_permutation_keys():
    """
    This function returns a set of "keys" that represent the 48 unique rotations &
    reflections of a 3D matrix.
    Each item of the set is a tuple:
    ((rotate_y, rotate_z), flip_x, flip_y, flip_z, transpose)
    As an example, ((0, 1), 0, 1, 0, 1) represents a permutation in which the data is
    rotated 90 degrees around the z-axis, then reversed on the y-axis, and then
    transposed.
    48 unique rotations & reflections:
    https://en.wikipedia.org/wiki/Octahedral_symmetry#The_isometries_of_the_cube
    """
    return set(itertools.product(
        itertools.combinations_with_replacement(range(4), 2), range(2), range(2), range(2), range(2)))

def random_permutation_key():
    """
    Generates and randomly selects a permutation key. See the documentation for the
    "generate_permutation_keys" function.
    """
    return random.choice(list(generate_permutation_keys()))

def permute_data(data_in, key):
    """
    Permutes the given data according to the specification of the given key. Input data
    must be of shape (n_modalities, z, y, x).
    Input key is a tuple: ((rotate_y, rotate_z), flip_x, flip_y, flip_z, transpose)
    As an example, ((0, 1), 0, 1, 0, 1) represents a permutation in which the data is
    rotated 90 degrees around the z-axis, then reversed on the y-axis, and then
    transposed.
    """
    data = np.copy(data_in)
    # (rotate_y, rotate_z), flip_x, flip_y, flip_z, transpose = key
    (__, rotate_z), flip_x, flip_y, flip_z, transpose = key

    # if rotate_y != 0:
    #     data = np.rot90(data, rotate_y, axes=(1, 3))
    if rotate_z != 0:
        data = np.rot90(data, rotate_z, axes=(2, 3))
    if flip_z:
        data = data[:, ::-1]
    if flip_y:
        data = data[:, :, ::-1]
    if flip_x:
        data = data[:, :, :, ::-1]
    if transpose:
        data = np.transpose(data, (0,1,3,2)) # transpose x, y
    return data.copy()

def random_permutation_x_y(x_data):
    """
    Performs random permutation on the data.
    :param x_data: numpy array containing the data. Data must be of shape (n_modalities, x, y, z).
    :return: the permuted data
    """
    key = random_permutation_key()
    return permute_data(x_data, key)

def augment_data(x_data, augment=None):
    return random_permutation_x_y(x_data)
 
