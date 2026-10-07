import numpy as np

# Domain blocks
# Optimization! Instead cutting to 8x8 Domain blocks and compression
# Algorithm compresses img to 128x128 and cuts to. 4x4 Domain blocks
def create_Domain_blocks(img):
    h, w = img.shape
    img_small = img[:h // 2 * 2, :w // 2 * 2].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    return convert_to_block(img_small)

# In PIFS Algorithm iterate all Domain blocks and:
# 1. Domain_block_size / 2 
# 2. Isometry : 1 of 8
# 3. Affine : s_i * x + o_i ; R_i = s*D_j + o
def get_isometries(block):
    isometries = []
    for k in range(4):
        isometries.append(np.rot90(block, k))

    flipped = np.fliplr(block)

    for k in range(4):
        isometries.append(np.rot90(flipped, k))

    return np.array(isometries)

# function to convert a photo in range-blocks (R) array
def convert_to_block(img):
    R_size = 4
    h, w = img.shape
    n_h = h // R_size
    n_w = w // R_size

    blocks = img.reshape(n_h, R_size, n_w, R_size)
    blocks = blocks.transpose(0, 2, 1, 3)
    blocks = blocks.reshape(-1, R_size, R_size)
    
    return blocks

def get_transforms(block):
    transforms = []
    for k in range(4):
        rot = np.rot90(block, k)
        transforms.append(rot)
        transforms.append(np.fliplr(rot))
    return transforms

