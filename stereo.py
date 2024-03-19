import numpy as np
# ==============No additional imports allowed ================================#


def get_ncc_descriptors(img, patchsize):
    '''
    Prepare normalized patch vectors for normalized cross
    correlation.

    Input:
        img -- height x width x channels image of type float32
        patchsize -- integer width and height of NCC patch region.
    Output:
        normalized -- height* width *(channels * patchsize**2) array

    For every pixel (i,j) in the image, your code should:
    (1) take a patchsize x patchsize window around the pixel,
    (2) compute and subtract the mean for every channel
    (3) flatten it into a single vector
    (4) normalize the vector by dividing by its L2 norm
    (5) store it in the (i,j)th location in the output

    If the window extends past the image boundary, zero out the descriptor

    If the norm of the vector is <1e-6 before normalizing, zero out the vector.

    '''
    height, width, channels = img.shape
    half_patch = patchsize // 2
    normalized = np.zeros(
        (height, width, channels * patchsize**2), dtype=np.float32)

    for i in range(height):
        for j in range(width):
            # Compute patch boundaries
            top = max(0, i - half_patch)
            bottom = min(height, i + half_patch + 1)
            left = max(0, j - half_patch)
            right = min(width, j + half_patch + 1)

            # Extract patch
            patch = img[top:bottom, left:right, :]

            if patch.shape[0] != patchsize or patch.shape[1] != patchsize:
                normalized[i, j, :] = 0
            else:
                # Subtract mean for every channel
                patch_mean = np.mean(patch, axis=(0, 1))
                patch -= patch_mean

                # Flatten and normalize
                patch_flat = patch.reshape(-1)
                patch_norm = np.linalg.norm(patch_flat)

                if patch_norm < 1e-6:
                    normalized[i, j, :] = 0
                else:
                    normalized[i, j, :] = patch_flat / patch_norm

    return normalized


def compute_ncc_vol(img_right, img_left, patchsize, dmax):
    '''
    Compute the NCC-based cost volume
    Input:
        img_right: the right image, H x W x C
        img_left: the left image, H x W x C
        patchsize: the patchsize for NCC, integer
        dmax: maximum disparity
    Output:
        ncc_vol: A dmax x H x W tensor of scores.

    ncc_vol(d,i,j) should give a score for the (i,j)th pixel for disparity d. 
    This score should be obtained by computing the similarity (dot product)
    between the patch centered at (i,j) in the right image and the patch centered
    at (i, j+d) in the left image.

    Your code should call get_ncc_descriptors to compute the descriptors once.
    '''
    height, width, channels = img_right.shape
    half_patch = patchsize // 2
    ncc_vol = np.zeros((dmax, height, width))

    # Compute descriptors for both images
    descriptors_right = get_ncc_descriptors(img_right, patchsize)
    descriptors_left = get_ncc_descriptors(img_left, patchsize)

    for d in range(dmax):
        for i in range(height):
            for j in range(width-d):
                if j + d < width:
                    # Extract patches centered at (i, j) and (i, j+d)
                    patch_right = descriptors_right[i, j, :]
                    patch_left = descriptors_left[i, j + d, :]

                    # Compute NCC score (dot product)
                    ncc_vol[d, i, j] = np.dot(patch_right, patch_left)

    return ncc_vol


def get_disparity(ncc_vol):
    '''
    Get disparity from the NCC-based cost volume
    Input: 
        ncc_vol: A dmax X H X W tensor of scores
    Output:
        disparity: A H x W array that gives the disparity for each pixel. 

    the chosen disparity for each pixel should be the one with the largest score for that pixel
    '''
    dmax, height, width = ncc_vol.shape
    disparity = np.zeros((height, width), dtype=int)

    # find the disparity with maximum score for each pixel
    for i in range(height):
        for j in range(width):
            max_score_index = np.argmax(ncc_vol[:, i, j])
            disparity[i, j] = max_score_index

    return disparity
