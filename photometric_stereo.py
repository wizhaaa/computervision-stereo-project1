import numpy as np
## ======================== No additional imports allowed ====================================##


def photometric_stereo_singlechannel(I, L):
    # L is 3 x k
    # I is k x n
    G = np.linalg.inv(L @ L.T) @ L @ I
    # G is  3 x n
    albedo = np.sqrt(np.sum(G*G, axis=0))

    normals = G/(albedo.reshape((1, -1)) +
                 (albedo == 0).astype(float).reshape((1, -1)))
    return albedo, normals


def photometric_stereo(images, lights):
    '''
        Use photometric stereo to compute albedos and normals
        Input:
            images: A list of N images, each a numpy float array of size H x W x 3
            lights: 3 x N array of lighting directions. 
        Output:
            albedo, normals
            albedo: H x W x 3 array of albedo for each pixel
            normals: H x W x 3 array of normal vectors for each pixel

        Assume light intensity is 1.
        Compute the albedo and normals for red, green and blue channels separately.
        The normals should be approximately the same for all channels, so average the three sets
        and renormalize so that they are unit norm

    '''
    h, w, c = images[0].shape
    red = []
    green = []
    blue = []
    for i in range(len(images)):
        red.append(images[i][:, :, 0].reshape(-1))
        green.append(images[i][:, :, 1].reshape(-1))
        blue.append(images[i][:, :, 2].reshape(-1))
    red = np.array(red)
    green = np.array(green)
    blue = np.array(blue)
    albedo_red, normals_red = photometric_stereo_singlechannel(red, lights)
    albedo_green, normals_green = photometric_stereo_singlechannel(
        green, lights)
    albedo_blue, normals_blue = photometric_stereo_singlechannel(blue, lights)
    albedo = np.stack([albedo_red, albedo_green, albedo_blue],
                      axis=1)
    albedo = albedo.reshape(h, w, 3)
    normals = (normals_red + normals_green + normals_blue)/3
    normals = normals/np.linalg.norm(normals, axis=0)
    normals = normals.T.reshape(h, w, 3)
    return albedo, normals
