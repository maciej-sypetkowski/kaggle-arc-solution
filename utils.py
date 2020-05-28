import numpy as np
from numba import jit


COLORS = 10


def flatten(l):
    ret = []
    for x in l:
        if isinstance(l, (list, tuple)):
            ret.extend(x)
        else:
            ret.append(x)
    return ret


def reshape_dataframe(df, shape):
    df = [df.iloc[i:i + 1] for i in range(len(df))]
    ret = np.zeros(len(df), dtype=object)
    ret[:] = df
    return ret.reshape(shape)


@jit('void(int32[:,:,:],int8[:,:],int64[:,:],int32,int32,int8,int32,int32)', nopython=True, nogil=True)
def raycast_kernel_single(target, vis, image, dx, dy, col, x, y):
    if vis[x, y]:
        return
    vis[x, y] = 1

    x1 = x + dx
    y1 = y + dy
    if 0 <= x1 < image.shape[0] and 0 <= y1 < image.shape[1] and image[x1, y1] == col:
        raycast_kernel_single(target, vis, image, dx, dy, col, x1, y1)
        target[x, y, 0] = target[x1, y1, 0]
        target[x, y, 1] = target[x1, y1, 1]
    else:
        target[x, y, 0] = x1
        target[x, y, 1] = y1


@jit('void(int32[:,:,:],int8[:,:],int64[:,:],int32,int32,int8)', nopython=True, nogil=True)
def raycast_kernel(target, vis, image, dx, dy, col):
    for ox in range(image.shape[0]):
        for oy in range(image.shape[1]):
            if col == -1:
                tcol = image[ox, oy]
            else:
                tcol = col

            raycast_kernel_single(target, vis, image, dx, dy, tcol, ox, oy)


def raycast(image, delta, col):
    vis = np.zeros_like(image, dtype=np.int8)
    target = np.zeros((*image.shape, 2), dtype=np.int32)
    raycast_kernel(target, vis, image, delta[0], delta[1], col if col is not None else -1)
    return target


@jit('void(int32[:,:], int8[:,:,:], int64[:,:], int8[:,:], int32[:,:])', nopython=True, nogil=True)
def components_kernel(sizes, borders, image, vis, stack):
    col = 0
    # stack[comp, 2 + c] -- counts of color c in component comp
    for a in range(image.shape[0]):
        for b in range(image.shape[1]):
            k = 0
            stack[k, 0] = a
            stack[k, 1] = b
            k += 1

            while k != 0:
                k -= 1
                x = stack[k, 0]
                y = stack[k, 1]

                if vis[x, y]:
                    continue
                vis[x, y] = True
                sizes[x, y] = col

                for i, j in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
                    u = x + i
                    v = y + j
                    if 0 <= u < image.shape[0] and 0 <= v < image.shape[1]:
                        c = image[u, v]
                        if not vis[u, v] and image[x, y] == image[u, v]:
                            stack[k, 0] = u
                            stack[k, 1] = v
                            k += 1
                        if c == image[x, y]:
                            c = -1
                    else:
                        c = 0

                    if c >= 0:
                        stack[col, 2 + c] += 1

            col += 1

    for i in range(col):
        stack[i, 0] = 0
    for a in range(image.shape[0]):
        for b in range(image.shape[1]):
            stack[sizes[a, b], 0] += 1

    for a in range(image.shape[0]):
        for b in range(image.shape[1]):
            borders[a, b] = stack[sizes[a, b], 2:]
            sizes[a, b] = stack[sizes[a, b], 0]


def components(image):
    vis = np.zeros_like(image, dtype=np.int8)
    sizes = np.zeros_like(image, dtype=np.int32)
    borders = np.zeros((*image.shape, COLORS), dtype=np.int8)
    stack = np.zeros((image.shape[0] * image.shape[1], 2 + COLORS), dtype=np.int32)
    components_kernel(sizes, borders, image, vis, stack)
    return sizes, borders


class Offset:

    @jit('void(int32[:,:,:], int32[:,:,:], int32[:,:,:])', nopython=True, nogil=True)
    def compose_offsets_kernel(result, offsets, target):
        for x in range(offsets.shape[0]):
            for y in range(offsets.shape[1]):
                a = offsets[x, y, 0]
                b = offsets[x, y, 1]
                if 0 <= a < offsets.shape[0] and 0 <= b < offsets.shape[1]:
                    result[x, y, 0] = target[a, b, 0]
                    result[x, y, 1] = target[a, b, 1]
                else:
                    result[x, y, 0] = a
                    result[x, y, 1] = b

    @jit('void(int8[:,:], int32[:,:], int32[:,:,:], int64[:,:])', nopython=True, nogil=True)
    def get_cols_dists_kernel(cols, dists, offsets, image):
        for x in range(offsets.shape[0]):
            for y in range(offsets.shape[1]):
                a = offsets[x, y, 0]
                b = offsets[x, y, 1]
                if 0 <= a < offsets.shape[0] and 0 <= b < offsets.shape[1]:
                    cols[x, y] = image[a, b]
                else:
                    cols[x, y] = 0
                dists[x, y] = max(abs(x - a), abs(y - b))

    def compose(offsets, target):
        result = np.zeros_like(offsets)
        Offset.compose_offsets_kernel(result, offsets, target)
        return result

    def identity(xyshape):
        a = np.arange(xyshape[0], dtype=np.int32).reshape(-1, 1, 1).repeat(xyshape[1], 1)
        b = np.arange(xyshape[1], dtype=np.int32).reshape(1, -1, 1).repeat(xyshape[0], 0)
        return np.concatenate([a, b], 2)

    def get_cols_dists(offsets, image):
        cols = np.zeros_like(image, dtype=np.int8)
        dists = np.zeros_like(image, dtype=np.int32)
        Offset.get_cols_dists_kernel(cols, dists, offsets, image)
        return cols, dists
