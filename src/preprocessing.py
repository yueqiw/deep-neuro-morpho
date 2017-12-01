import numpy as np


def normalize_swc(swc_ori, canvas_shape=(768, 768, 192), shrink=2, structure='dendrites'):
    swc = swc_ori.copy()
    swc[:,2:5] /= shrink
    if structure == 'dendrites':
        swc_subset = swc[np.isin(swc[:,1], [1,3,4]), :]
    else:
        raise ValueError('Unknown structure.')
    max_arr = np.max(swc_subset[:,2:5], axis=0)
    min_arr = np.min(swc_subset[:,2:5], axis=0)
    swc[:,2:5] += - min_arr + (canvas_shape - max_arr + min_arr) / 2
    return swc

def normalize_fit_swc(swc_ori, canvas_shape=(768, 768, 192), dims=3, structure='dendrites'):
    swc = swc_ori.copy()
    if structure == 'dendrites':
        swc_subset = swc[np.isin(swc[:,1], [1,3,4]), :]
    else:
        raise ValueError('Unknown structure.')
    max_arr = np.max(swc_subset[:,2:5], axis=0)
    min_arr = np.min(swc_subset[:,2:5], axis=0)

    ratio = canvas_shape / (max_arr - min_arr)
    ratio = np.min(ratio[:dims]) * 0.9
    swc[:,2:5] *= ratio
    swc[:,2:5] += - min_arr * ratio + (canvas_shape - (max_arr - min_arr) * ratio) / 2
    return swc

def draw_sk_img_vertices(imshape, swc, dims=3, structure='dendrites'):
    if structure == 'dendrites':
        swc_subset = swc[np.isin(swc[:,1], [1,3,4]), :]
    else:
        raise ValueError('Unknown structure.')
    if dims == 3:
        skimg = np.zeros(imshape)
        nodes = swc_subset[:,2:5].astype(int)
        for x in nodes:
            skimg[x[0], x[1], x[2]] = 1
    elif dims == 2:
        skimg = np.zeros(imshape[0:2])
        nodes = swc_subset[:,2:4].astype(int)
        for x in nodes:
            skimg[x[0], x[1]] = 1
    else:
        raise ValueError("dims should be 2 or 3")
    return np.rot90(skimg)


def draw_sk_img(imshape, swc, dims=3, structure='dendrites'):
    if structure == 'dendrites':
        subset = np.isin(swc[:,1], [1,3,4])
        swc_subset = swc[subset, :]  # add 6 - neurites only for retinal bipolar/amacrine cells
    else:
        raise ValueError('Unknown structure.')

    # delete soma edges that are not connected with dendrites
    n_soma = np.sum(swc_subset[:,1] == 1)
    if n_soma == 3 and swc_subset[3, 6] == 1:
        skip_soma = 2
    elif n_soma == 2 and swc_subset[2, 6] == 1:
        skip_soma = 1
    else:
        skip_soma = 0

    skimg = np.zeros(imshape[0:dims])
    nodes = swc[:,2:dims+2].astype(int)
    nodes_subset = nodes[subset, :]
    parents = swc_subset[skip_soma+1:, 6]

    if dims == 2:
        skimg[nodes_subset[0,0], nodes_subset[0,1]] = 1

        for i, x in enumerate(nodes_subset[skip_soma+1:]):
            p = nodes[int(parents[i] - 1), :]
            if adjacent_voxel(p, x, dims=dims):
                skimg[x[0], x[1]] = 1
            else:
                #print('not adjacent:')
                #print(p, x)
                points_on_line = draw_line_bresenham_2d(p, x)
                for point in points_on_line[1:]:
                    skimg[point[0], point[1]] = 1

    elif dims == 3:
        skimg[nodes_subset[0,0], nodes_subset[0,1], nodes_subset[0,2]] = 1

        for i, x in enumerate(nodes_subset[skip_soma+1:]):
            p = nodes[int(parents[i] - 1), :]
            if adjacent_voxel(p, x, dims=dims):
                skimg[x[0], x[1], x[2]] = 1
            else:
                #print('not adjacent:')
                #print(p, x)
                points_on_line = draw_line_bresenham_3d(p, x)
                for point in points_on_line[1:]:
                    skimg[point[0], point[1], point[2]] = 1
    else:
        raise ValueError("dims should be 2 or 3")
    return np.rot90(skimg)


def adjacent_voxel(p, x, dims=3):
    # assums p and x are integer arrays.
    # return true if p and x are adjacent
    diff = np.abs(x - p)[:dims]
    return np.all(diff<=1)


def draw_line_bresenham_2d(start, end):
    # assum int coordinates
    x0, y0 = start[:2]
    x1, y1 = end[:2]
    dx = np.abs(x1-x0)
    dy = -np.abs(y1-y0)
    sx = (x0 < x1) * 2 - 1
    sy = (y0 < y1) * 2 - 1
    err = dx + dy

    points = []
    for i in range(dx - dy + 1):
    #while True:
        points.append((x0, y0))
        if (x0 == x1) and (y0 == y1):
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx  # e_xy+e_x > 0
        if e2 <= dx:
            err += dx
            y0 += sy  # e_xy+e_y < 0
    return points


def draw_line_bresenham_3d(start, end):
    # assum int coordinates
    x0, y0, z0 = start
    x1, y1, z1 = end
    dx = np.abs(x1-x0)
    dy = np.abs(y1-y0)
    dz = np.abs(z1-z0)
    sx = (x0 < x1) * 2 - 1
    sy = (y0 < y1) * 2 - 1
    sz = (z0 < z1) * 2 - 1
    dm = max(dx, dy, dz)  # maximum difference
    i = dm
    x1 = y1 = z1 = dm/2  # error offset

    points = []
    for k in range(dx + dy + dz + 1):
        points.append((x0, y0, z0))
        i -= 1
        if i < 0:
            break

        x1 -= dx
        if x1 < 0:
            x1 += dm
            x0 += sx
        y1 -= dy
        if y1 < 0:
            y1 += dm
            y0 += sy
        z1 -= dz
        if z1 < 0:
            z1 += dm
            z0 += sz
    return points

'''
void plotLine3d(int x0, int y0, int z0, int x1, int y1, int z1)
{
   int dx = abs(x1-x0), sx = x0<x1 ? 1 : -1;
   int dy = abs(y1-y0), sy = y0<y1 ? 1 : -1;
   int dz = abs(z1-z0), sz = z0<z1 ? 1 : -1;
   int dm = max(dx,dy,dz), i = dm; /* maximum difference */
   x1 = y1 = z1 = dm/2; /* error offset */

   for(;;) {  /* loop */
      setPixel(x0,y0,z0);
      if (i-- == 0) break;
      x1 -= dx; if (x1 < 0) { x1 += dm; x0 += sx; }
      y1 -= dy; if (y1 < 0) { y1 += dm; y0 += sy; }
      z1 -= dz; if (z1 < 0) { z1 += dm; z0 += sz; }
   }
}
'''

'''
void plotLine(int x0, int y0, int x1, int y1)
{
   int dx =  abs(x1-x0), sx = x0<x1 ? 1 : -1;
   int dy = -abs(y1-y0), sy = y0<y1 ? 1 : -1;
   int err = dx+dy, e2; /* error value e_xy */

   for(;;){  /* loop */
      setPixel(x0,y0);
      if (x0==x1 && y0==y1) break;
      e2 = 2*err;
      if (e2 >= dy) { err += dy; x0 += sx; } /* e_xy+e_x > 0 */
      if (e2 <= dx) { err += dx; y0 += sy; } /* e_xy+e_y < 0 */
   }
}
'''

def draw_line_bresenham_2(start, end, dims):
    """Bresenham's Line Algorithm
    From http://www.roguebasin.com/index.php?title=Bresenham%27s_Line_Algorithm
    Produces a list of tuples from start and end

    >>> points1 = get_line((0, 0), (3, 4))
    >>> points2 = get_line((3, 4), (0, 0))
    >>> assert(set(points1) == set(points2))
    >>> print points1
    [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
    >>> print points2
    [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
    """
    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1

    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points


def padimg(img, margin):
    # from rivuletpy
    pimg = np.zeros((img.shape[0] + 2 * margin, img.shape[1] + 2 * margin,
                     img.shape[2] + 2 * margin))
    pimg[margin:margin + img.shape[0], margin:margin + img.shape[1], margin:
         margin + img.shape[2]] = img
    return pimg


def unpadimg(img, margin):
    # from rivuletpy
    pimg = np.zeros((img.shape[0] - 2 * margin, img.shape[1] - 2 * margin,
                     img.shape[2] - 2 * margin))
    pimg = img[margin:margin + img.shape[0], margin:margin + img.shape[1],
               margin:margin + img.shape[2]]
    return pimg

def maxip(img_3d):
    return np.max(img_3d, axis=2)

def padswc(swc, margin, dims=3):
    # from rivuletpy
    if dims == 3:
        swc[:, 2:5] = swc[:, 2:5] + margin
    elif dims == 2:
        swc[:, 2:4] = swc[:, 2:4] + margin
    else:
        raise ValueError("dims should be 2 or 3")
    return swc


def convertswc(swc):
    # need to convert from zero-centered to all positive
    pass


def loadswc(filepath):
    '''
    Load swc file as a N X 7 numpy array
    '''
    swc = []
    with open(filepath) as f:
        lines = f.read().split("\n")
        for l in lines:
            if not l.startswith('#'):
                cells = l.strip().split()
                if len(cells) ==7:
                    cells = [float(c) for c in cells]
                    # cells[2:5] = [c-1 for c in cells[2:5]]
                    swc.append(cells)
    return np.array(swc)


def saveswc(filepath, swc):
    if swc.shape[1] > 7:
        swc = swc[:, :7]

    with open(filepath, 'w') as f:
        for i in range(swc.shape[0]):
            print('%d %d %.3f %.3f %.3f %.3f %d' %
                  tuple(swc[i, :].tolist()), file=f)


def loadtiff3d(filepath):
    """Load a tiff file into 3D numpy array"""
    from libtiff import TIFF
    tiff = TIFF.open(filepath, mode='r')
    stack = []
    for sample in tiff.iter_images():
        stack.append(np.rot90(np.fliplr(np.flipud(sample))))
    out = np.dstack(stack)
    tiff.close()

    return out
