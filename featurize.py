from collections import defaultdict
from functools import partial, reduce
from itertools import product

import cv2
import numpy as np
import pandas as pd

from utils import COLORS, raycast, Offset, components, reshape_dataframe


class FException(Exception):
    """
    Featurizer exception. A featurizer can throw this exception to indicate
    that it doesn't want or doesn't know how to featurize a given task.
    """

    pass


class FBase:
    """
    The base class of task featurizers. A featurizer featurizes a task by
    extracting features from input images, and getting labels (pixel-level) from output images.
    It also performs assembling of the output image based on predictions.
    """

    class SampleFeaturizer:
        """
        The base class of sample featurizer. A sample featurizer featurizes
        one sample of the task.
        """

        def __init__(self, task_featurizer, in_image):
            self.task_featurizer = task_featurizer
            self.in_image = in_image

        def calc_features(self):
            self.features = self.featurize()

        def featurize(self):
            """
            Extracts features from `self.in_image`. Returns a data frame.
            """

            raise NotImplementedError()

        def get_labels(self, out_image):
            """
            Extracts labels from `out_image`. Return a 1D np.array of integers.
            The length of this np.array must equal the length of the dataframe
            returned in `featurize` function. $i$-th element of the array
            corresponds to $i$-th element of the data frame.
            """

            raise NotImplementedError()

        def assemble(self, prediction):
            """
            Assembles predictions into the output image. `prediction` is
            a 1D np.array of integers that correspond to the data frame
            returned in `featurize` function.
            """

            raise NotImplementedError()

    def __init__(self, in_images, out_images):
        self.in_images = in_images
        self.out_images = out_images

    def get_featurizer(self, in_image):
        return self.SampleFeaturizer(self, in_image)

    def get_features_visualization(self):
        return []


class FGlobal(FBase):
    """
    Experimental featurizer, that wraps another featurizer,
    and appends image-level features to each dataframe row.
    Here, global features are based on a simple object detection algorithm.
    """

    class SampleFeaturizer(FBase.SampleFeaturizer):
        def __init__(self, wrapped_sample_featurizer, featurizer, in_image, in_image_gfeatures):
            super().__init__(featurizer, in_image)
            self.wrapped_sample_featurizer = wrapped_sample_featurizer
            self.featurizer = featurizer
            self.in_image = in_image
            self.in_image_gfeatures = in_image_gfeatures

        def featurize(self):
            df = self.wrapped_sample_featurizer.featurize()
            assert len(self.in_image_gfeatures) > 0
            global_features = np.array(list(chain.from_iterable(f.reshape(-1)
                    for f in self.in_image_gfeatures)))
            assert len(global_features) > 1
            names = [f'global_feat{i}' for i in range(len(global_features))]
            for name, col in zip(names, global_features):
                df[name] = [col] * len(df)
            return df

        def get_labels(self, out_image):
            return self.wrapped_sample_featurizer.get_labels(out_image)

        def assemble(self, prediction):
            return self.wrapped_sample_featurizer.assemble(prediction)

    def _calc_global_features(self):
        """
        Some heuristic -- object detection.
        """
        bboxes_per_sample = []
        for img in self.in_images:
            bboxes = defaultdict(list)
            for col in range(10):
                num_component, component = cv2.connectedComponents((img == col).astype(np.uint8))
                for c in range(1, num_component):
                    p = (component == c).astype(np.uint8)
                    if p.sum():
                        bbox = cv2.boundingRect(p)
                        bboxes[bbox[2:]].append(bbox)
            bboxes_per_sample.append(bboxes)

        all_keys = sorted(set(chain.from_iterable(b.keys() for b in bboxes_per_sample)))
        for w, h in all_keys:
            crops = []
            if (w, h) == (1, 1):
                continue
            if all(len(b[(w, h)]) == 1 for b in bboxes_per_sample):
                bboxes = []
                for in_img, bbox in zip(self.in_images, bboxes_per_sample):
                    bbox = bbox[(w, h)]
                    assert len(bbox) == 1
                    if (h, w) == in_img.shape:
                        break
                    bboxes.append(bbox[0])
                else:
                    for in_img, bbox in zip(self.in_images, bboxes):
                        x, y, _, _ = bbox
                        crops.append(in_img[y:y + h, x:x + w])

                    # skip detection if there are only zero-entropy images
                    if all(len(np.unique(img)) == 1 for img in crops):
                        continue

                    self.in_images_gfeatures.append(crops)

    def __init__(self, wrapped_featurizer, in_images, out_images):
        self.in_images = in_images
        self.out_images = out_images
        self.wrapped_featurizer = wrapped_featurizer(in_images, out_images)

        # 2D array of images (crops): n_detections (most often 0 or 1) x len(in_images)
        self.in_images_gfeatures = []

        self._calc_global_features()
        if not self.in_images_gfeatures:
            raise FException()

    def get_featurizer(self, in_image):
        wrapped_sample_featurizer = self.wrapped_featurizer.get_featurizer(in_image)
        in_image_gfeatures = []

        # gather features corresponding to the given in_image
        for one_detection_images in self.in_images_gfeatures:
            assert len(self.in_images) == len(one_detection_images)
            for img, gfeatures in zip(self.in_images, one_detection_images):
                if img is in_image:
                    in_image_gfeatures.append(gfeatures)

        assert in_image_gfeatures
        return self.SampleFeaturizer(wrapped_sample_featurizer, self, in_image, in_image_gfeatures)

    def get_features_visualization(self):
        return self.in_images_gfeatures


class FConstant(FBase):
    """
    A featurizer for tasks in which input size equals output size.
    """

    class SampleFeaturizer(FBase.SampleFeaturizer):
        def featurize(self):
            features = []  # list of np.array[nfeatures, xsize, ysize] or np.array[xsize, ysize]
            feature_names = []  # list of strings

            deltas = (-1, 0, 1)
            neigh = np.zeros((len(deltas) ** 2, *self.in_image.shape), dtype=np.int8)
            for ox, oy in product(*map(range, self.in_image.shape)):
                # absolute neighbourhood color
                for k, (i, j) in enumerate(product(*[deltas] * 2)):
                    x = (ox + i)  # % self.in_image.shape[0]
                    y = (oy + j)  # % self.in_image.shape[1]
                    if 0 <= x < self.in_image.shape[0] and 0 <= y < self.in_image.shape[1]:
                        neigh[k, ox, oy] = self.in_image[x, y]
                    else:
                        neigh[k, ox, oy] = 0
                    k += 1
            features.append(neigh)
            feature_names.extend(['Neigh{}'.format(i) for i in product(*[deltas] * 2)])

            # unique = np.zeros((3, *self.in_image.shape), dtype=np.int8) # 3 because row, column, neighborhood
            # unique_names = []
            # for ox, oy in product(*map(range, self.in_image.shape)):
            #     # absolute neighbourhood color
            #     unique[0, ox, oy] = len(np.unique(self.in_image[:, oy]))
            #     unique[1, ox, oy] = len(np.unique(self.in_image[ox, :]))
            #     unique[2, ox, oy] = len(np.unique(self.in_image[max(ox - 1, 0) : ox + 2,
            #                                                     max(oy - 1, 0) : oy + 2]))
            # features.append(unique)
            # feature_names.extend(['UniqueInRow', 'UniqueInCol', 'UniqueInNeigh'])

            if self.task_featurizer is None or self.task_featurizer.use_rot_90:
                rotations = (False, True)
            else:
                rotations = (False,)

            for use_rot in rotations:
                sym = np.zeros((4, *self.in_image.shape), dtype=np.int8)
                for ox, oy in product(*map(range, self.in_image.shape)):
                    for k, (i, j) in enumerate(product(*([[0, 1]] * 2))):
                        x, y = ox, oy
                        if i:
                            x = self.in_image.shape[0] - x - 1
                        if j:
                            y = self.in_image.shape[1] - y - 1
                        if use_rot:
                            if y >= self.in_image.shape[0] or x >= self.in_image.shape[1]:
                                sym[k, ox, oy] = -1
                            else:
                                sym[k, ox, oy] = self.in_image[y, x]
                        else:
                            sym[k, ox, oy] = self.in_image[x, y]
                features.append(sym)
                feature_names.extend(f'SymRot90_{i}{j}' if use_rot else f'Sym_{i}{j}'
                                     for i, j in product(*([[0, 1]] * 2)))

            deltas = [[1, 0], [-1, 0], [0, 1], [0, -1],
                      [1, 1], [-1, -1], [-1, 1], [1, -1]]
            for di, d in enumerate(deltas):
                col_opts = [0, None]
                targets = [raycast(self.in_image, d, col) for col in col_opts]
                for t1, col in zip(targets, col_opts):
                    cols, dists = Offset.get_cols_dists(t1, self.in_image)
                    features.extend([dists % 2, dists % 3])
                    dists = dists.astype(np.float32)
                    features.extend([cols, dists])
                    ray_type_label = 'Same' if col is None else 'Notbg'
                    feature_names.extend([f'RayCol{ray_type_label}_{d}_mod2',
                                          f'RayCol{ray_type_label}_{d}_mod3'])
                    feature_names.extend([f'RayCol{ray_type_label}_{d}',
                                          f'RayDist{ray_type_label}_{d}'])
                    for t2, col2 in zip(targets, col_opts):
                        off = Offset.compose(t1, t2)

                        cols2, dists2 = Offset.get_cols_dists(off, self.in_image)
                        dists2 = dists2.astype(np.float32)
                        features.extend([cols2, dists2 - dists])
                        ray_type_label2 = 'Same' if col2 is None else 'Notbg'
                        feature_names.extend([f'RayCol{ray_type_label}_{ray_type_label2}_{d}',
                                              f'RayDist{ray_type_label}_{ray_type_label2}_{d}'])

            sizes, borders = components(self.in_image)
            sizes = sizes.astype(np.float32)
            border_exclusive = ((borders == borders.sum(2, keepdims=True)) * np.arange(1, COLORS + 1)).sum(2) - 1
            features.extend([sizes, border_exclusive])
            feature_names.extend(['ComponentSize', 'ComponentBorderExclusive'])

            features = np.concatenate(list(map(
                lambda x: (x if len(x.shape) == 3 else x.reshape((-1, *x.shape))).astype(object), features)))
            features = features.reshape((features.shape[0], -1)).transpose([1, 0])
            return pd.DataFrame(features, columns=feature_names)

        def get_labels(self, out_image):
            if self.in_image.shape != out_image.shape:
                raise FException()
            return out_image.reshape((-1,))

        def assemble(self, prediction):
            return prediction.reshape(self.in_image.shape)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_rot_90 = all(i.shape[0] == i.shape[1] for i in self.in_images)
        if not all(i.shape == o.shape for i, o in zip(self.in_images, self.out_images)):
            raise FException()


class FFactorBase(FBase):
    """
    The base class of featurizers for tasks where an input size is a multiplication
    of an output size (the same multiplication factor for every pair input / output image).
    """

    @staticmethod
    def calculate_factor(i, o):
        round_num = 6
        return round(float(o[0] / i[0]), round_num), round(float(o[1] / i[1]), round_num)

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        factors = [self.calculate_factor(i.shape, o.shape)
                   for i, o in zip(self.in_images, self.out_images)]
        if len(set(factors)) != 1:
            raise FException()
        self.factor = factors[0]
        self.factor = tuple(map(lambda x: round(x) if x == round(x) else x, self.factor))

    class SampleFeaturizer(FBase.SampleFeaturizer):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.factor = self.task_featurizer.factor
            self.out_size = tuple((round(float(f * s))
                    for f, s in zip(self.task_featurizer.factor, self.in_image.shape)))
            if (self.task_featurizer.calculate_factor(self.in_image.shape, self.out_size)
                    != self.task_featurizer.factor):
                raise FException()

        def get_labels(self, out_image):
            return out_image.reshape((-1,))

        def assemble(self, prediction):
            return prediction.reshape(self.out_size)


class FFactorUp(FFactorBase):
    def __init__(self, coord_func, *a, **k):
        self.coord_func = coord_func
        super().__init__(*a, **k)

    class SampleFeaturizer(FFactorBase.SampleFeaturizer):
        def featurize(self):
            if (not (self.factor == tuple(map(round, self.factor))
                    and (self.factor[0] >= 1 and self.factor[1] >= 1)
                    and (self.factor[0] > 1 or self.factor[1] > 1))):
                raise FException()

            single_featurizer = FConstant.SampleFeaturizer(None, self.in_image)
            single_featurizer.calc_features()
            feats = reshape_dataframe(single_featurizer.features, self.in_image.shape)

            features = np.zeros(self.out_size, dtype=object)
            for x, y in product(range(self.in_image.shape[0]), range(self.in_image.shape[1])):
                for i, j in product(range(self.factor[0]), range(self.factor[1])):
                    f = feats[x, y].copy()
                    f['CoordX'] = [i]
                    f['CoordY'] = [j]
                    for a, b in product(*[[0, 1]] * 2):
                        f[f'SymC_{a}{b}'] = f[f'Sym_{(i % 2) ^ a}{b}']
                        f[f'Sym_{a}C{b}'] = f[f'Sym_{a}{(j % 2) ^ b}']
                        f[f'SymC_{a}C{b}'] = f[f'Sym_{(i % 2) ^ a}{(j % 2) ^ b}']
                    features[self.task_featurizer.coord_func(x, y, i, j, self)] = f

            features = pd.concat(features.reshape(-1).tolist(), ignore_index=True)
            return features


def tile_coord(x, y, i, j, self):
    return x + self.in_image.shape[0] * i, y + self.in_image.shape[1] * j


def scale_coord(x, y, i, j, self):
    return x * self.factor[0] + i, y * self.factor[1] + j


FTile = partial(FFactorUp, tile_coord)
FScale = partial(FFactorUp, scale_coord)


class FFactorDown1(FFactorBase):
    class SampleFeaturizer(FFactorBase.SampleFeaturizer):
        def featurize(self):
            down_factor = (round(1 / self.factor[0], 3), round(1 / self.factor[1], 3))
            down_factor = tuple(map(lambda x: round(x) if x == round(x) else x, down_factor))

            if (not (down_factor == tuple(map(round, down_factor))
                    and (self.factor[0] <= 1 and self.factor[1] <= 1)
                    and (self.factor[0] < 1 or self.factor[1] < 1))):
                raise FException()

            features = []
            for x, y in product(*map(range, down_factor)):
                fragment = self.in_image[x * self.out_size[0]: (x + 1) * self.out_size[0],
                                         y * self.out_size[1]: (y + 1) * self.out_size[1]]
                single_featurizer = FConstant.SampleFeaturizer(None, fragment)
                single_featurizer.calc_features()
                f = single_featurizer.features.copy()
                f.columns = list(map(lambda col: 'x{}-y{}-{}'.format(x, y, col), f.columns))
                features.append(f)

            features = reduce(pd.DataFrame.join, features)
            return features


class FFactorDown2(FFactorBase):
    class SampleFeaturizer(FFactorBase.SampleFeaturizer):
        def featurize(self):
            down_factor = (round(1 / self.factor[0], 3), round(1 / self.factor[1], 3))
            down_factor = tuple(map(lambda x: round(x) if x == round(x) else x, down_factor))

            if (not (down_factor == tuple(map(round, down_factor))
                    and (self.factor[0] <= 1 and self.factor[1] <= 1)
                    and (self.factor[0] < 1 or self.factor[1] < 1))):
                raise FException()

            features = np.zeros(self.out_size, dtype=object)
            for x, y in product(*map(range, self.out_size)):
                fragment = self.in_image[x * down_factor[0]: (x + 1) * down_factor[0],
                                         y * down_factor[1]: (y + 1) * down_factor[1]]
                single_featurizer = FConstant.SampleFeaturizer(None, fragment)
                single_featurizer.calc_features()
                f = single_featurizer.features
                f = reshape_dataframe(f, -1)
                pf = []
                for k, r in enumerate(f):
                    r = r.copy().reset_index()
                    r.columns = list(map(lambda col: 'k{}-{}'.format(k, col), r.columns))
                    pf.append(r)

                f = reduce(pd.DataFrame.join, pf)
                features[x, y] = f

            features = pd.concat(features.reshape(-1).tolist(), ignore_index=True)
            return features


class FSquare(FBase):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)

        for i, o in zip(self.in_images, self.out_images):
            if i.shape[0] ** 2 != o.shape[0] or i.shape[1] ** 2 != o.shape[1]:
                raise FException()

    class SampleFeaturizer(FFactorBase.SampleFeaturizer):
        def __init__(self, *a, **k):
            FBase.SampleFeaturizer.__init__(self, *a, **k)
            self.out_size = (self.in_image.shape[0] ** 2, self.in_image.shape[1] ** 2)

        def featurize(self):
            single_featurizer = FConstant.SampleFeaturizer(None, self.in_image)
            single_featurizer.calc_features()

            features = np.zeros(self.out_size, dtype=object)
            for x, y in product(*map(range, self.in_image.shape)):
                f = single_featurizer.features

                i = x * self.in_image.shape[1] + y
                p = single_featurizer.features.iloc[i:i + 1]
                p.columns = list(map(lambda x: 'square-{}'.format(x), p.columns))
                f = f.join(pd.concat([p] * len(f), ignore_index=True))

                f = reshape_dataframe(f, self.in_image.shape)
                features[x * self.in_image.shape[0]: (x + 1) * self.in_image.shape[0],
                         y * self.in_image.shape[1]: (y + 1) * self.in_image.shape[1]] = f

            features = pd.concat(features.reshape(-1).tolist(), ignore_index=True)
            return features
