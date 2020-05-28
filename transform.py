import random
from functools import partial
from itertools import chain, permutations, product, starmap

import numpy as np

from utils import flatten, COLORS


class TException(Exception):
    """
    Transform exception. A transformer can throw this exception to indicate
    that it doesn't want or doesn't know how to transform a given task.
    """

    pass


class TBase:
    """
    The base class of task transformers. A transformer transforms a task by
    transforming or augmenting input images and corresponding output images.
    It also performs inverse transformation of the output prediction.
    """

    class SampleTransformer:
        """
        The base class of sample transformers. A sample transformer transforms
        one sample (one input image possibly with an output image) of the task.
        """

        def __init__(self, task_transformer, in_image):
            self.task_transformer = task_transformer
            self.in_image = in_image
            self.transformed_images = self.transform()

        def transform(self):
            """
            Transforms or augments `self.in_image`. Returns a list of images.
            """

            raise NotImplementedError()

        def transform_output(self, out_image):
            """
            Transforms or augments `out_images`. Returns a list of images.
            The length of that list must equal the length of the list returned
            in `transform` function.
            """

            raise NotImplementedError()

        def transform_output_inverse(self, prediction):
            """
            Performs aggregation of image predictions and transforms it back.
            `prediction` is a list of the same length as returned in `transform`
            and `transform_output` functions. Returns one image.
            """

            raise NotImplementedError()

    def __init__(self, in_images, out_images):
        """
        The length of `in_images` must be greater or equal the length of `out_images`.
        Input images that don't have corresponding output images are test images,
        and are used for making predictions.
        """

        assert len(in_images) >= len(out_images)

        self.in_images = in_images
        self.out_images = out_images

    def get_transformer(self, in_image):
        """
        Return SampleTransformer for one image
        """

        return self.SampleTransformer(self, in_image)

    def unsequence(self):
        return [self]


class TSequential(TBase):
    """
    The transformer class that executes sequentially the list of transformers.
    """

    class SampleTransformer(TBase.SampleTransformer):
        def transform(self):
            self.tr1 = self.task_transformer.tr1.get_transformer(self.in_image)
            ins = self.tr1.transformed_images

            self.tr2s = []  # [(transformer, images_count)]
            ret = []
            for i in ins:
                t = self.task_transformer.tr2.get_transformer(i)
                ret.append(t.transformed_images)
                self.tr2s.append((t, len(ret[-1])))
            return flatten(ret)

        def transform_output(self, out_image):
            outs = self.tr1.transform_output(out_image)
            assert len(self.tr2s) == len(outs)
            ret = []
            for (t, c), o in zip(self.tr2s, outs):
                ret.append(t.transform_output(o))
                assert c == len(ret[-1])
            return flatten(ret)

        def transform_output_inverse(self, prediction):
            k = 0
            outs = []
            for t, c in self.tr2s:
                outs.append(t.transform_output_inverse(prediction[k: k + c]))
                k += c
            return self.tr1.transform_output_inverse(outs)

    def __init__(self, tr1, tr2, *a, **k):
        super().__init__(*a, **k)

        self.tr1 = tr1(self.in_images, self.out_images)

        transformers = [self.tr1.get_transformer(i) for i in self.in_images]
        ins = flatten([t.transformed_images for t in transformers])
        outs = flatten([t.transform_output(o)
                        for t, o in zip(transformers, self.out_images)])
        self.tr2 = tr2(ins, outs)

    def unsequence(self):
        return [*self.tr1.unsequence(), *self.tr2.unsequence()]

    @classmethod
    def make(cls, trs, *a, **k):
        if len(trs) == 0:
            return TIdentity
        if len(trs) == 1:
            return trs[0]
        if len(trs) == 2:
            return cls(trs[0], trs[1], *a, **k)
        return cls(trs[0], partial(cls.make, trs[1:]), *a, **k)


class TCBase(TBase):
    """
    The base class for color transformers.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.constant_colors = sorted(list(set(range(COLORS))
            - set(np.unique(np.concatenate([i.reshape(-1) for i in self.in_images])))))

    class SampleTransformer(TBase.SampleTransformer):
        def __init__(self, task_transformer, in_image, color_mapping, out_color_mapping=None):
            self.color_mapping = color_mapping
            self.out_color_mapping = out_color_mapping if out_color_mapping is not None else color_mapping
            super().__init__(task_transformer, in_image)

        def inversed_out_mapping(self):
            assert len(set(self.out_color_mapping)) == len(
                self.out_color_mapping)
            assert set(self.out_color_mapping) == set(range(COLORS))
            ret = np.zeros_like(self.out_color_mapping)
            ret[self.out_color_mapping] = np.arange(
                len(self.out_color_mapping))
            return ret

        def transform(self):
            return [self.color_mapping[self.in_image]]

        def transform_output(self, out_image):
            return [self.out_color_mapping[out_image]]

        def transform_output_inverse(self, prediction):
            return self.inversed_out_mapping()[prediction[0]]

    def get_transformer(self, in_image):
        raise NotImplementedError()


class TCCount(TCBase):
    """
    A color transformer that colors each image independenly by the number
    of color occurences, i.e. after coloring, the color 0 is the frequent color
    in the image, the color 1 is the second frequent color in the image, etc.
    """

    def get_transformer(self, in_image):
        mapping = np.arange(COLORS)
        colors, counts = np.unique(np.concatenate(
            [in_image.reshape(-1), np.arange(COLORS)]), return_counts=True)
        pos = [((in_image == c).nonzero() + (np.array([c, 0]),))[0].tolist()
               for c in range(COLORS)]
        i = 0
        for _, _, col in sorted(zip(counts, pos, colors), reverse=True):
            if col in self.constant_colors:
                mapping[col] = COLORS - 1 - self.constant_colors.index(col)
            else:
                mapping[col] = i
                i += 1
        return self.SampleTransformer(self, in_image, mapping)


class TCCountBg(TCBase):
    """
    A color transformer that works in the same way as `TCCount` but the color 0 is
    reserved for the global color 0.
    """

    def get_transformer(self, in_image):
        mapping = np.arange(COLORS)
        colors, counts = np.unique(np.concatenate(
            [in_image.reshape(-1), np.arange(COLORS), np.array([0] * 10000)]), return_counts=True)
        pos = [((in_image == c).nonzero() + (np.array([c, 0]),))[0].tolist()
               for c in range(COLORS)]
        i = 0
        for _, _, col in sorted(zip(counts, pos, colors), reverse=True):
            if col in self.constant_colors:
                mapping[col] = COLORS - 1 - self.constant_colors.index(col)
            else:
                mapping[col] = i
                i += 1
        return self.SampleTransformer(self, in_image, mapping)


class TCPos(TCBase):
    """
    A color transformer that colors images accordingly to the order of occurrences. 
    """

    def get_transformer(self, in_image):
        mapping = {}
        for col in range(COLORS):
            if col in self.constant_colors:
                mapping[col] = COLORS - 1 - self.constant_colors.index(col)
        k = 0
        for x, y in product(*map(range, in_image.shape)):
            c = in_image[x, y]
            if c not in mapping:
                mapping[c] = k
                k += 1
        for c in range(COLORS):
            if c not in mapping:
                mapping[c] = k
                k += 1
        mapping = np.array([mapping[i] for i in range(COLORS)])

        return self.SampleTransformer(self, in_image, mapping)


class TIdentity(TBase):
    class SampleTransformer(TBase.SampleTransformer):
        def transform(self):
            return [self.in_image]

        def transform_output(self, out_image):
            return [out_image]

        def transform_output_inverse(self, prediction):
            return prediction[0]


class TSeparationRemover(TBase):
    """
    A transformer that removes vertical and horizontal separations / frames
    from the image. Every image has to have the same number of separations.
    """

    @staticmethod
    def calc_delta_start_count(x, y, borders, shape, extend_nonexisting=False):
        if any(starmap(lambda s, k: (s - k + 1 - 2 * borders) % k != 0, zip(shape, [x, y]))):
            return None, None, None
        delta = tuple(starmap(lambda s, k: (s + 1 - 2 * borders) // k, zip(shape, [x, y])))
        if borders:
            start = (0, 0)
        else:
            if extend_nonexisting:
                start = (-1, -1)
            else:
                start = (delta[0] - 1, delta[1] - 1)
        if extend_nonexisting:
            count = (x + 1, y + 1)
        else:
            count = (x - 1 + 2 * borders, y - 1 + 2 * borders)
        return delta, start, count

    def __init__(self, *a, **k):
        super().__init__(*a, **k)

        number_of_match = 0

        matches = []  # [(x_parts, y_parts, with_borders)]
        for x in range(1, min(map(lambda x: x.shape[0], self.in_images))):
            for y in range(1, min(map(lambda x: x.shape[1], self.in_images))):
                for borders in [False, True]:
                    if (x, y) == (1, 1):
                        continue

                    for img in self.in_images:
                        delta, start, count = self.calc_delta_start_count(
                            x, y, borders, img.shape)
                        if delta is None:
                            break

                        arr = []
                        for d in [0, 1]:
                            for a in range(count[d]):
                                z = start[d] + a * delta[d]
                                if d == 0:
                                    r = img[z, :]
                                else:
                                    r = img[:, z]
                                arr = np.unique(np.concatenate([arr, np.unique(r)]))
                                if len(arr) > 1:
                                    break
                            if len(arr) > 1:
                                break
                        if len(arr) != 1:
                            break
                    else:
                        matches.append((x, y, borders))

        matches = sorted(matches, key=lambda x: (-x[0] - x[1], not x[2], *x))
        if len(matches) <= number_of_match:
            raise TException()

        self.match = matches[number_of_match]

    class SampleTransformer(TBase.SampleTransformer):
        def transform(self):
            x, y, borders = self.task_transformer.match
            delta, start, count = self.task_transformer.calc_delta_start_count(
                x, y, borders, self.in_image.shape, extend_nonexisting=True)
            ret = []
            for a in range(count[0]):
                range1 = slice(start[0] + a * delta[0] + 1,
                               start[0] + (a + 1) * delta[0])
                r = []
                for b in range(count[1]):
                    range2 = slice(start[1] + b * delta[1] + 1,
                                   start[1] + (b + 1) * delta[1])
                    r.append(self.in_image[range1, range2])

                ret.append(np.concatenate(r, 1))
            ret = np.concatenate(ret)
            return [ret]

        def transform_output(self, out_image):
            return [out_image]

        def transform_output_inverse(self, prediction):
            return prediction[0]


class TTransposeInput(TBase):
    class SampleTransformer(TBase.SampleTransformer):
        def transform(self):
            return [self.in_image.copy().T]

        def transform_output(self, out_image):
            return [out_image]

        def transform_output_inverse(self, prediction):
            return prediction[0]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if all([i.shape[0] == i.shape[1] for i in self.in_images]):
            raise TException()


def _get_flips(img):
    return [img[:, :], img[::-1, :], img[:, ::-1], img[::-1, ::-1]]


class TAugFlips(TBase):
    """
    A transformer that augments images by adding their flips.
    """

    class SampleTransformer(TBase.SampleTransformer):
        def transform(self):
            return _get_flips(self.in_image)

        def transform_output(self, out_image):
            return _get_flips(out_image)

        def transform_output_inverse(self, prediction):
            return prediction[0]


class TAugFlipsWRotation(TBase):
    class SampleTransformer(TBase.SampleTransformer):
        def transform(self):
            return _get_flips(self.in_image) + _get_flips(self.in_image.transpose(1, 0))

        def transform_output(self, out_image):
            return _get_flips(out_image) + _get_flips(out_image.transpose(1, 0))

        def transform_output_inverse(self, prediction):
            return prediction[0]


class TAugColor(TBase):
    """
    A transformer that augments images by recoloring them using random color
    permutations, except for the color 0 which is not changed.
    """

    class SampleTransformer(TBase.SampleTransformer):

        def transform(self):
            return [m[self.in_image] for m in self.task_transformer.mappings]

        def transform_output(self, out_image):
            return [m[out_image] for m in self.task_transformer.mappings]

        def transform_output_inverse(self, prediction):
            return self.task_transformer.inv_mappings[0][prediction[0]]

    def inverted_mapping(self, mapping):
        ret = np.zeros_like(mapping)
        ret[mapping] = np.arange(len(mapping))
        return ret

    def __init__(self, in_images, out_images):
        self.in_images = in_images
        self.out_images = out_images

        all_input_colors = set(chain.from_iterable(map(np.unique, in_images)))
        all_output_colors = set(chain.from_iterable(map(np.unique, out_images)))
        self.all_colors = set(range(COLORS))

        self.permutable_colors = sorted(all_input_colors.difference({0}))

        self.mappings = []
        self.inv_mappings = []

        perms = list(permutations(self.permutable_colors))
        for perm in random.sample(perms, min(12, len(perms))):
            mapping = np.arange(COLORS)
            mapping[self.permutable_colors] = perm
            self.mappings.append(mapping)
            self.inv_mappings.append(self.inverted_mapping(mapping))

        if len(self.mappings) <= 1:
            raise TException()
