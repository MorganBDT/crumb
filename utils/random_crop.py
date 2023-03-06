import random
import math
from torch.nn.functional import interpolate


class RandomResizeCrop(object):
    """Randomly crops tensor then resizes uniformly between given bounds.
    Copied from: https://github.com/tyler-hayes/REMIND/blob/master/image_classification_experiments/utils.py
    Args:
        size (sequence): Bounds of desired output sizes.
        scale (sequence): Range of size of the origin size cropped
        ratio (sequence): Range of aspect ratio of the origin aspect ratio cropped
        interpolation (int, optional): Desired interpolation. Default is 'bilinear'
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation='bilinear'):
        #        assert (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def get_params(self, img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (3-d tensor (C,H,W)): Tensor to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = img.size(1) * img.size(2)

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size(1) and h <= img.size(2):
                i = random.randint(0, img.size(2) - h)
                j = random.randint(0, img.size(1) - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size(1) / img.size(2)
        if (in_ratio < min(ratio)):
            w = img.size(1)
            h = int(w / min(ratio))
        elif (in_ratio > max(ratio)):
            h = img.size(2)
            w = int(h * max(ratio))
        else:  # whole image
            w = img.size(1)
            h = img.size(2)
        i = int((img.size(2) - h) // 2)
        j = int((img.size(1) - w) // 2)
        return i, j, h, w

    def __call__(self, img):
        """
        Args:
            img (3-D tensor (C,H,W)): Tensor to be cropped and resized.
        Returns:
            Tensor: Randomly cropped and resized Tensor.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = img[:, i:i + h, j:j + w]  ##crop
        return interpolate(img.unsqueeze(0), self.size, mode=self.interpolation, align_corners=False).squeeze(0)

    def __repr__(self):
        interpolate_str = self.interpolation
        return self.__class__.__name__ + '(size={0}, scale={1}, ratio={2}, interpolation={3})'.format(self.size,
                                                                                                      self.scale,
                                                                                                      self.ratio,
                                                                                                      interpolate_str)
