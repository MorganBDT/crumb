"""CRUMBR (Compositional Replay Using Memory Blocks Reconstructor)
Contains the CrumbReconstructor class, which contains the functionality needed to use a codebook matrix to
reconstruct batches of "feature banks" (feature level representations of images or other inputs).
Functions for initializing codebook matrices using various initialization strategies are also included.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional
from torch.nn.parameter import Parameter


def init_mem_standard_normal(n_memblocks, memblock_length):
    """Initialize codebook matrix of size n_memblocks x memblock_length, with all values drawn from standard normal dist
    """
    return torch.randn(n_memblocks, memblock_length)


def init_mem_normal_MobileNet(n_memblocks, memblock_length):
    """Initialize codebook matrix of size n_memblocks x memblock_length, with all values drawn from normal dist.
    Multiply by 0.5 to match the standard deviation of MobileNet feature maps.
    """
    return torch.randn(n_memblocks, memblock_length)*0.5


def init_mem_rand_uniform(n_memblocks, memblock_length):
    """Initialize codebook matrix of size n_memblocks x memblock_length, with all values drawn from a [0, 1] uniform dist
    """
    return torch.rand(n_memblocks, memblock_length)


def init_mem_rand_distmatch_dense(n_memblocks, memblock_length, pos_val_std=14.0):
    """ Dense matrix designed to match the distribution of NON-ZERO VALUES in natural feature maps """
    return torch.abs(torch.randn(n_memblocks, memblock_length) * (pos_val_std * (5.0/3.0)))


def init_mem_rand_distmatch_sparse(n_memblocks, memblock_length, pos_val_std=14.0, sparsity=0.64):
    """ Sparse matrix with same sparsity (proportion of zeros) as natural feature maps, and with non-zero values
    following a similar distribution to the values in natural feature maps """
    mem_dense = init_mem_rand_distmatch_dense(n_memblocks, memblock_length, pos_val_std=pos_val_std)
    sparse = torch.ones(n_memblocks * memblock_length, 1)
    sparse[0:int(round(sparse.size(0) * sparsity))] = 0
    sparse = sparse.view(mem_dense.size())
    # shuffle
    idx = torch.randperm(sparse.nelement())
    sparse = sparse.view(-1)[idx].view(sparse.size())
    mem = mem_dense * sparse
    return mem


def init_mem_rand_distmatch_augmented_sparse(n_memblocks, memblock_length, pos_val_std=14.0, sparsity=0.64):
    """ Begin with a matrix designed to match the sparsity and distribution of natural feature maps (see documentation
    for init_mem_rand_distmatch_sparse). Replace all zero values with random values drawn from a distribution with a
    much smaller st. deviation (but otherwise similar) to the distribution used for the initial non-zero values. """

    mem_dense = init_mem_rand_distmatch_dense(n_memblocks, memblock_length, pos_val_std=pos_val_std)
    low_random_values = init_mem_rand_distmatch_dense(n_memblocks, memblock_length, pos_val_std=pos_val_std*0.05)

    sparse = torch.ones(n_memblocks * memblock_length, 1)
    sparse[0:int(round(sparse.size(0) * sparsity))] = 0
    sparse = sparse.view(mem_dense.size())
    # shuffle
    idx = torch.randperm(sparse.nelement())
    sparse = sparse.view(-1)[idx].view(sparse.size())

    mem = mem_dense * sparse + torch.where(sparse == 0.0, 1.0, 0.0) * low_random_values

    return mem


def init_mem_rand_everyperm(memblock_length, pos_val_std=14.0):
    """Initialize a "random every permutation" codebook matrix of size 2^memblock_length x memblock_length.

    The codebook matrix is used to reconstruct activations (outputs from a neural network layer). Therefore, the
    distribution of its values ideally should resemble that of actual activations derived from network activity.
    Activations from CIFAR100 images after the "fire9" layer of Squeezenet contain about 64% 0 values due to the ReLU
    after fire9. Discarding all zeros, the remaining positive values have a distribution resembling an exponential
    distribution, or a normal distribution with a mean of roughly zero and all negative values removed. The standard
    deviation of this distribution is roughly 14 (hence pos_val_std=14 by default).

    The initialization strategy is as follows. Begin with a matrix that contains, as its rows, every possible binary
    permutation of length memblock_length. For example, if memblock_length=3, the starting matrix will contain the following rows:
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1]
    The number of rows will equal 2^memblock_length. For every value of 1 in the binary matrix, draw a value from a standard
    normal distribution with a mean of zero and a standard deviation of pos_val_std * (5/3), take its absolute value,
    and then use it to replace the original 1. For every value of 0 in the binary matrix, perform the same steps, but
    multiply the value by 0.05 to greatly reduce its magnitude. The result is a matrix based on every possible
    binary permutation, with ones replaced by random positive values with a high standard deviation, and zeros replaced
    by random positive values with a low standard deviation.
    """
    bin_mat = torch.cartesian_prod(*[torch.tensor([0.0, 1.0]) for _ in range(memblock_length)]).float()
    hi = torch.abs(torch.randn(bin_mat.shape) * (pos_val_std * (5.0/3.0)))
    lo = torch.abs(torch.randn(bin_mat.shape) * (pos_val_std * (5.0/3.0)) * 0.05)
    mem = bin_mat * hi + torch.where(bin_mat == 0.0, 1.0, 0.0) * lo
    return mem


def init_mem_rand_everyperm_shuffled(memblock_length, pos_val_std=14):
    """Initialize a "random every permutation" codebook matrix of size 2^memblock_length x memblock_length, but shuffle its
    values randomly to ablate the every permutation structure
    """
    mem_everyperm = init_mem_rand_everyperm(memblock_length, pos_val_std=pos_val_std)
    idx = torch.randperm(mem_everyperm.nelement())
    mem = mem_everyperm.view(-1)[idx].view(mem_everyperm.size())
    return mem


def init_mem_rand_everyperm_remove_half_rows(memblock_length, pos_val_std=14.0):
    """Initialize a "random every permutation" codebook matrix of size 2^memblock_length x memblock_length, but remove half
    of its rows
    """
    mem_everyperm = init_mem_rand_everyperm(memblock_length, pos_val_std=pos_val_std)
    inds = np.arange(0, mem_everyperm.size(0), 1, dtype="uint8")
    np.random.shuffle(inds)
    inds = inds[0:int(len(inds) / 2)]
    remaining_rows = []
    for ind in inds:
        remaining_rows.append(mem_everyperm[ind, :])
    mem = torch.vstack(remaining_rows)
    return mem


def init_mem_rigid_everyperm(memblock_length, high=1, low=0, jitter_std=0):
    """Initialize a "rigid every permutation" codebook matrix of size 2^memblock_length x memblock_length.

    Begin with a binary matrix as in init_mem_rand_everyperm. But instead of randomizing based on the zeros and ones,
    replace each 1 with function parameter "high" and each 0 with "low". Then add normally distributed noise (standard
    deviation = jitter_std) to all values in the resulting matrix (this last step is skipped if jitter_std=0).
    """
    mem_binary = torch.cartesian_prod(*[torch.tensor([0.0, 1.0]) for _ in range(memblock_length)]).float()
    mem = mem_binary*high + torch.where(mem_binary == 0.0, 1.0, 0.0)*low
    if jitter_std > 0:
        mem = mem + torch.randn(mem_binary.shape) * jitter_std
    return mem


class CrumbReconstructor(nn.Module):
    def __init__(self, num_feat, spat_d1, spat_d2, mem_init="random_distmatch_sparse", n_memblocks=256, memblock_length=8, fbanks_std=14, fbanks_sparsity=0.64, load_filename=None):
        """Initialize the CrumbReconstructor, including its codebook matrix.

        Parameters:
        num_feat: integer
            Size of the feature dimension in the feature banks this unit will be reconstructing. E.g. in a 13x13
            feature bank with 512 features (size 13, 13, 512), num_feat = 512.
        spat_d1, spat_d2: integer
            The spatial dimensions of the feature bank (e.g. width and height of an image in feature space)
        mem_init: string
            Denotes which strategy to use for codebook initialization.
        n_memblocks: integer
            Number of memory blocks (rows) in the codebook matrix, where each block may be used as a unit as part of a
            reconstruction of a feature bank. May be constrained by the codebook initialization strategy - e.g.
            for "every binary permutation" approaches, must be equal to 2^memblock_length
        memblock_length: integer
            Number of features in each memory block (width of codebook matrix). Must be a factor of num_feat.
        fbanks_std: number
            Approximate standard deviation of the feature banks that this CrumbReconstructor will be reconstructing.
            Relevant only for some initialization strategies (e.g. everyperm and distmatch)
        fbanks_sparsity: number
            Approximate sparsity (proportion of zeros) of the feature banks that this CrumbReconstructor will be reconstructing.
            Relevant only for some initialization strategies (i.e. distmatch_sparse)
        """

        super(CrumbReconstructor, self).__init__()

        self.num_feat = num_feat
        self.spat_d1 = spat_d1
        self.spat_d2 = spat_d2
        self.n_memblocks = n_memblocks
        self.memblock_length = memblock_length

        assert self.num_feat % self.memblock_length == 0, "memblock_length must be a factor of num_feat = " + str(self.num_feat)

        if mem_init == "load_from_file":
            self.memory = torch.load(load_filename)
        elif mem_init == "random_standard_normal":
            self.memory = Parameter(init_mem_standard_normal(self.n_memblocks, self.memblock_length))
        elif mem_init == "normal_MobileNet":
            self.memory = Parameter(init_mem_normal_MobileNet(self.n_memblocks, self.memblock_length))
        elif mem_init == "random_uniform":
            self.memory = Parameter(init_mem_rand_uniform(self.n_memblocks, self.memblock_length))
        elif mem_init == "random_distmatch_dense":
            self.memory = Parameter(init_mem_rand_distmatch_dense(self.n_memblocks, self.memblock_length, pos_val_std=fbanks_std))
        elif mem_init == "random_distmatch_sparse":
            self.memory = Parameter(init_mem_rand_distmatch_sparse(self.n_memblocks, self.memblock_length, pos_val_std=fbanks_std, sparsity=fbanks_sparsity))
        elif mem_init == "random_distmatch_sparse_scratch_pretrain":
            self.memory = Parameter(init_mem_rand_distmatch_sparse(self.n_memblocks, self.memblock_length, pos_val_std=0.8523, sparsity=0.38))
        elif mem_init == "random_everyperm":
            assert self.n_memblocks == 2**self.memblock_length, "With random everyperm initializations, n_memblocks must be equal to 2^memblock_length. If using pretrained codebook, you can set '--memory_init_strat zeros' instead."
            self.memory = Parameter(init_mem_rand_everyperm(self.memblock_length, pos_val_std=fbanks_std))
        elif mem_init == "random_everyperm_shuffled":
            assert self.n_memblocks == 2**self.memblock_length, "With random everyperm initializations, n_memblocks must be equal to 2^memblock_length. If using pretrained codebook, you can set '--memory_init_strat zeros' instead."
            self.memory = Parameter(init_mem_rand_everyperm_shuffled(self.memblock_length, pos_val_std=fbanks_std))
        elif mem_init == "random_everyperm_remove_half_rows":
            assert self.n_memblocks == (2 ** self.memblock_length)/2, "With this 'remove half the rows' random everyperm initialization, n_memblocks must be equal to half of 2^memblock_length. If using pretrained codebook, you can set '--memory_init_strat zeros' instead."
            self.memory = Parameter(init_mem_rand_everyperm_remove_half_rows(self.memblock_length, pos_val_std=fbanks_std))
        elif mem_init in ["sample_first_batch", "zeros"]:
            self.memory = Parameter(torch.zeros(n_memblocks, memblock_length))
        else:
            raise ValueError("Invalid mem_init initialization strategy passed to HamnReconstructor constructor")

    def similarity(self, key):
        """Computes similarity of feature bank slices to "memory blocks" (rows in codebook matrxix)

        Given a batch of feature banks (key), computes similarity scores for each slice of each feature bank
        (each Zt,d,l) with each row of the "memory" codebook matrix. These can be used as attention values.
        Parameters:
        key: torch.Tensor
            A batch of feature banks (activations/outputs from one network layer that are passed into the next layer),
            separated into vectors of size memblock_length. Shape: (batch_size, spat_d1, spat_d2, num_feat/memblock_length, memblock_length)
        Returns:
        sim: torch.Tensor
            The similarity scores between each Zt,d,l in the input feature bank (key) with each memory block.
            Shape: (batch_size, spat_d1, spat_d2, num_features/memblock_length, n_memblocks)
        """
        return key.matmul(nn.functional.normalize(self.memory, p=2, dim=1).t())

    def fbank_to_att(self, feature_banks):
        """Given a batch of feature banks, generate similarity-based reading attention over an codebook matrix

        Parameters:
        feature_banks: torch.Tensor
            A batch of feature banks (activations/outputs from one network layer that are passed into the next layer).
            Shaped like (batch_size, num_features, spatial_dim1, spatial_dim2). E.g. With a batch size of 21, if we
            have a 13x13 feature map with 512 features, shape = (21, 512, 13, 13).
        Returns:
        att_read: torch.Tensor
            Reading attention to be applied over an codebook matrix.
            Shape: (batch_size, spat_d1, spat_d2, num_features/memblock_length, n_memblocks)
        """
        # Reshape feature banks for similarity function
        x = feature_banks.view(-1, self.num_feat, self.spat_d1, self.spat_d2).permute(0, 2, 3, 1)
        x = x.view(-1, self.spat_d1, self.spat_d2, int(self.num_feat/self.memblock_length), self.memblock_length)

        # Compute similarity and use as attention over codebook matrix for reconstruction
        att_read = self.similarity(x).detach()

        return att_read

    def fbank_to_top1_inds(self, feature_banks):
        """Given a batch of feature banks, generate indices to get top-1 most similar memory block to each Zt,d,l slice

        Parameters:
        feature_banks: torch.Tensor
            A batch of feature banks (activations/outputs from one network layer that are passed into the next layer).
            Shaped like (batch_size, num_features, spatial_dim1, spatial_dim2). E.g. With a batch size of 21, if we
            have a 13x13 feature map with 512 features, shape = (21, 512, 13, 13).
        Returns:
        top1_mem_inds: torch.Tensor
            Indices of most similar memory blocks for each "Zt,d,l" slice of the feature banks. Can be used to
            reconstruct a similar feature bank to the original using top1_inds_to_fbank.
            Shape: (batch_size, spat_d1, spat_d2, num_features/memblock_length)
        """

        sim = self.fbank_to_att(feature_banks)
        top1_mem_inds = torch.argmax(sim, dim=4)
        return top1_mem_inds

    def top1_inds_to_fbank(self, top1_mem_inds):
        """Apply a set of top-1 indices into codebook (from fbank_to_top1_inds) to reconstruct a batch of fbanks

        Parameters:
        top1_mem_inds: torch.Tensor
            Indices of most similar memory blocks for each "Zt,d,l" slice of the feature banks, e.g. those generated by
            the function fbank_to_top1_inds.
            Shape: (batch_size, spat_d1, spat_d2, num_features/memblock_length)
        Returns:
        feature_banks: torch.Tensor
            A batch of reconstructed feature banks, (hopefully) resembling those passed to fbank_to_top1_inds as input.
            Shaped like (batch_size, num_features, spatial_dim1, spatial_dim2). E.g. With a batch size of 21, if we
            have a 13x13 feature map with 512 features, shape = (21, 512, 13, 13).
        """

        # Select memory blocks for each Zt,d,l using top1 memory block indices
        selected_blocks = self.memory[top1_mem_inds]

        # Reshape result back into the original dimensions of the feature banks
        selected_blocks = selected_blocks.view(-1, self.spat_d1, self.spat_d2, self.num_feat).permute(0, 3, 1, 2)
        reconstr_fbanks = selected_blocks.view(-1, self.num_feat, self.spat_d1, self.spat_d2)

        return reconstr_fbanks

    def reconstruct_fbank_top1(self, feature_banks):
        """Reconstruct a given batch of feature bank using similarity-derived top-1 indices into codebook matrix

        In the reconstructed feature banks, each Zt,d,l is replaced by the contents of the memory block
        that is most similar to its original values, as determined by the output of the similarity function.
        Parameters:
        feature_banks: torch.Tensor
            A batch of feature banks (activations/outputs from one network layer that are passed into the next layer).
            Shaped like (batch_size, num_features, spatial_dim1, spatial_dim2). E.g. With a batch size size of 21, if we
            have a 13x13 feature map with 512 features, key.shape = (21, 512, 13, 13).
        Returns:
        reconstr_fbanks: torch.Tensor
            A batch of feature banks reconstructed from similarity-derived top1 indices into codebook matrix.
            Same dimensions as input feature_banks
        """

        # Obtain indices into codebook calculating similarity of each Zt,d,l to each memory block
        mem_inds = self.fbank_to_top1_inds(feature_banks)

        # Reconstruct feature banks using reading attention
        reconstr_fbanks = self.top1_inds_to_fbank(mem_inds)

        return reconstr_fbanks

    def top1_inds_to_replay_storage(self, top1_mem_inds):
        """Convert top-1 indices into replay storage form: indices into codebook reshaped to 1 row per image

        Parameters:
        top1_mem_inds: torch.Tensor
            Indices of most similar memory blocks for each "Zt,d,l" slice of the feature banks, e.g. those generated by
            the function fbank_to_top1_inds.
            Shape: (batch_size, spat_d1, spat_d2, num_features/memblock_length)
        Returns:
        replay_storage_inds: torch.Tensor
            Indices into codebook to be used for replay reconstruction, organized with 1 row per image
            Shape: (batch_size, spat_d1*spat_d2*(num_features/memblock_length))
        """

        replay_storage_inds = top1_mem_inds.view(top1_mem_inds.size(0), -1)
        return replay_storage_inds

    def replay_storage_to_top1_inds(self, replay_storage_inds):
        """Unpack replay storage (1 image per row) into top-1 indices ready to be used for memory retrieval

        E.g. Can use the unpacked indices with top1_inds_to_fbank to reconstruct feature banks using codebook
        Parameters:
        replay_storage_inds: torch.Tensor
            Indices into codebook to be used for replay reconstruction, organized with 1 row per image
            Shape: (batch_size, spat_d1*spat_d2*(num_features/memblock_length))
        Returns:
        top1_mem_inds: torch.Tensor
            Indices of most similar memory blocks for each "Zt,d,l" slice of the feature banks, e.g. those generated by
            the function fbank_to_top1_inds.
            Shape: (batch_size, spat_d1, spat_d2, num_features/memblock_length)
        """

        top1_mem_inds = replay_storage_inds.view(-1, self.spat_d1, self.spat_d2, int(self.num_feat/self.memblock_length))
        return top1_mem_inds

    def mem_stats(self, original_fbanks=None, reconstructed_fbanks=None):
        """Generate stats on values in codebook, + (optional) stats on a feature bank and its reconstruction
        """

        mem = self.memory.clone().detach().cpu()
        mem_stats = {
            "codebook_mean": round(mem.mean().item(), 5),
            "codebook_std": round(mem.std().item(), 5),
            "codebook_max": round(mem.max().item(), 5),
            "codebook_min": round(mem.min().item(), 5),
            "codebook_sparsity": round(mem.numpy()[np.where(mem.numpy() == 0)].size / torch.numel(mem), 2)
        }

        if original_fbanks is not None:
            orig_fbanks = original_fbanks.clone().detach().cpu()
            orig_fbank_stats = {
                "orig_fbanks_mean": round(orig_fbanks.mean().item(), 5),
                "orig_fbanks_std": round(orig_fbanks.std().item(), 5),
                "orig_fbanks_max": round(orig_fbanks.max().item(), 5),
                "orig_fbanks_min": round(orig_fbanks.min().item(), 5),
                "orig_fbanks_sparsity": round(orig_fbanks.numpy()[np.where(orig_fbanks.numpy() == 0)].size / torch.numel(orig_fbanks), 2)
            }
            mem_stats = {**mem_stats, **orig_fbank_stats}

        if reconstructed_fbanks is not None:
            reconstr_fbanks = reconstructed_fbanks.clone().detach().cpu()
            reconstr_fbank_stats = {
                "reconstr_fbanks_mean": round(reconstr_fbanks.mean().item(), 5),
                "reconstr_fbanks_std": round(reconstr_fbanks.std().item(), 5),
                "reconstr_fbanks_max": round(reconstr_fbanks.max().item(), 5),
                "reconstr_fbanks_min": round(reconstr_fbanks.min().item(), 5),
                "reconstr_fbanks_sparsity": round(reconstr_fbanks.numpy()[np.where(reconstr_fbanks.numpy() == 0)].size / torch.numel(reconstr_fbanks), 2)
            }
            mem_stats = {**mem_stats, **reconstr_fbank_stats}

        if original_fbanks is not None and reconstructed_fbanks is not None:
            mem_stats["reconstr_mse"] = ((orig_fbanks.numpy() - reconstr_fbanks.numpy())**2).mean(axis=None)

        return mem_stats
