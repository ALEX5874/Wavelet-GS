import torch
import torch.nn.functional as F
from torch import nn

class res_opacity(nn.Module):
    def __init__(self, feat_dim, n_offsets):
        super(res_opacity, self).__init__()
        # self.cov = nn.Sequential(
        #     nn.Linear(feat_dim * 2, feat_dim),
        #     nn.ReLU(True),
        # ).cuda()

        self.mlp = nn.Sequential(
            # nn.Linear(feat_dim + 3 + self.opacity_dist_dim, feat_dim),
            # nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh()
        ).cuda()

    def forward(self, x, y):
        # o = self.mlp(self.cov(torch.cat([x, x + y], dim=1)))
        # o = self.mlp(x + y)
        o = self.mlp(x)
        return o

class res_cov(nn.Module):
    def __init__(self, feat_dim, n_offsets):
        super(res_cov, self).__init__()
        # self.cov = nn.Sequential(
        #     nn.Linear(feat_dim * 2, feat_dim),
        #     nn.ReLU(True),
        # ).cuda()

        self.mlp = nn.Sequential(
            # nn.Linear(feat_dim + 3 + self.cov_dist_dim, feat_dim),
            # nn.ReLU(True),
            nn.Linear(feat_dim, 7 * n_offsets),
        ).cuda()

    def forward(self, x, y):
        # o = self.mlp(self.cov(torch.cat([x, x + y], dim=1)))
        # o = self.mlp(x + y)
        o = self.mlp(x)
        return o

class res_color(nn.Module):
    def __init__(self, feat_dim, n_offsets):
        super(res_color, self).__init__()
        # self.cov = nn.Sequential(
        #     nn.Linear(feat_dim * 2, feat_dim),
        #     nn.ReLU(True),
        # ).cuda()

        self.mlp = nn.Sequential(
            # nn.Linear(feat_dim + 3 + self.color_dist_dim, feat_dim),
            # nn.ReLU(True),
            nn.Linear(feat_dim, 3 * n_offsets),
            nn.Sigmoid()
        ).cuda()

    def forward(self, x, y):
        # o = self.mlp(self.cov(torch.cat([x, x + y], dim=1)))
        # o = self.mlp(x + y)
        o = self.mlp(x)
        return o

class mlp_opacity(nn.Module):
    def __init__(self, feat_dim, add_opacity_dist):
        super(mlp_opacity, self).__init__()
        self.add_opacity_dist = add_opacity_dist
        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + 3 + self.opacity_dist_dim, feat_dim),
            nn.ReLU(True),
            # nn.Linear(feat_dim, n_offsets),
            # nn.Tanh()
        ).cuda()

    def forward(self, x):
        o = self.mlp(x)
        return o

class mlp_cov(nn.Module):
    def __init__(self, feat_dim, add_cov_dist):
        super(mlp_cov, self).__init__()
        self.add_cov_dist = add_cov_dist
        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + 3 + self.cov_dist_dim, feat_dim),
            nn.ReLU(True),
            # nn.Linear(feat_dim, 7 * self.n_offsets),
        ).cuda()

    def forward(self, x):
        o = self.mlp(x)
        return o

class mlp_color(nn.Module):
    def __init__(self, feat_dim, add_color_dist):
        super(mlp_color, self).__init__()
        self.color_dist_dim = 1 if add_color_dist else 0
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + 3 + self.color_dist_dim, feat_dim),
            nn.ReLU(True),
            # nn.Linear(feat_dim, 3 * self.n_offsets),
            # nn.Sigmoid()
        ).cuda()

    def forward(self, x):
        o = self.mlp(x)
        return o


class del_opacity(nn.Module):
    def __init__(self, feat_dim, add_opacity_dist, n_offsets=2):
        super(del_opacity, self).__init__()
        self.add_opacity_dist = add_opacity_dist
        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + 3 + self.opacity_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh()
        ).cuda()

    def forward(self, x):
        o = self.mlp(x)
        return o


class del_cov(nn.Module):
    def __init__(self, feat_dim, add_cov_dist, n_offsets=2):
        super(del_cov, self).__init__()
        self.add_cov_dist = add_cov_dist
        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + 3 + self.cov_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7 * n_offsets),
        ).cuda()

    def forward(self, x):
        o = self.mlp(x)
        return o


class del_color(nn.Module):
    def __init__(self, feat_dim, add_color_dist, n_offsets=2):
        super(del_color, self).__init__()
        self.color_dist_dim = 1 if add_color_dist else 0
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + 3 + self.color_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3 * n_offsets),
            nn.Sigmoid()
        ).cuda()

    def forward(self, x):
        o = self.mlp(x)
        return o


class albedo_mlp(nn.Module):
    def __init__(self, feat_dim, n_offsets=2):
        super(albedo_mlp, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3 * n_offsets),
            nn.Tanh()
        ).cuda()

    def forward(self, x):
        o = self.mlp(x)
        o = o.reshape(-1, 3)
        return o



class lumi_sh(nn.Module):
    def __init__(self, feat_dim, max_sh_degree, add_color_dist, n_offsets=2):
        super(lumi_sh, self).__init__()
        self.color_dist_dim = 1 if add_color_dist else 0

        self.pos = nn.Sequential(
            nn.Linear(feat_dim + 3 + self.color_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3 * n_offsets),
            nn.Sigmoid()
        ).cuda()


        self.neg = nn.Sequential(
            nn.Linear(feat_dim + 3 + self.color_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3 * n_offsets),
            nn.Tanh()
        ).cuda()

        degree = (max_sh_degree + 1) ** 2

        self.posr = nn.Linear(1, degree - 1)
        self.negr = nn.Linear(1, degree - 1)

        # self.n_offsets = n_offsets

    def forward(self, x):
        f_pos = self.pos(x)
        f_neg = self.neg(x)
        f_pos = f_pos.reshape([-1, 3]).unsqueeze(-1)
        f_neg = f_neg.reshape([-1, 3]).unsqueeze(-1)
        f_pos = torch.cat((f_pos, self.posr(f_pos)), dim=-1)
        f_neg = torch.cat((f_neg, self.negr(f_neg)), dim=-1)
        return f_pos, f_neg


class env_sh_mlp(nn.Module):
    def __init__(self, sh_degree=2, mlp_W=32, mlp_D=3, N_a=32, N_vocab=100, pixel_num=100):
        """
        """
        super().__init__()

        self.D = mlp_D - 1
        self.W = mlp_W
        self.N_a = N_a

        self.sh_degree = sh_degree
        self.features_dc_dim = 1
        if self.sh_degree == 0:
            self.features_rest_dim = 0
        elif self.sh_degree == 1:
            self.features_rest_dim = 3
        elif self.sh_degree == 2:
            self.features_rest_dim = 8
        elif self.sh_degree == 3:
            self.features_rest_dim = 15
        else:
            raise NotImplemented('sh>3 not implemented')

        self.inputs_dim = self.N_a

        # encoding layers
        for i in range(self.D):
            if i == 0:
                layer = nn.Linear(self.inputs_dim, self.W)
            else:
                layer = nn.Linear(self.W, self.W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"encoding_{i + 1}", layer)

        # self.embedding = torch.nn.Embedding(N_vocab, self.N_a).cuda()

        # self.idx_sh = nn.Linear(self.W, 3 * (self.features_dc_dim + self.features_rest_dim))

        self.env_sh = nn.Sequential(
            nn.Linear(pixel_num, mlp_W),
            nn.ReLU(True),
            nn.Linear(mlp_W, mlp_W),
            nn.Tanh(),
            nn.Linear(mlp_W, 3 * (self.features_dc_dim + self.features_rest_dim)),
            nn.Tanh()
        ).cuda()

    def forward(self, id, env):
        """
        """
        outputs_env = self.env_sh(env)
        outputs_env  = outputs_env.reshape(3, (self.features_dc_dim + self.features_rest_dim))
        return outputs_env





def wavelet_decompose_pointcloud(point_cloud, wavelet_name="db2", levels=1):
    """
    Perform multi-level wavelet decomposition on a 3D point cloud with tensor batch operations.

    Args:
        point_cloud (torch.Tensor): Input tensor of shape [N, 3].
        wavelet_name (str): Name of the wavelet filter (e.g., "db2", "sym16").
        levels (int): Number of decomposition levels.

    Returns:
        dict: Decomposition result containing approximation and detail coefficients for each level.
    """
    low_pass, high_pass = get_wavelet_filters(wavelet_name)
    low_pass, high_pass = low_pass.to(point_cloud.device), high_pass.to(point_cloud.device)

    # Initialize decomposition results
    decomposed = {"approximation": point_cloud.clone(), "details": []}

    for level in range(levels):
        current = decomposed["approximation"]  # Current approximation
        # Add batch and channel dimensions for conv1d
        current = current.T.unsqueeze(0).unsqueeze(0)  # [N, 3] -> [3, N] -> [1, 1, 3*N]

        # Perform low-pass and high-pass filtering along all axes in parallel
        low_coeff = torch.nn.functional.conv1d(
            current, low_pass.unsqueeze(0).unsqueeze(0), stride=2, padding=len(low_pass) // 2
        ).squeeze()  # Result shape: [1, 1, 3*N//2] -> [3*N//2]

        high_coeff = torch.nn.functional.conv1d(
            current, high_pass.unsqueeze(0).unsqueeze(0), stride=2, padding=len(high_pass) // 2
        ).squeeze()

        # Split back into 3 axes
        low_coeff = low_coeff.view(3, -1).T  # [3*N//2] -> [N//2, 3]
        high_coeff = high_coeff.view(3, -1).T

        # Store details and update approximation
        decomposed["details"].append(high_coeff)
        decomposed["approximation"] = low_coeff

    return decomposed


def wavelet_reconstruct_pointcloud(decomposed, wavelet_name="db2"):
    """
    Perform wavelet reconstruction on a 3D point cloud with tensor batch operations.

    Args:
        decomposed (dict): Decomposed data containing approximation and details.
        wavelet_name (str): Name of the wavelet filter (e.g., "db2", "sym16").

    Returns:
        torch.Tensor: Reconstructed point cloud of shape [N, 3].
    """
    low_pass, high_pass = get_wavelet_filters(wavelet_name)
    low_pass, high_pass = low_pass.to(decomposed["approximation"].device), high_pass.to(decomposed["approximation"].device)

    current = decomposed["approximation"]

    for level in reversed(range(len(decomposed["details"]))):
        details = decomposed["details"][level]

        # Add batch and channel dimensions for conv_transpose1d
        low_coeff = current.T.unsqueeze(0).unsqueeze(0)  # [N//2, 3] -> [3, N//2] -> [1, 1, 3*N//2]
        high_coeff = details.T.unsqueeze(0).unsqueeze(0)

        # Perform transposed convolutions for reconstruction
        upsampled_low = torch.nn.functional.conv_transpose1d(
            low_coeff, low_pass.unsqueeze(0).unsqueeze(0), stride=2
        ).squeeze()

        upsampled_high = torch.nn.functional.conv_transpose1d(
            high_coeff, high_pass.unsqueeze(0).unsqueeze(0), stride=2
        ).squeeze()

        # Combine low and high-frequency components
        reconstructed = upsampled_low + upsampled_high

        # Reshape back to point cloud dimensions
        current = reconstructed.view(3, -1).T  # [3*N] -> [N, 3]

    return current




def multi_level_wavelet_decompose(input_tensor, wavelet_name="haar", levels=2):
    """
    Perform multi-level 3D wavelet decomposition.
    Args:
        input_tensor (torch.Tensor): Input tensor of shape (B, C, D, H, W).
        wavelet_name (str): Type of wavelet ("haar", "daubechies", etc.).
        levels (int): Number of decomposition levels.
    Returns:
        list: A list of decomposed subbands for each level.
    """
    all_subbands = []
    current_tensor = input_tensor

    for level in range(levels):
        # Perform single-level decomposition
        subbands = wavelet_decompose_3d(current_tensor, wavelet_name=wavelet_name)
        all_subbands.append(subbands)

        # The low-frequency component (LLL) becomes the input for the next level
        current_tensor = subbands[:, :current_tensor.size(1) // 8, :, :, :]  # Extract LLL component

    return all_subbands


def multi_level_wavelet_reconstruct(all_subbands, wavelet_name="haar"):
    """
    Perform multi-level 3D wavelet reconstruction.
    Args:
        all_subbands (list): List of subbands from multi-level decomposition.
        wavelet_name (str): Type of wavelet ("haar", "daubechies", etc.).
    Returns:
        torch.Tensor: Reconstructed tensor of original shape.
    """
    current_tensor = None

    # Start from the last level and reconstruct iteratively
    for level in reversed(range(len(all_subbands))):
        subbands = all_subbands[level]
        if current_tensor is None:
            current_tensor = wavelet_reconstruct_3d(subbands, wavelet_name=wavelet_name)
        else:
            # Reconstruct the previous level
            subbands[:, :current_tensor.size(1), :, :, :] = current_tensor
            current_tensor = wavelet_reconstruct_3d(subbands, wavelet_name=wavelet_name)

    return current_tensor


def wavelet_decompose_3d(input_tensor, wavelet_name="haar"):
    """
    Perform 3D wavelet decomposition using the specified wavelet filters.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape (B, C, D, H, W).
        wavelet_name (str): Type of wavelet filter to use ("haar", "daubechies").

    Returns:
        torch.Tensor: Decomposed subbands of shape (B, C*8, D/2, H/2, W/2).
    """
    filters = get_wavelet_filters(wavelet_name).to(input_tensor.device)  # [8, D, H, W]
    filters = filters.unsqueeze(1)  # Shape: [8, 1, D, H, W]

    # Grouped convolution to apply all filters in one operation
    subbands = F.conv3d(input_tensor, filters, stride=2, padding=0, groups=1)

    # Reshape to merge subbands
    B, C, D, H, W = subbands.shape
    subbands = subbands.view(B, C // 8, 8, D, H, W).permute(0, 2, 1, 3, 4, 5).reshape(B, C * 8, D, H, W)
    return subbands


def wavelet_reconstruct_3d(subbands, wavelet_name="haar"):
    """
    Perform 3D wavelet reconstruction using the specified wavelet filters.

    Args:
        subbands (torch.Tensor): Subbands from wavelet decomposition (B, C*8, D/2, H/2, W/2).
        wavelet_name (str): Type of wavelet filter to use ("haar", "daubechies").

    Returns:
        torch.Tensor: Reconstructed 3D tensor (B, C, D, H, W).
    """
    filters = get_wavelet_filters(wavelet_name).to(subbands.device)  # [8, D, H, W]
    filters = filters.unsqueeze(1)  # Shape: [8, 1, D, H, W]

    # Reshape subbands to match filter grouping
    B, C, D, H, W = subbands.shape
    subbands = subbands.view(B, 8, C // 8, D, H, W).permute(0, 2, 1, 3, 4, 5).reshape(B * (C // 8), 8, D, H, W)

    # Apply transposed convolution to reconstruct
    reconstructed = F.conv_transpose3d(subbands, filters, stride=2, padding=0, groups=1)

    # Reshape back to original shape
    reconstructed = reconstructed.view(B, C // 8, D * 2, H * 2, W * 2)
    return reconstructed


def get_wavelet_filters(wavelet_name="haar"):
    """
    Generate 3D wavelet filters for the specified wavelet type.
    Supports "haar" and "daubechies".
    """
    if wavelet_name.lower() == "haar":
        # Haar wavelet filters
        low_pass = torch.tensor([1.0, 1.0]) / 2.0
        high_pass = torch.tensor([1.0, -1.0]) / 2.0

    elif wavelet_name.lower() == "db2":
        # Daubechies 2 filters
        low_pass = torch.tensor([0.48296, 0.83652, 0.22414, -0.12941])
        high_pass = torch.tensor([-0.12941, -0.22414, 0.83652, -0.48296])

    elif wavelet_name.lower() == "sym2":
        # Symlet 2 filters (identical to Daubechies 2)
        low_pass = torch.tensor([0.48296, 0.83652, 0.22414, -0.12941])
        high_pass = torch.tensor([-0.12941, -0.22414, 0.83652, -0.48296])

    elif wavelet_name.lower() == "db8":
        # Daubechies 8 filters (low-pass and high-pass)
        low_pass = torch.tensor([
            -0.00011747678400228192, 0.0006754494059985568, -0.0003917403733770651,
            -0.004870352993451574, 0.008746094047015654, 0.013981027917015516,
            -0.04408825393106472, -0.017369301001807546, 0.128747426620186,
            0.00047248457399797254, -0.284015542961582, -0.01582910525634921,
            0.5853546836548691, 0.6756307362980128, 0.3128715909144659,
            0.05441584224308161
        ])
        high_pass = torch.tensor([
            -0.05441584224308161, 0.3128715909144659, -0.6756307362980128,
            0.5853546836548691, 0.01582910525634921, -0.284015542961582,
            -0.00047248457399797254, 0.128747426620186, 0.017369301001807546,
            -0.04408825393106472, -0.013981027917015516, 0.008746094047015654,
            0.004870352993451574, -0.0003917403733770651, -0.0006754494059985568,
            -0.00011747678400228192
        ])

    elif wavelet_name.lower() == "sym16":
        # Symlet 16 filters (low-pass and high-pass)
        low_pass = torch.tensor([
            -0.0000548305866227465, 0.00039843567297594335, -0.00015631628241694423,
            -0.002638236236268469, 0.003976749970286799, 0.006839599965757502,
            -0.01203240006240691, -0.01752494256510462, 0.026305552950737684,
            0.03275449309325558, -0.03575470106699817, -0.09687026068692348,
            0.06673479523799974, 0.5020280111489308, 0.8152060002043817,
            0.41784910915027457
        ])
        high_pass = torch.tensor([
            -0.41784910915027457, 0.8152060002043817, -0.5020280111489308,
            0.06673479523799974, 0.09687026068692348, -0.03575470106699817,
            -0.03275449309325558, 0.026305552950737684, 0.01752494256510462,
            -0.01203240006240691, -0.006839599965757502, 0.003976749970286799,
            0.002638236236268469, -0.00015631628241694423, -0.00039843567297594335,
            -0.0000548305866227465
        ])

    elif wavelet_name.lower() == "coif1":
        # Coiflet 1 filters
        low_pass = torch.tensor([
            -0.015655, -0.072732, 0.384865, 0.852572,
            0.337897, -0.072732, -0.015655
        ])
        high_pass = torch.tensor([
            0.015655, -0.072732, -0.337897, 0.852572,
            -0.384865, -0.072732, 0.015655
        ])
    else:
        raise ValueError(f"Unsupported wavelet type: {wavelet_name}")

    # Stack filters for batch operation
    filters_1d = torch.stack([low_pass, high_pass], dim=0)  # Shape: [2, L]

    # Compute outer product using broadcasting
    filters_3d = torch.einsum('ai,bj,ck->abcijk', filters_1d, filters_1d, filters_1d)
    filters_3d = filters_3d.reshape(-1, *filters_3d.shape[-3:])  # Flatten the first 3 dimensions
    return filters_3d

