import copy

import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from torch_scatter import scatter_add

from models.model_utils import *

NORMALIZE_EPS = 1e-5
TIMEOUT_STEPS = 81  # Reset the codebook vector if unused for more than `TIMEOUT_STEPS` steps.


class Mlp(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, act_layer=nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Conv(nn.Module):
    def __init__(self, category_dim, hidden_dim=64, negative_slope=0.2):
        super().__init__()
        self.conv = nn.Conv1d(category_dim, hidden_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class MiniPointNetOri(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.first_conv = nn.Sequential(nn.Conv1d(in_dim, 128, 1, bias=False),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(inplace=True),
                                        nn.Conv1d(128, 256, 1))

        self.second_conv = nn.Sequential(nn.Conv1d(512, 512, 1, bias=False),
                                         nn.BatchNorm1d(512),
                                         nn.ReLU(inplace=True),
                                         nn.Conv1d(512, out_dim, 1))

    def forward(self, points):
        """
        Apply convolution on each point within each cluster, followed by max pooling.
        @param points:
        @return:
        """
        feature = self.first_conv(points.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True).values
        feature = torch.cat([feature_global.expand(-1, -1, feature.shape[2]), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2).values
        return feature_global


class MiniPointNetOurs(nn.Module):
    def __init__(self, num_clusters, out_dim):
        super().__init__()
        self.num_clusters = num_clusters
        self.out_dim = out_dim
        self.first_conv = nn.Sequential(nn.Conv1d(3, 128, 1),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(inplace=True),
                                        nn.Conv1d(128, 256, 1))
        self.second_conv = nn.Sequential(nn.Conv1d(512, 512, 1),
                                         nn.BatchNorm1d(512),
                                         nn.ReLU(inplace=True),
                                         nn.Conv1d(512, self.out_dim, 1))

    def forward(self, normalized_xyz, choice):
        # --------------------------------------------------------------------------------------------------------------
        # The first convolution is used to extract features.
        # --------------------------------------------------------------------------------------------------------------
        feature = self.first_conv(normalized_xyz.transpose(1, 2))
        # --------------------------------------------------------------------------------------------------------------
        # Use `scatter_reduce` to perform max pooling within each cluster.
        # --------------------------------------------------------------------------------------------------------------
        batch_size, _, num_points = feature.shape
        expanded_choice = choice.unsqueeze(-1).expand(-1, -1, feature.shape[1])
        feature_global = torch.full((batch_size, self.num_clusters, feature.shape[1]),
                                    float('-inf'), device=feature.device)
        feature_global = feature_global.scatter_reduce(1, expanded_choice, feature.transpose(1, 2), reduce="amax")
        # --------------------------------------------------------------------------------------------------------------
        # Use the pooled global feature to update each point's feature.
        # --------------------------------------------------------------------------------------------------------------
        feature_global = torch.gather(feature_global, 1,
                                      choice.unsqueeze(-1).expand(-1, -1, feature_global.shape[-1]))
        feature = torch.cat([feature_global.transpose(1, 2), feature], dim=1)
        # --------------------------------------------------------------------------------------------------------------
        # The second convolution is used to extract features.
        # --------------------------------------------------------------------------------------------------------------
        feature = self.second_conv(feature)
        # --------------------------------------------------------------------------------------------------------------
        # Use `scatter_reduce` again to perform max pooling.
        # --------------------------------------------------------------------------------------------------------------
        expanded_choice = choice.unsqueeze(-1).expand(-1, -1, feature.shape[1])
        feature_global = torch.full((batch_size, self.num_clusters, feature.shape[1]), float('-inf'),
                                    device=feature.device)
        feature_global = feature_global.scatter_reduce(1, expanded_choice, feature.transpose(1, 2), reduce="amax")
        return feature_global


class PointCloudTokenizer(nn.Module):
    def __init__(self, num_group, group_size, in_dim, out_dim):
        super().__init__()
        self.num_groups = num_group
        self.group_size = group_size
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mini_pointnet_ori = MiniPointNetOri(self.in_dim, self.out_dim)
        self.mini_pointnet = MiniPointNetOurs(self.num_groups, self.out_dim)

    def forward(self, pts_data):
        # --------------------------------------------------------------------------------------------------------------
        # Clustering strategy: FPS + KNN or Optimal Transportation
        # --------------------------------------------------------------------------------------------------------------
        # grouped_data, center_data = point_cloud_grouping(pts_data, self.num_groups, self.group_size)
        center_data, grouped_data, choice = cluster_ot(pts_data, self.num_groups)
        # --------------------------------------------------------------------------------------------------------------
        # mini-PointNet extracts the feature of each center point.
        # --------------------------------------------------------------------------------------------------------------
        # B, G, K, C = grouped_data.shape
        # center_feature = self.mini_pointnet(grouped_data.reshape(B * G, K, C)).reshape(B, G, self.out_dim)
        center_feature = self.mini_pointnet(grouped_data, choice)
        return center_data, center_feature


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, drop_path, mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_dim=dim, hidden_dim=int(dim * mlp_ratio), out_dim=dim)
        self.attn = Attention(dim, num_heads=num_heads)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerModule(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, drop_path_rate):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.drop_path_rate = drop_path_rate
        self.blocks = nn.ModuleList([Block(dim=self.embed_dim,
                                           num_heads=self.num_heads,
                                           drop_path=self.drop_path_rate[i])
                                     for i in range(depth)
                                     ])
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, pc_tokens, pc_pos):
        x = pc_tokens
        for _, block in enumerate(self.blocks):
            x = block(x + pc_pos)
        x = self.norm(x)
        return x


class StudentModel(nn.Module):
    def __init__(self, trans_dim, encoder_depth, decoder_depth, num_heads, drop_path_rate, mask_type, mask_ratio):
        super().__init__()
        self.trans_dim = trans_dim
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.num_heads = num_heads
        self.drop_path_rate = drop_path_rate
        self.mask_type = mask_type
        self.mask_ratio = mask_ratio

        self.mask_token = nn.Parameter(torch.zeros(self.trans_dim))
        nn.init.trunc_normal_(self.mask_token, mean=0, std=0.02, a=-0.02, b=0.02)

        encoder_dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.encoder_depth)]
        self.encoder = TransformerModule(embed_dim=self.trans_dim,
                                         depth=self.encoder_depth,
                                         num_heads=self.num_heads,
                                         drop_path_rate=encoder_dpr)

        decoder_dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.decoder = TransformerModule(embed_dim=self.trans_dim,
                                         depth=self.decoder_depth,
                                         num_heads=self.num_heads,
                                         drop_path_rate=decoder_dpr)

    def forward(self, feature, pos_embed, center):
        B, G, C = feature.shape
        # --------------------------------------------------------------------------------------------------------------
        # Generate mask, positions with mask value 1 are replaced by mask_token
        # --------------------------------------------------------------------------------------------------------------
        if self.mask_type == 'rand':
            mask = mask_center_rand(center, self.mask_ratio)
        else:
            mask = mask_center_block(center, self.mask_ratio)
        corrupted_tokens = feature * (1 - mask.unsqueeze(-1)) + self.mask_token * mask.unsqueeze(-1)
        # --------------------------------------------------------------------------------------------------------------
        # Encoder only accepts visible point cloud block features
        # --------------------------------------------------------------------------------------------------------------
        visible_tokens = corrupted_tokens[~mask.bool()].reshape(B, -1, C)
        mask_tokens = corrupted_tokens[mask.bool()].reshape(B, -1, C)
        visible_pos = pos_embed[~mask.bool()].reshape(B, -1, C)
        mask_pos = pos_embed[mask.bool()].reshape(B, -1, C)
        # --------------------------------------------------------------------------------------------------------------
        # Encoder extracts features
        # --------------------------------------------------------------------------------------------------------------
        encoded_tokens = self.encoder(visible_tokens, visible_pos)
        # --------------------------------------------------------------------------------------------------------------
        # Decoder extracts features
        # --------------------------------------------------------------------------------------------------------------
        decoded_tokens = self.decoder(torch.cat([encoded_tokens, mask_tokens], dim=1),
                                      torch.cat([visible_pos, mask_pos], dim=1))
        return decoded_tokens, mask


class EMA(nn.Module):
    def __init__(self, model, update_after_step):
        """
        EMA updates the teacher model.
        @param model: student encoder
        @param update_after_step: start updating the teacher only after this step
        """
        super().__init__()
        self.online_model = model
        # --------------------------------------------------------------------------------------------------------------
        # Deep copy: self.ema_model has the same structure and parameters as model,
        # modifying one won't affect the other
        # --------------------------------------------------------------------------------------------------------------
        self.ema_model = copy.deepcopy(model)
        self.ema_model.requires_grad_(False)
        self.update_after_step = update_after_step
        self.register_buffer("step", torch.tensor([0]))

    def copy_params_from_model_to_ema(self):
        # --------------------------------------------------------------------------------------------------------------
        # Copy model parameters
        # Variables updated by the optimizer, e.g., convolution weights and biases, fully connected layer weights, etc.
        # --------------------------------------------------------------------------------------------------------------
        for ema_params, online_params in zip(list(self.ema_model.parameters()), list(self.online_model.parameters())):
            if not is_float_dtype(online_params.dtype):
                continue
            ema_params.data.copy_(online_params.data)
        # --------------------------------------------------------------------------------------------------------------
        # Copy model buffers
        # State variables not updated by the optimizer but need to be saved, e.g., running_mean and running_var in BN layers
        # --------------------------------------------------------------------------------------------------------------
        for ema_buffers, online_buffers in zip(list(self.ema_model.buffers()), list(self.online_model.buffers())):
            if not is_float_dtype(online_buffers.dtype):
                continue
            ema_buffers.data.copy_(online_buffers.data)

    @torch.no_grad()
    def update_moving_average(self, ema_model, online_model, tau_ema):
        """
        If tau_ema is close to 1, ema_model relies more on historical information and changes smoothly.
        If tau_ema is close to 0, ema_model follows the current model more closely and changes more drastically.
        @param ema_model:
        @param online_model:
        @param tau_ema:
        @return:
        """
        for (_, online_params), (_, ema_params) in zip(list(online_model.named_parameters()),
                                                       list(ema_model.named_parameters())):
            difference = ema_params.data - online_params.data
            difference.mul_(1.0 - tau_ema)
            ema_params.sub_(difference)

        for (_, online_buffer), (_, ema_buffer) in zip(list(online_model.named_buffers()),
                                                       list(ema_model.named_buffers())):
            difference = ema_buffer - online_buffer
            difference.mul_(1.0 - tau_ema)
            ema_buffer.sub_(difference)

    def update(self, tau_ema):
        step = self.step.item()
        self.step += 1
        # --------------------------------------------------------------------------------------------------------------
        # At this point, the student model parameters have been updated based on the current batch.
        # Before self.update_after_step batches, the teacher model directly copies student model parameters.
        # --------------------------------------------------------------------------------------------------------------
        if step <= self.update_after_step:
            self.copy_params_from_model_to_ema()
        else:
            self.update_moving_average(self.ema_model, self.online_model, tau_ema)

    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)


class CodebookGenerator(nn.Module):
    def __init__(self, num_words, num_channels, dic_momentum, max_buffer_size):
        """
        Codebook generator using online k-means for updates.
        """
        super(CodebookGenerator, self).__init__()
        self.buffer_ptr = 0
        self.num_words = num_words
        self.num_channels = num_channels
        self.dic_momentum = dic_momentum
        self.max_buffer_size = max_buffer_size

        # Initialize dictionary
        dictionary = torch.randn(num_words, num_channels)
        self.register_buffer("dictionary", dictionary)
        self.register_buffer("dictionary_sum", self.dictionary.clone())
        self.register_buffer("dictionary_num", torch.ones(num_words))
        self.register_buffer("dictionary_usage", torch.zeros(num_words))
        self.register_buffer("usage_history", torch.zeros(num_words))
        # Record training iteration count
        self.register_buffer("iteration_time", torch.zeros(1))
        # Record last usage step for codewords
        self.register_buffer("last_usage", torch.full((num_words,), -TIMEOUT_STEPS))
        # Anchor feature buffer pool
        self.register_buffer("feature_buffer", torch.zeros(max_buffer_size, self.num_channels))

    @torch.no_grad()
    def reset_unused_codewords(self, feature):
        """
        Replace codebook vectors that have not been used for TIMEOUT_STEPS steps.
        @param feature:
        @return:
        """
        device = feature.device
        # Find codewords unused for TIMEOUT_STEPS steps
        unused_mask = self.last_usage < (self.iteration_time - TIMEOUT_STEPS)
        if unused_mask.any():
            # Find codewords unused for TIMEOUT_STEPS steps
            num_unused = unused_mask.sum().item()
            # Find codewords unused for TIMEOUT_STEPS steps
            random_samples = feature[torch.randint(0, feature.shape[0], (num_unused,))]
            # Replace unused codewords
            self.dictionary[unused_mask] = random_samples
            self.dictionary_sum[unused_mask] = random_samples.clone()
            # Replace unused codewords
            self.dictionary_num[unused_mask] = torch.ones(num_unused).to(device)
            # Reset usage counts
            self.dictionary_usage[unused_mask] = torch.zeros(num_unused).to(device)
            # Reset usage counts
            self.last_usage[unused_mask] = self.iteration_time.long() - 1

    @torch.no_grad()
    def maintain_codewords(self, feature):
        """
        Codebook maintenance to avoid collapse.
        @param feature:
        @param feature:
        @return:
        """
        device = feature.device
        unused_mask = self.last_usage < (self.iteration_time - TIMEOUT_STEPS)  # 找到 TIMEOUT_STEPS 步未使用的码本向量
        if unused_mask.any():
            # ----------------------------------------------------------------------------------------------------------
            # For each dead codeword, select an anchor vector from the buffer pool
            # ----------------------------------------------------------------------------------------------------------
            normed_feature = F.normalize(feature, p=2, dim=-1, eps=NORMALIZE_EPS)
            normed_dictionary = F.normalize(self.dictionary[unused_mask], p=2, dim=-1, eps=NORMALIZE_EPS)
            similarity = torch.matmul(normed_dictionary, normed_feature.t())
            closest_indices = similarity.argmax(dim=1)
            anchor_samples = feature[closest_indices]
            # ----------------------------------------------------------------------------------------------------------
            # Calculate weight of anchor vectors, ranging from 0 to 1
            # The more historical updates, the less perturbation added
            # ----------------------------------------------------------------------------------------------------------
            mu_x = TIMEOUT_STEPS / 2
            epsilon = 0.1
            alpha = 1 / (1 + torch.exp(epsilon * (self.usage_history[unused_mask] - mu_x)))
            # ----------------------------------------------------------------------------------------------------------
            # Update dead codewords according to formula 8
            # ----------------------------------------------------------------------------------------------------------
            self.dictionary[unused_mask] = self.dictionary[unused_mask] * (1 - alpha[:, None]) + anchor_samples * alpha[
                                                                                                                  :,
                                                                                                                  None]
            # ----------------------------------------------------------------------------------------------------------
            # Reset some counters
            # ----------------------------------------------------------------------------------------------------------
            self.dictionary_sum[unused_mask] = self.dictionary[unused_mask].clone()
            self.dictionary_num[unused_mask] = 1  # Reset counts
            self.usage_history = self.dictionary_usage.clone()  # Save previous usage counts
            self.dictionary_usage = torch.zeros(self.num_words).to(device)  # Reset usage counts
            self.last_usage[unused_mask] = self.iteration_time.long() - 1  # Reset last usage time

    @torch.no_grad()
    def update_dictionary(self, feature, sim_index):
        """
        Update codebook using online k-means algorithm.
        """
        sorted_features, sorted_index = sort_by_labels(feature, sim_index)
        unique_index, inverse_indices, ema_counts = torch.unique(sorted_index, return_inverse=True, return_counts=True)
        ema_features = scatter_add(sorted_features, inverse_indices, dim=0)
        # --------------------------------------------------------------------------------------------------------------
        # Exponential moving average update
        # --------------------------------------------------------------------------------------------------------------
        self.dictionary_sum[unique_index] = self.dic_momentum * self.dictionary_sum[unique_index] + (
                1 - self.dic_momentum) * ema_features
        self.dictionary_num[unique_index] = self.dic_momentum * self.dictionary_num[unique_index] + (
                1 - self.dic_momentum) * ema_counts
        self.dictionary_usage[unique_index] += 1
        # --------------------------------------------------------------------------------------------------------------
        # Record last usage step of codewords
        # --------------------------------------------------------------------------------------------------------------
        self.last_usage[unique_index] = int(self.iteration_time.item())
        # --------------------------------------------------------------------------------------------------------------
        # Update dictionary vectors
        # --------------------------------------------------------------------------------------------------------------
        self.dictionary[unique_index] = self.dictionary_sum[unique_index] / self.dictionary_num[unique_index][:, None]
        self.iteration_time += 1
        # --------------------------------------------------------------------------------------------------------------
        # Codebook maintenance
        # --------------------------------------------------------------------------------------------------------------
        if self.iteration_time % TIMEOUT_STEPS == 0:
            # self.maintain_codewords(self.feature_buffer)
            self.maintain_codewords(feature)

    @torch.no_grad()
    def update_feature_buffer(self, feature):
        """
        Add batch features to buffer pool for future dead vector updates.
        :param feature:
        :return:
        """
        bsz = feature.size(0)
        end_ptr = (self.buffer_ptr + bsz) % self.max_buffer_size
        if end_ptr > self.buffer_ptr:
            self.feature_buffer[self.buffer_ptr:end_ptr] = feature
        else:
            self.feature_buffer[self.buffer_ptr:] = feature[:self.max_buffer_size - self.buffer_ptr]
            self.feature_buffer[:end_ptr] = feature[self.max_buffer_size - self.buffer_ptr:]
        self.buffer_ptr = end_ptr

    def forward(self, feature):
        """
        Compute codebook indices for features and update codebook.
        """
        feature = feature.view(-1, self.num_channels)
        # self.update_feature_buffer(feature)
        normed_feature = F.normalize(feature, p=2, dim=-1, eps=NORMALIZE_EPS)
        normed_dictionary = F.normalize(self.dictionary, p=2, dim=-1, eps=NORMALIZE_EPS)
        # --------------------------------------------------------------------------------------------------------------
        # Compute similarity between features and codebook, find nearest codeword
        # --------------------------------------------------------------------------------------------------------------
        similarity = torch.matmul(normed_feature, normed_dictionary.t())
        sim_index = torch.argmax(similarity, dim=1)
        # --------------------------------------------------------------------------------------------------------------
        # Update codebook
        # --------------------------------------------------------------------------------------------------------------
        self.update_dictionary(feature, sim_index)
        return self.dictionary


class PointTransformerClsHead(nn.Module):
    def __init__(self, trans_dim, hidden_dim, cls_dim):
        """
        Classification head
        :param trans_dim:
        :param hidden_dim:
        :param cls_dim:
        """
        super().__init__()
        self.cls_head_finetune = nn.Sequential(
            nn.Linear(trans_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, cls_dim))

    def forward(self, pts_feature):
        result = self.cls_head_finetune(pts_feature)
        return result


class FeaturePropagation(nn.Module):
    def __init__(self, in_channel, hidden_dim):
        """
        Trilinear interpolation for segmentation
        :param in_channel:
        :param hidden_dim:
        """
        super(FeaturePropagation, self).__init__()
        self.conv_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        last_channel = in_channel
        for out_channel in hidden_dim:
            self.conv_list.append(nn.Conv1d(last_channel, out_channel, 1))
            self.bn_list.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, point_xyz_few, point_feature_few, point_xyz_large, point_feature_large):
        B, N, C = point_xyz_large.shape
        _, S, _ = point_xyz_few.shape
        if S == 1:
            interpolated_points = point_feature_few.repeat(1, N, 1)
        else:
            dists = square_distance(point_xyz_large, point_xyz_few)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(point_feature_few, idx) * weight.view(B, N, 3, 1), dim=2)
        if point_feature_large is not None:
            new_point_feature_large = torch.cat([point_feature_large, interpolated_points], dim=-1)
        else:
            new_point_feature_large = interpolated_points
        new_point_feature_large = new_point_feature_large.permute(0, 2, 1)
        for conv, bn in list(zip(self.conv_list, self.bn_list)):
            new_point_feature_large = F.relu(bn(conv(new_point_feature_large)))
        return new_point_feature_large


class PointTransformerSegHead(nn.Module):
    def __init__(self, in_channels, hidden_dims, part_dim, dropout_rate=0.5):
        """
        Segmentation head
        :param in_channels:
        :param hidden_dims:
        :param part_dim:
        :param dropout_rate:
        """
        super().__init__()
        self.part_dim = part_dim

        self.conv1 = nn.Conv1d(in_channels, hidden_dims[0], 1)
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv1d(hidden_dims[0], hidden_dims[1], 1)
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv1d(hidden_dims[1], part_dim, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        return x
