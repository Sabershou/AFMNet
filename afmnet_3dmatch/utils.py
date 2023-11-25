import time
import torch
from torch.nn.functional import grid_sample, interpolate, pad, softmax, unfold


class InputPadder:
    def __init__(self, dims, x=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // x) + 1) * x - self.ht) % x
        pad_wd = (((self.wd // x) + 1) * x - self.wd) % x
        self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [pad(x, self._pad, mode='replicate').contiguous() for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


class Timer:
    def __init__(self):
        self.enabled = False
        self.timing_stat = {}

    def timer_func(self, func):
        # This function shows the execution time of the function object passed
        def wrap_func(*args, **kwargs):
            if not self.enabled:
                return func(*args, **kwargs)

            t1 = time.time()
            result = func(*args, **kwargs)
            torch.cuda.synchronize()
            t2 = time.time()

            if func.__qualname__ in self.timing_stat.keys():
                self.timing_stat[func.__qualname__] += (t2 - t1) * 1000
            else:
                self.timing_stat[func.__qualname__] = (t2 - t1) * 1000

            return result

        return wrap_func

    def clear_timing_stat(self):
        self.timing_stat = {}

    def get_timing_stat(self):
        return self.timing_stat

    def set_enabled(self, enabled):
        self.enabled = enabled


timer = Timer()


def batch_indexing(batched_data: torch.Tensor, batched_indices: torch.Tensor, layout='channel_first'):
    def batch_indexing_channel_first(batched_data: torch.Tensor, batched_indices: torch.Tensor):
        """
        :param batched_data: [batch_size, C, N]
        :param batched_indices: [batch_size, I1, I2, ..., Im]
        :return: indexed data: [batch_size, C, I1, I2, ..., Im]
        """
        def product(arr):
            p = 1
            for i in arr:
                p *= i
            return p
        assert batched_data.shape[0] == batched_indices.shape[0]
        batch_size, n_channels = batched_data.shape[:2]
        indices_shape = list(batched_indices.shape[1:])
        batched_indices = batched_indices.reshape([batch_size, 1, -1])
        batched_indices = batched_indices.expand([batch_size, n_channels, product(indices_shape)])
        result = torch.gather(batched_data, dim=2, index=batched_indices.to(torch.int64))
        result = result.view([batch_size, n_channels] + indices_shape)
        return result

    def batch_indexing_channel_last(batched_data: torch.Tensor, batched_indices: torch.Tensor):
        """
        :param batched_data: [batch_size, N, C]
        :param batched_indices: [batch_size, I1, I2, ..., Im]
        :return: indexed data: [batch_size, I1, I2, ..., Im, C]
        """
        assert batched_data.shape[0] == batched_indices.shape[0]
        batch_size = batched_data.shape[0]
        view_shape = [batch_size] + [1] * (len(batched_indices.shape) - 1)
        expand_shape = [batch_size] + list(batched_indices.shape)[1:]
        indices_of_batch = torch.arange(batch_size, dtype=torch.long, device=batched_data.device)
        indices_of_batch = indices_of_batch.view(view_shape).expand(expand_shape)  # [bs, I1, I2, ..., Im]
        if len(batched_data.shape) == 2:
            return batched_data[indices_of_batch, batched_indices.to(torch.long)]
        else:
            return batched_data[indices_of_batch, batched_indices.to(torch.long), :]

    if layout == 'channel_first':
        return batch_indexing_channel_first(batched_data, batched_indices)
    elif layout == 'channel_last':
        return batch_indexing_channel_last(batched_data, batched_indices)
    else:
        raise ValueError


mesh_grid_cache = {}


def mesh_grid(n, h, w, device, channel_first=True):
    global mesh_grid_cache
    str_id = '%d,%d,%d,%s,%s' % (n, h, w, device, channel_first)
    if str_id not in mesh_grid_cache:
        x_base = torch.arange(0, w, dtype=torch.float32, device=device)[None, None, :].expand(n, h, w)
        y_base = torch.arange(0, h, dtype=torch.float32, device=device)[None, None, :].expand(n, w, h)  # NWH
        grid = torch.stack([x_base, y_base.transpose(1, 2)], 1)  # B2HW
        if not channel_first:
            grid = grid.permute(0, 2, 3, 1)  # BHW2
        mesh_grid_cache[str_id] = grid
    return mesh_grid_cache[str_id]


def backwarp_2d(x, flow12, padding_mode):
    def norm_grid(g):
        grid_norm = torch.zeros_like(g)
        grid_norm[:, 0, :, :] = 2.0 * g[:, 0, :, :] / (g.shape[3] - 1) - 1.0
        grid_norm[:, 1, :, :] = 2.0 * g[:, 1, :, :] / (g.shape[2] - 1) - 1.0
        return grid_norm.permute(0, 2, 3, 1)

    assert x.size()[-2:] == flow12.size()[-2:]
    batch_size, _, image_h, image_w = x.size()
    grid = mesh_grid(batch_size, image_h, image_w, device=x.device)
    grid = norm_grid(grid + flow12)

    return grid_sample(x, grid, padding_mode=padding_mode, align_corners=True)


def convex_upsample(flow, mask, scale_factor=8):
    """
    Upsample flow field [image_h / 8, image_w / 8, 2] -> [image_h, image_w, 2] using convex combination.
    """
    batch_size, _, image_h, image_w = flow.shape
    mask = mask.view(batch_size, 1, 9, scale_factor, scale_factor, image_h, image_w)
    mask = softmax(mask.float(), dim=2)

    up_flow = unfold(flow.float() * scale_factor, [3, 3], padding=1)
    up_flow = up_flow.view(batch_size, 2, 9, 1, 1, image_h, image_w)
    up_flow = torch.sum(mask * up_flow, dim=2)
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)

    return up_flow.reshape(batch_size, 2, image_h * scale_factor, image_w * scale_factor)


def resize_flow2d(flow, target_h, target_w):
    origin_h, origin_w = flow.shape[2:]
    if target_h == origin_h and target_w == origin_w:
        return flow
    flow = interpolate(flow, size=(target_h, target_w), mode='bilinear', align_corners=True)
    flow[:, 0] *= target_w / origin_w
    flow[:, 1] *= target_h / origin_h
    return flow


def resize_to_64x(inputs, target, x=64):
    n, c, h, w = inputs.shape

    if h % x == 0 and w % x == 0:
        return inputs, target

    resized_h, resized_w = ((h + x - 1) // x) * x, ((w + x - 1) // x) * x
    inputs = interpolate(inputs, size=(resized_h, resized_w), mode='bilinear', align_corners=True)

    if target is not None:
        target = interpolate(target, size=(resized_h, resized_w), mode='bilinear', align_corners=True)
        target[:, 0] *= resized_w / w
        target[:, 1] *= resized_h / h

    return inputs, target


def project_pc2image(pc, camera_info):
    assert pc.shape[1] == 3  # channel first
    batch_size, n_points = pc.shape[0], pc.shape[-1]

    if isinstance(camera_info['cx'], torch.Tensor):
        cx = camera_info['cx'][:, None].expand([batch_size, n_points])
        cy = camera_info['cy'][:, None].expand([batch_size, n_points])
    else:
        cx = camera_info['cx']
        cy = camera_info['cy']

    if camera_info['projection_mode'] == 'perspective':
        f = camera_info['f'][:, None].expand([batch_size, n_points])
        pc_x, pc_y, pc_z = pc[:, 0, :], pc[:, 1, :], pc[:, 2, :]
        image_x = cx + (f / pc_z) * pc_x
        image_y = cy + (f / pc_z) * pc_y
    elif camera_info['projection_mode'] == 'parallel':
        image_x = pc[:, 0, :] + cx
        image_y = pc[:, 1, :] + cy
    else:
        raise NotImplementedError

    return torch.cat([
        image_x[:, None, :],
        image_y[:, None, :],
    ], dim=1)


@torch.cuda.amp.autocast(enabled=False)
def grid_sample_wrapper(feat_2d, uv):       # uv:(n, 2) 2 on behalf of (x, y)
    uv = uv.unsqueeze(0).permute(0, 2, 1)  # (b, 2, n) 2 on behalf of (x, y)
    image_h, image_w = feat_2d.shape[2:]
    new_x = 2.0 * uv[:, 0] / (image_w - 1) - 1.0  # [bs, n_points]
    new_y = 2.0 * uv[:, 1] / (image_h - 1) - 1.0  # [bs, n_points]
    new_xy = torch.cat([new_x[:, :, None, None], new_y[:, :, None, None]], dim=-1)  # [bs, n_points, 1, 2]
    result = grid_sample(feat_2d.float(), new_xy, 'bilinear', padding_mode='reflection', align_corners=True)  # [bs, n_channels, n_points, 1]
    return result[..., 0]


def dist_reduce_sum(value):
    if torch.distributed.is_initialized():
        value_t = torch.Tensor([value]).cuda()
        torch.distributed.all_reduce(value_t)
        return value_t
    else:
        return value
