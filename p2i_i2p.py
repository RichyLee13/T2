# 对于图像分块，相邻两个块之间需要有一定的叠，这主要是为了避免在处理图像时边缘信息处理不到位的情况；
def img2patchs(img, patch_size=(256, 256), overlap_size=(20, 20)):
    """
    将图像分割成多个补丁，并返回这些补丁以及它们在原始图像中的位置。

    参数:
    img: 输入图像，Numpy数组。
    patch_size: 每个补丁的大小，形如(width, height)的元组。
    overlap_size: 补丁之间的重叠大小，形如(width, height)的元组。

    返回值:
    patchs_all: 包含所有补丁的列表。
    target_size: 原始图像的大小，形如(height, width)的元组。
    remain_size: 最后一行和最后一列的剩余大小，形如(height, width)的元组。
    """
    # 获取图像尺寸
    h, w, c = img.shape
    ph, pw = patch_size
    oh, ow = overlap_size

    # 计算调整后的目标大小，以确保补丁可以完全覆盖图像
    r_h = (h - ph) % (ph - oh)
    r_w = (w - pw) % (pw - ow)

    target_w, target_h = w, h

    # 如果图像大小小于补丁大小减去重叠大小，则直接返回整个图像
    if not (h >= ph > oh and w >= pw > ow):
        img = img.permute(2, 0, 1).unsqueeze(0)
        img = F.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False)
        img = img.squeeze(0).permute(1, 2, 0)
        return [[img]], (target_h, target_w), (0, 0)

    # 计算需要分割的行数和列数
    N = math.ceil((target_h - ph) / (ph - oh)) + 1
    M = math.ceil((target_w - pw) / (pw - ow)) + 1

    # 存储所有补丁的列表
    patchs_all = []
    for n in range(N):
        patchs_row = []
        for m in range(M):

            # 根据当前行和列计算补丁的起始位置
            if n == N - 1:
                ph_start = target_h - ph
            else:
                ph_start = n * (ph - oh)

            if m == M - 1:
                pw_start = target_w - pw
            else:
                pw_start = m * (pw - ow)
            patch = img[ph_start:(ph_start + ph), pw_start:(pw_start + pw), :]
            patchs_row.append(patch)
        patchs_all.append(patchs_row)

    return patchs_all, (target_h, target_w), (r_h, r_w)

def patchs2img(patchs, target_h,target_w,r_size, overlap_size=(20, 20)):
    """
    将补丁重新拼接成原始图像。

    参数:
    patchs: 补丁列表，每个补丁是一个Numpy数组。
    r_size: 原始图像的剩余大小，形如(height, width)的元组。
    overlap_size: 补丁之间的重叠大小，形如(width, height)的元组。

    返回值:
    拼接后的图像，Numpy数组。
    """
    N = len(patchs)
    M = len(patchs[0])

    oh, ow = overlap_size

    patch_shape = patchs[0][0].shape
    ph, pw = patch_shape[:2]
    r_h, r_w = r_size

    c = 1
    # 如果只有一个补丁，则直接返回该补丁
    if N == 1 and M == 1:
        return_img = patchs[0][0]
        return_img = return_img.permute(2, 0, 1).unsqueeze(0)
        return_img = F.interpolate(return_img, size=(target_h, target_w), mode='bilinear', align_corners=False)
        return_img = return_img.squeeze(0).permute(1, 2, 0)
        return return_img
    row_imgs = []
    for n in range(N):
        row_img = patchs[n][0]
        for m in range(1, M):
            # 根据当前列计算重叠宽度
            if m == M - 1 and r_w != 0:
                ow_new = pw - r_w
            else:
                ow_new = ow
            patch = patchs[n][m]
            h, w = row_img.shape[:2]
            new_w = w + pw - ow_new
            big_row_img = np.zeros((h, new_w, c), dtype=np.float32)
            # 将 GPU Tensor 转换为 numpy 数组之前，先转移到 CPU 并使用 .detach()
            if isinstance(row_img, torch.Tensor):
                big_row_img[:, :w - ow_new, :] = row_img[:, :w - ow_new, :].detach().cpu().numpy()
            else:
                big_row_img[:, :w - ow_new, :] = row_img[:, :w - ow_new, :]
            big_row_img[:, w:, :] = patch[:, ow_new:, :].detach().cpu().numpy()
            # 处理重叠区域
            if isinstance(row_img, torch.Tensor):
                overlap_row_01 = row_img[:, w - ow_new:, :].detach().cpu().numpy()
            else:
                overlap_row_01 = row_img[:, w - ow_new:, :]
            overlap_row_02 = patch[:, :ow_new, :].detach().cpu().numpy()

            # 计算重叠区域的权重
            weight = 0.5
            overlap_row = (overlap_row_01 * (1 - weight))+ (overlap_row_02 * weight)
            big_row_img[:, w - ow_new:w, :] = overlap_row
            row_img = big_row_img

        row_imgs.append(row_img)

    column_img = row_imgs[0]
    for i in range(1, N):
        # 根据当前行计算重叠高度
        if i == N - 1 and r_h != 0:
            oh_new = ph - r_h
        else:
            oh_new = oh

        row_img = row_imgs[i]
        h, w = column_img.shape[:2]
        new_h = h + ph - oh_new

        big_column_img = np.zeros((new_h, w, c), dtype=np.float32)
        big_column_img[:h - oh_new, :, :] = column_img[:h - oh_new, :, :]
        big_column_img[h:, :, :] = row_img[oh_new:, :, :]
        # 处理重叠区域
        overlap_column_01 = column_img[h - oh_new:, :, :]
        overlap_column_02 = row_img[:oh_new, :, :]

        # 计算重叠区域的权重
        weight = 0.5
        overlap_column = (overlap_column_01 * (1 - weight))+ (overlap_column_02 * weight)
        big_column_img[h - oh_new:h, :, :] = overlap_column

        column_img = big_column_img

    return column_img