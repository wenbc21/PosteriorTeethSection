import numpy as np

# def delete_lowerhalf(centers):
#
#     # 删除误判到下半部分的牙齿
#     return np.delete(centers, centers[:,1]>300, 0)

def delete_lowerhalf(centers):
    # 确保 centers 是一个二维数组
    centers = np.atleast_2d(centers)

    # 确保数组有足够的列
    if centers.shape[1] >= 2:
        # 删除 y 值大于 300 的行
        return np.delete(centers, np.where(centers[:, 1] > 300), axis=0)


# 原始代码
# def delete_duplicate(centers):
#     # 计算各牙与最近牙的距离
#     distances = []
#
#     for j in range(len(centers)) :
#         min1 = 1000000
#         for k in range(len(centers)) :
#             if j != k :
#                 dis = (centers[j][0] - centers[k][0]) ** 2 + (centers[j][1] - centers[k][1]) ** 2
#                 if dis < min1:
#                     min1 = dis
#         distances.append(min1**0.5)
#     distances = np.array(distances)
#     mean = np.mean(distances)
#     std = np.std(distances)
#
#     # 计算重复点的位置
#     hash = []
#     for d in range(len(distances)) :
#         # if distances[d] < mean - 2 * std :
#         if distances[d] < 15 :
#             hash.append([d, centers[d]])
#
#     # 删除重复点
#     for cc in range(len(hash) // 2):
#         # print(i+1, count, len(centers))
#         # print(hash)
#         if hash[cc][1][0] < 250:
#             if hash[cc][1][0] < hash[cc+1][1][0] :
#                 centers = np.delete(centers, hash[cc][0], 0)
#             else :
#                 centers = np.delete(centers, hash[cc+1][0], 0)
#         else :
#             if hash[cc][1][0] < hash[cc+1][1][0] :
#                 centers = np.delete(centers, hash[cc+1][0], 0)
#             else :
#                 centers = np.delete(centers, hash[cc][0], 0)
#
#     return centers
def delete_duplicate(centers):
    # 确保 centers 是一个二维数组
    centers = np.atleast_2d(centers)

    # 只有在 centers 是二维数组且至少有两列时才执行后续逻辑
    if centers.shape[1] >= 2:
        # 计算各牙与最近牙的距离
        distances = []
        for j in range(len(centers)):
            min1 = 1000000  # 初始化一个很大的数值用于存储最小距离
            for k in range(len(centers)):
                if j != k:
                    # 计算两点之间的欧几里得距离的平方
                    dis = (centers[j][0] - centers[k][0]) ** 2 + (centers[j][1] - centers[k][1]) ** 2
                    if dis < min1:
                        min1 = dis
            # 存储最小距离的平方根，即欧几里得距离
            distances.append(min1 ** 0.5)

        # 将距离列表转换为NumPy数组
        distances = np.array(distances)
        mean = np.mean(distances)
        std = np.std(distances)

        # 计算重复点的位置
        hash = []
        for d in range(len(distances)):
            # 如果距离小于15，认为是重复点
            if distances[d] < 15:
                hash.append([d, centers[d]])

        # 删除重复点
        for cc in range(len(hash) // 2):
            # 比较重复点的位置，删除较靠左的点
            if hash[cc][1][0] < 250:
                if hash[cc][1][0] < hash[cc + 1][1][0]:
                    centers = np.delete(centers, hash[cc][0], 0)
                else:
                    centers = np.delete(centers, hash[cc + 1][0], 0)
            else:
                if hash[cc][1][0] < hash[cc + 1][1][0]:
                    centers = np.delete(centers, hash[cc + 1][0], 0)
                else:
                    centers = np.delete(centers, hash[cc][0], 0)

    return centers


def delete_large_interval(left_centers, right_centers, left_baseline, right_baseline) :
    
    left_out_idx = []
    for pointidx in range(len(left_centers)) :
        if left_centers[pointidx][1] - left_baseline > 50:
            left_out_idx.append(pointidx)
        else :
            left_baseline = left_centers[pointidx][1]
            
    right_out_idx = []
    for pointidx in range(len(right_centers)) :
        if right_centers[pointidx][1] - right_baseline > 50:
            right_out_idx.append(pointidx)
        else :
            right_baseline = right_centers[pointidx][1]
            
    left_centers = np.delete(left_centers, left_out_idx, 0)
    right_centers = np.delete(right_centers, right_out_idx, 0)
    
    return left_centers, right_centers