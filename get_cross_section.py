import numpy as np
import cv2
import copy
from get_dicom import window_transform_3d


def get_cross_section(dicom, if_vis, number, results_path) :
    dcm_3d_array = window_transform_3d(dicom, window_width=1700, window_center=1500).astype(np.uint8)

    # Maximum Intensity Projection
    mip_img = np.max(dcm_3d_array, axis=2)
    mip_img[mip_img != 255] = 0

    # open and close
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    mip_img = cv2.morphologyEx(mip_img, cv2.MORPH_OPEN, kernel, iterations=1)
    mip_img = cv2.morphologyEx(mip_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # get roi slice base on corner detection
    bi_img, coordinates = harris_corner_detection(mip_img, gamma=0.25)  # 对投影图使用哈里斯角点检测
    index = np.argsort(coordinates[:, 1])  # 输出第二维度从小到大排序的索引序列
    coordinates = coordinates[index]  # 将角点坐标序列排序
    del_idx = np.array([], dtype=np.int16)
    for i in range(coordinates.shape[0]):
        x = int(coordinates[i, 0])
        y = int(coordinates[i, 1])
        if np.count_nonzero(mip_img[x-1, (y-1):(y+2)]) >= 2 or x >= 200:
            del_idx = np.append(del_idx, int(i))
    
    coordinates = np.delete(coordinates, del_idx, axis=0)  # 删除不符合要求的角点
    coordinates = coordinates.astype(int)
    ori_coordinates = coordinates
    
    # find anomalies
    coordinates_y = coordinates[:, 0]
    data_std = np.std(coordinates_y)
    data_mean = np.mean(coordinates_y)
    anomaly = data_std * 3

    coordinates = []
    for num in coordinates_y:
        if num <= data_mean + anomaly and num >= data_mean - anomaly :
            coordinates.append(num)

    cs_num = np.min(coordinates[:len(coordinates) // 4])   # 选择前三个坐标中最大的
    cross_section = dicom[cs_num, :, :]
    
    if if_vis :
        mip_img = cv2.cvtColor(mip_img, cv2.COLOR_GRAY2BGR)
        cv2.line(mip_img, (0, cs_num), (mip_img.shape[1], cs_num), (255, 0, 0), 2)
        for coor in ori_coordinates:
            cv2.circle(mip_img, coor[::-1], radius=3, color=(0, 0, 255), thickness=-1)  # 填充圆
        cv2.imwrite(f"{results_path}/mip/{str(number).zfill(4)}_mip.png", mip_img)
        cv2.imwrite(f"{results_path}/cross_section/{str(number).zfill(4)}_cross_section.png", cross_section)
        
    return cross_section



def harris_corner_detection(img, gamma):
    # print(gamma)
    img = img.astype("uint8")  # 转换格式
    img = np.float32(img)
    width, height = img.shape
    
    # 对图像执行harris
    Harris_detector = cv2.cornerHarris(img, 2, 3, 0.04)  # cv库的函数
    # img - 数据类型为 float32 的输入图像。
    # blockSize - 角点检测中要考虑的领域大小。
    # ksize - Sobel 求导中使用的窗口大小
    # k - Harris 角点检测方程中的自由参数,取值参数为 [0,04,0.06]

    dst = Harris_detector
    # 设置阈值
    thres = gamma * dst.max()  # 阈值，大于thres为角点, gamma值可以考究一下
    # print('thres =', thres)
    gray_img = copy.deepcopy(img)  # 复制一个投影图
    gray_img[dst <= thres] = 0
    gray_img[dst > thres] = 255
    gray_img = gray_img.astype("uint8")

    coor = np.array([])
    for i in range(width):
        for j in range(height):
            if gray_img[i][j] == 255:  # 角点
                # print([i, j])
                coor = np.append(coor, [i, j], axis=0)  # 嵌入坐标
    
    coor = np.reshape(coor, (-1, 2))  # 变成两列
    return gray_img, coor