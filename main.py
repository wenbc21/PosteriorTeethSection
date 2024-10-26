import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import math
import copy
import cv2
import argparse

import torch
from torchvision import transforms
from sklearn.linear_model import LinearRegression

from network_files import MaskRCNN
from backbone import resnet50_fpn_backbone
from draw_box_utils import draw_objs, mask_tooth_center
from get_cross_section import get_cross_section
from rotate import rotate_img_and_point, rotate_plane
from get_dicom import get_dcm_3d_array, window_transform_3d
import centers_filter


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--dicom_path', type=str, default='datasets/CBCT')
    parser.add_argument('--results_path', type=str, default='results')
    parser.add_argument('--weights_file', type=str, default='weights/model_79.pth')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    return parser


def main(args) :
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")
    
    dicom_dirs = [item.path for item in os.scandir(args.dicom_path) if item.is_dir()]
    dicom_dirs.sort()
    
    os.makedirs(f"{args.results_path}/auxiliary", exist_ok=True)
    os.makedirs(f"{args.results_path}/cross_section", exist_ok=True)
    os.makedirs(f"{args.results_path}/mip", exist_ok=True)
    os.makedirs(f"{args.results_path}/instance_seg", exist_ok=True)
    os.makedirs(f"{args.results_path}/auxiliary", exist_ok=True)
    os.makedirs(f"{args.results_path}/coronal", exist_ok=True)
    
    for it in range(len(dicom_dirs)) :
        # get dicom file and adjustment value according to window
        dicom_dir = list([item.path for item in os.scandir(dicom_dirs[it]) if item.is_dir()])[0]
        dicom = get_dcm_3d_array(dicom_dir)
        
        # get cross section
        cross_section = get_cross_section(dicom, if_vis=True, number=it, results_path=args.results_path)
        
        # build maskrcnn
        backbone = resnet50_fpn_backbone()
        model = MaskRCNN(backbone,
                        num_classes=1+1,
                        rpn_score_thresh=0.5,
                        box_score_thresh=0.5)
        
        # load train weights
        assert os.path.exists(args.weights_file), "{} file dose not exist.".format(args.weights_file)
        weights_dict = torch.load(args.weights_file, map_location='cpu')
        weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
        model.load_state_dict(weights_dict)
        model.to(device)

        # from pil image to tensor, do not normalize image
        data_transform = transforms.Compose([transforms.ToTensor()])
        cross_section = Image.fromarray(cross_section).convert('RGB')
        img = data_transform(cross_section)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()
        with torch.no_grad():
            # init
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            # inference
            predictions = model(img.to(device))[0]

            # get prediction
            predict_boxes = predictions["boxes"].to("cpu").numpy() # 矩形框信息
            predict_classes = predictions["labels"].to("cpu").numpy() # 分类信息
            predict_scores = predictions["scores"].to("cpu").numpy() # 评价box/mask分数
            predict_mask = predictions["masks"].to("cpu").numpy() # mask信息，每个mask经过处理可以为每个牙齿(item)
            predict_mask = np.squeeze(predict_mask, axis=1)  # [batch, 1, h, w] -> [batch, h, w]

            if len(predict_boxes) == 0:
                print("没有检测到任何目标!")

            # visualize
            plot_img = draw_objs(cross_section,
                             boxes=predict_boxes,
                             classes=predict_classes,
                             scores=predict_scores,
                             masks=predict_mask,
                             category_index=None,
                             line_thickness=1,
                             font='arial.ttf',
                             font_size=20,
                             alpha=1,
                             box_thresh=0.96)
            # 保存预测的图片结果
            plot_img.save(f"{args.results_path}/instance_seg/{str(it+1).zfill(4)}_seg.png")

            # 用连通区域函数获取牙齿中心
            centers = mask_tooth_center(cross_section, 
                                boxes=predict_boxes, 
                                classes=predict_classes,
                                scores=predict_scores,
                                masks=predict_mask,
                                thresh=0.96)

            centers = centers_filter.delete_lowerhalf(centers) # 删除下半部分的异常点
            centers = centers_filter.delete_duplicate(centers) # 删除重复点
            
            y_index = np.argsort(centers[:, 1], kind='stable') # 按y轴排序
            y_centers = centers[y_index]
            centers = y_centers[4:]
            y_centers = y_centers[:4] # 取前4个做关键
            cred_centers = y_centers[:6] # 取前六个做平均
            x_y_index = np.argsort(y_centers[:, 0], kind='stable') # 再按x轴排序
            x_y_centers = y_centers[x_y_index]
            x_mid = np.mean(y_centers.T[0]) # 做中心轴
        
            # 把中心点分为左右两侧
            left_centers = centers[centers[:,0]<x_mid]
            left_centers = left_centers[np.argsort(left_centers[:,1])]
            right_centers = centers[centers[:,0]>=x_mid]
            right_centers = right_centers[np.argsort(right_centers[:,1])]
            
            # 计算平均最小距离，用于确定范围
            distances = []
            for j in range(len(cred_centers)) :
                min1 = 1000000
                for k in range(len(cred_centers)) :
                    if j != k :
                        dis = (cred_centers[j][0] - cred_centers[k][0]) ** 2 + (cred_centers[j][1] - cred_centers[k][1]) ** 2
                        if dis < min1:
                            min1 = dis
                distances.append(min1**0.5)
            distances = np.array(distances)
            mean = np.mean(distances)
            std = np.std(distances)
            
            # 计算牙弓曲线斜率
            if len(left_centers) >= 3 and len(right_centers) >= 3:
                left_model = LinearRegression()
                left_model.fit(left_centers[:,0].reshape(-1, 1), left_centers[:,1])
                left_coe = left_model.coef_[0]
                left_int = left_model.intercept_
                right_model = LinearRegression()
                right_model.fit(right_centers[:,0].reshape(-1, 1), right_centers[:,1])
                right_coe = right_model.coef_[0]
                right_int = right_model.intercept_
            elif len(left_centers) >= 3:
                left_model = LinearRegression()
                left_model.fit(left_centers[:,0].reshape(-1, 1), left_centers[:,1])
                left_coe = left_model.coef_[0]
                left_int = left_model.intercept_
                cross_y = x_mid * left_coe + left_int
                right_coe = -left_coe
                right_int = cross_y - right_coe * x_mid
            elif len(right_centers) >= 3:
                right_model = LinearRegression()
                right_model.fit(right_centers[:,0].reshape(-1, 1), right_centers[:,1])
                right_coe = right_model.coef_[0]
                right_int = right_model.intercept_
                cross_y = x_mid * right_coe + right_int
                left_coe = -right_coe
                left_int = cross_y - left_coe * x_mid
            else: 
                print()
                print(f"Raise error at {i+1} when computing slope!!!")
                print()
                continue
            
            # 删除间隔距离过大的点
            left_centers, right_centers = centers_filter.delete_large_interval(
                left_centers, right_centers, x_y_centers[0][1], x_y_centers[3][1])
            
            # 根据567牙位置（或预测位置）确定范围
            if len(left_centers) >= 5 and len(right_centers) >= 5:
                upper = (left_centers[2][1] + right_centers[2][1]) // 2
                bottom = (left_centers[4][1] + right_centers[4][1]) // 2
            elif len(left_centers) >= 5:
                upper = left_centers[2][1]
                bottom = left_centers[4][1]
            elif len(right_centers) >= 5:
                upper = right_centers[2][1]
                bottom = right_centers[4][1]
            else:
                if len(left_centers) >= 3 and len(right_centers) >= 3:
                    upper = (left_centers[2][1] + right_centers[2][1]) // 2
                    bottom = (mean * (5 - len(left_centers)) + (left_centers[-1][1]) + 
                        mean * (5 - len(right_centers)) + right_centers[-1][1]) // 2
                elif len(left_centers) >= 3:
                    upper = left_centers[2][1]
                    bottom = mean * (5 - len(left_centers)) + (left_centers[-1][1])
                elif len(right_centers) >= 3:
                    upper = right_centers[2][1]
                    bottom = mean * (5 - len(right_centers)) + (right_centers[-1][1])
                else:
                    if len(left_centers) == 0 and len(right_centers) == 0:
                        upper = (mean * 6 + x_y_centers[0][1] + x_y_centers[3][1]) // 2
                        bottom = (mean * 10 + x_y_centers[0][1] + x_y_centers[3][1]) // 2
                    elif len(left_centers) == len(right_centers):
                        upper = (mean * (3 - len(left_centers)) + (left_centers[-1][1]) +
                            mean * (3 - len(right_centers)) + (right_centers[-1][1])) // 2
                        bottom = (mean * (5 - len(right_centers)) + (right_centers[-1][1]) +
                            mean * (5 - len(right_centers)) + (right_centers[-1][1])) // 2
                    elif len(left_centers) > len(right_centers):
                        upper = mean * (3 - len(left_centers)) + (left_centers[-1][1])
                        bottom = mean * (5 - len(right_centers)) + (right_centers[-1][1])
                    elif len(left_centers) < len(right_centers):
                        upper = mean * (3 - len(right_centers)) + (right_centers[-1][1])
                        bottom = mean * (5 - len(right_centers)) + (right_centers[-1][1])
            
            # 设定下界为40，避免间隔太小
            if bottom - upper < 40:
                delta = (40 - (bottom - upper)) / 2
                upper = upper - delta
                bottom = bottom + delta
            # 设定每侧截图数量
            section_num = 9
            # 从4个间隔扩大到8个
            interval = (bottom - upper) / 4
            upper -= 2 * interval
            bottom += 2 * interval
            
            # 计算截面位置
            left_slope = -1 / left_coe
            right_slope = -1 / right_coe
            y_intercepts = np.linspace(upper, bottom, section_num, dtype = int)
            left_x_cross = []
            right_x_cross = []
            for y in y_intercepts :
                left_x_cross.append((y - left_int) / left_coe)
                right_x_cross.append((y - right_int) / right_coe)
            left_intercepts = []
            right_intercepts = []
            for sec in range(section_num) :
                left_intercepts.append(y_intercepts[sec] - left_slope * left_x_cross[sec])
                right_intercepts.append(y_intercepts[sec] - right_slope * right_x_cross[sec])
            
            '''
            left_slope (float) : section slope on left side
            left_x_cross (array(5)) : point location X on left sections
            right_slope (float) : section slope on right side
            right_x_cross (array(5)) : point location X on right sections
            y_intercepts (array(5)) : point location Y on both sides sections
            '''
            
            # 作图
            x_y_centers = x_y_centers.T
            left_centers = left_centers.T
            right_centers = right_centers.T
            plt.figure(figsize=(10, 10))
            plt.imshow(cross_section, cmap=plt.cm.bone)
            plt.plot(left_centers[0], left_centers[1], 'ro')
            plt.plot(right_centers[0], right_centers[1], 'yo')
            plt.plot(x_y_centers[0][0], x_y_centers[1][0], 'bo')
            plt.plot(x_y_centers[0][3], x_y_centers[1][3], 'bo')
            
            plt.axhline(upper)
            plt.axhline(bottom)
            
            vals_x = np.linspace(0, 512)
            left_vals_y = left_int + left_coe * vals_x
            plt.plot(vals_x, left_vals_y, '--', c='orange')
            right_vals_y = right_int + right_coe * vals_x
            plt.plot(vals_x, right_vals_y, '--', c='orange')
            
            for sec in range(section_num) :
                vals_x = np.linspace(left_x_cross[sec] - 50, left_x_cross[sec] + 50)
                left_vals_y = left_intercepts[sec] + left_slope * vals_x
                plt.plot(vals_x, left_vals_y, '--', c='red')
            for sec in range(section_num) :
                vals_x = np.linspace(right_x_cross[sec] - 50, right_x_cross[sec] + 50)
                right_vals_y = right_intercepts[sec] + right_slope * vals_x
                plt.plot(vals_x, right_vals_y, '--', c='red')
                
            plt.xlim(0, 512)
            plt.ylim(512, 0)
            plt.savefig(f"{args.results_path}/auxiliary/{str(it+1).zfill(4)}_direction.png")
            # plt.show()
            plt.close()
            
            
            # 将整个CBCT图像转化到0~255
            dicom = get_dcm_3d_array(dicom_dir)
            dcm_3d_array = window_transform_3d(dicom, window_width=4000, window_center=1000)
            height, depth, width = dcm_3d_array.shape
            os.makedirs(f"{args.results_path}/coronal/{str(it+1).zfill(4)}", exist_ok=True)
            # 获取左侧截面
            for sec in range(section_num):
                dcm_3d_new = copy.deepcopy(dcm_3d_array)
                angle = math.atan(left_coe) * 180 / math.pi
                center_x = depth / 2
                center_y = width / 2
                left_points = list(map(list, zip(*[left_x_cross, y_intercepts])))
                tmp_img, rotate_points = rotate_img_and_point(dcm_3d_new[0, :, :], left_points, angle, center_x, center_y)
                # print(rotate_points)
                col = int(rotate_points[sec][0])
                result = rotate_plane(dcm_3d_new, angle, col, center_x, center_y)[::-1]
                cv2.imencode(".png", result)[1].tofile(f"{args.results_path}/coronal/{str(it+1).zfill(4)}/left_{sec+1}.png")
            # 获取右侧截面
            for sec in range(section_num):
                dcm_3d_new = copy.deepcopy(dcm_3d_array)
                angle = math.atan(right_coe) * 180 / math.pi
                center_x = depth / 2
                center_y = width / 2
                right_points = list(map(list, zip(*[right_x_cross, y_intercepts])))
                tmp_img, rotate_points = rotate_img_and_point(dcm_3d_new[0, :, :], right_points, angle, center_x, center_y)
                # print(rotate_points)
                col = int(rotate_points[sec][0])
                result = rotate_plane(dcm_3d_new, angle, col, center_x, center_y)[::-1]
                cv2.imencode(".png", result)[1].tofile(f"{args.results_path}/coronal/{str(it+1).zfill(4)}/right_{sec+1}.png")
            
        print(f"{str(it+1).zfill(4)} done! ")



if __name__ == '__main__':
    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)