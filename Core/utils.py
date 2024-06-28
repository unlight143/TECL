import torch
import numpy as np
import random
import os
import shutil
import scipy.sparse as sp
from sklearn.cluster import KMeans
import pyproj
import math
from sklearn.preprocessing import StandardScaler

def Init_random_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def clearOldLogs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def compute_precision(pred_probs, y, k=4):
    values, preds = pred_probs.topk(k, dim=1, largest=True, sorted=True)
    all_corr  = 0
    recall = 0
    for i in range(y.shape[0]):
        correct = 0
        for p in preds[i]:
            if int(y[i][p]) == 1:
                correct += 1
        all_corr += correct / k
        # recall += correct / len(labels[i])
    return all_corr / y.shape[0]

def compute_recall(pred_probs, y, k=4):
    values, preds = pred_probs.topk(k, dim=1, largest=True, sorted=True)
    all_corr  = 0
    recall = 0
    for i in range(y.shape[0]):
        correct = 0
        for p in preds[i]:
            if int(y[i][p]) == 1:
                correct += 1
        # all_corr += correct / 5
        # recall += correct / torch.sum(y[i])
        recall += correct
        all_corr += torch.sum(y[i])
    return recall / all_corr

def compute_mrr(pred_probs, y):
    epoch_reciprocal_rank = []
    for pred, y1 in zip(pred_probs, y):
        y2 = torch.nonzero(y1==1)
        rank = [torch.argwhere(torch.argsort(pred, descending=True) == target).item() + 1 for target in y2]
        reciprocal_rank = (1 / torch.tensor(rank)).tolist()
        epoch_reciprocal_rank.extend(reciprocal_rank)
    mrr = np.mean(epoch_reciprocal_rank)
    return mrr

# def compute_rmse(pred_probs, y, k=4):
#     values, preds = pred_probs.topk(k, dim=1, largest=True, sorted=True)
#     all_corr  = 0
#     all_check = 0
#     rmse = 0
#     for i in range(y.shape[0]):
#         correct = 0
#         for p in preds[i]:
#             if int(y[i][p]) == 1:
#                 all_corr += 1
#         # all_corr += correct / 5
#         # recall += correct / torch.sum(y[i])
#         # recall += correct
#         all_check += torch.sum(y[i]) if torch.sum(y[i]) <= k else k
#         # all_check += torch.sum(y[i])
#     rmse = np.sqrt((all_check-all_corr) / all_check)
#     return rmse

def compute_rmse(pred_probs, y, k=4):
    all_corr  = 0
    all_check = 0
    rmse = 0
    for i in range(y.shape[0]):
        correct = 0
        num_rec = int(torch.sum(y[i]).item()) if torch.sum(y[i]) <= k else k
        values, pred = pred_probs[i].topk(k * num_rec, dim=0, largest=True, sorted=True)
        for p in pred:
            if int(y[i][p]) == 1:
                all_corr += 1
        # all_corr += correct / 5
        # recall += correct / torch.sum(y[i])
        # recall += correct
        # all_check += torch.sum(y[i]) if torch.sum(y[i]) <= k else k
        all_check += torch.sum(y[i])
    rmse = np.sqrt((all_check-all_corr) / all_check)
    return rmse

def rotation(data):
    '''
    # most frequently used degrees are 30,45,60
    input: dataframe containing Latitude(x) and Longitude(y)
    '''
    rot_45_x = (0.707 * data[0]) + (0.707 * data[1])
    rot_45_y = (0.707 * data[1]) + (0.707 * data[0])
    rot_30_x = (0.866 * data[0]) + (0.5 * data[1])
    rot_30_y = (0.866 * data[1]) + (0.5 * data[0])

    return [rot_45_x, rot_45_y, rot_30_x, rot_30_y]

def cluster(data, n_clusters):
  '''
  input: dataframe containing Latitude(x) and Longitude(y) coordinates
  output: series of cluster labels that each row of coordinates belongs to.
  '''
  model = KMeans(n_clusters=n_clusters)
  labels = model.fit_predict(data)
  return labels

def lonlat_to_mercator(data):
    res = []
    for d in data:
        lon = d[1]
        lat = d[0]
        proj_merc = pyproj.Proj('epsg:3857')
        proj_wgs84 = pyproj.Proj('epsg:4326')
        transformer = pyproj.Transformer.from_proj(proj_wgs84, proj_merc)
        x, y = transformer.transform(lon, lat)
        res.append([x, y])
    return res

def geo_to_cartesian(data, radius=6371):
    """
    Converts geographic coordinates to Cartesian coordinates.
    
    Args:
        lat (float): latitude in degrees.
        lon (float): longitude in degrees.
        radius (float): radius of the earth.
    
    Returns:
        (float, float, float): cartesian coordinates (x, y, z).
    """
    res = []
    for d in data:
        lon = d[1]
        lat = d[0]
        lat_radians = math.radians(lat)
        lon_radians = math.radians(lon)
        
        x = radius * math.cos(lat_radians) * math.cos(lon_radians)
        y = radius * math.cos(lat_radians) * math.sin(lon_radians)
        z = radius * math.sin(lat_radians)
        res.append([x, y, z])
    
    X_cartesian = np.array(res, dtype=np.dtype(float))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cartesian)
    
    return X_scaled

def normalize_coordinates(data):
    """
    Normalizes geographic coordinates to the range [-1, 1].
    
    Args:
        lat (float): latitude in degrees.
        lon (float): longitude in degrees.
    
    Returns:
        (float, float): normalized coordinates (normalized_lat, normalized_lon).
    """
    max_lat = 90.0
    max_lon = 180.0
    
    res = []
    for d in data:
        lon = d[1]
        lat = d[0]
        normalized_lat = lat / max_lat
        normalized_lon = lon / max_lon
        res.append([normalized_lat, normalized_lon])
    
    return res

def normalize_longitude_latitude(longitude_latitude):
    """
    将经纬度数据进行最小-最大规范化（Min-Max Normalization）
    """
    min_longitude = np.min(longitude_latitude[:, 0])
    max_longitude = np.max(longitude_latitude[:, 0])
    min_latitude = np.min(longitude_latitude[:, 1])
    max_latitude = np.max(longitude_latitude[:, 1])

    normalized_longitude = (longitude_latitude[:, 0] - min_longitude) / (max_longitude - min_longitude)
    normalized_latitude = (longitude_latitude[:, 1] - min_latitude) / (max_latitude - min_latitude)

    return np.column_stack((normalized_longitude, normalized_latitude))

def compute_relevant_socre(checkin, DU, DP):
    num_user = len(DU)
    num_poi = len(DP)

    relevant_socre_list = []
    D_max_U = max(DU)
    D_min_U = min(DU)
    D_max_P = max(DP)
    D_min_P = min(DP)

    for ux in range(num_user):
        tmp_rel_soc = []
        for oi in range(num_poi):
            if checkin[ux][oi] == 1:
                rel_soc = ((DU[ux] - D_min_U) / (D_max_U - D_min_U)) + ((DP[oi] - D_min_P)/(D_max_P - D_min_P))
            else:
                rel_soc = 0
            tmp_rel_soc.append(rel_soc)
        relevant_socre_list.append(tmp_rel_soc)
    
    relevant_socre_list = np.array(relevant_socre_list)
    all_mean = np.mean(relevant_socre_list)
    all_std = np.std(relevant_socre_list)
    row_mean = np.mean(relevant_socre_list, axis=1)
    row_std = np.std(relevant_socre_list, axis=1)

    new_rel_soc_list = []
    for ux in range(num_user):
        tmp_rel_soc = []
        for oi in range(num_poi):
            new_rel_soc = all_mean + (relevant_socre_list[ux][oi] - row_mean[ux]) * (all_std / row_std[ux])
            tmp_rel_soc.append(new_rel_soc)
        new_rel_soc_list.append(tmp_rel_soc)
    
    new_rel_soc_list = np.array(new_rel_soc_list, dtype=np.dtype(float))

    return new_rel_soc_list
