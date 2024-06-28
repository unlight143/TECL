from cProfile import label
from cmath import nan
from tabnanny import check
from typing import ItemsView
import torch
import numpy as np
import scipy.io as scio
import os
from sklearn.model_selection import KFold
import scipy.sparse as sp
import pandas as pd
import random
import requests
from geopy.geocoders import Nominatim

from utils import *

class dataset():
    def __init__(self, datadir = "./Datasets"):
        self.datadir = datadir + "/" + self.Name() + "/umn_foursquare_datasets/"
        self.datafile = os.path.join(self.datadir, 'checkins.dat')
        self.xfile = os.path.join(self.datadir, 'venues.dat')
        self.adjfile = os.path.join(self.datadir, 'socialgraph.dat')
        self.num_class = None

    def Data(self):
        #get data
        all_data = np.genfromtxt(self.datafile, delimiter = '|', skip_header = 2, skip_footer = 1, autostrip = True, dtype = np.dtype(str))
        Xdata = np.genfromtxt(self.datadir + 'users.dat', delimiter = '|', skip_header = 2, skip_footer = 1, autostrip = True, dtype = np.float32)
        ydata = np.genfromtxt(self.datadir + 'venues.dat', delimiter = '|', skip_header = 2, skip_footer = 3, autostrip = True, dtype = np.float32)
        adj_data = np.genfromtxt(self.datadir + 'socialgraph.dat', delimiter = '|', skip_header = 2, skip_footer = 16, autostrip = True, dtype = np.int32)

        X = []
        y = []
        data = []
        XList = []
        XSize = 800

        idx = np.array(list(Xdata[:, 0]), dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        check_x = np.array(sorted(list(set(all_data[:, 1]))), dtype=np.int32)
        # check_dic = {}

        # checkx_list = []
        # for cx in check_x:
        #     if cx not in checkx_list:
        #         if idx_map.get(cx) != None and not np.isnan(Xdata[idx_map.get(cx), 1]) and not np.isnan(Xdata[idx_map.get(cx), 2]):
        #             tmp_count = 0
        #             for d in all_data:
        #                 if int(d[1]) == cx:
        #                     tmp_count += 1
        #             if tmp_count >= 5:
        #                 if len(checkx_list) == XSize:
        #                     break
        #                 checkx_list.append(cx)
        #         if len(checkx_list) == XSize:
        #             break
        # checkx_map = {j: i for i, j in enumerate(checkx_list)}
        # print(check_dic)

        train_inds = range(XSize//2)
        test_inds = range(XSize//2, XSize)

        # get all data and X
        checkx_list = []
        for cx in check_x:
            if len(checkx_list) < XSize:
                if idx_map.get(cx) != None:
                    if not (np.isnan(Xdata[idx_map.get(cx), 1]) and np.isnan(Xdata[idx_map.get(cx), 2])):
                        X.append(Xdata[idx_map.get(cx), 1:])
                    else:
                        X.append([0.0, 0.0])
                else:
                    X.append([0.0, 0.0])
                checkx_list.append(cx)
                # X.append(rotation(Xdata[idx_map.get(cx), 1:]))
                for d in all_data:
                    if int(d[1]) == cx:
                        data.append(d[1:3])
        # for cx in checkx_list:
        #     X.append(Xdata[idx_map.get(cx), 1:])
        #     for d in all_data:
        #         if int(d[1]) == cx:
        #             data.append(d[1:3])
        
        checkx_map = {j: i for i, j in enumerate(checkx_list)}
        data = np.array(data, dtype = np.int32)
        X = np.array(X, dtype = np.dtype(float))

        # get X area
        area = []
        n_clusters = 40
        X_area = cluster(X, n_clusters)
        for a in X_area:
            tmp = [0] * n_clusters
            tmp[a] = 1
            area.append(tmp)
        X = np.array(area, dtype = np.dtype(float))

        X_train = X[train_inds]
        X_test = X[test_inds]

        # get label(y)
        idy = np.array(sorted(list(set(data[:, 1]))), dtype=np.int32)
        idy_map = {j: i for i, j in enumerate(idy)}
        self.num_class = idy.size
        for x1 in checkx_list:
            tmp = [0] * self.num_class
            for d in data:
                if d[0] == x1:
                    tmp[idy_map[int(d[1])]] = 1
            y.append(tmp)
        y = np.array(y, dtype = np.dtype(float))

        y_train = y[train_inds]
        y_test = y[test_inds]
        # get label adj
        # idy_train = np.array(sorted(list(set(data_train[:, 0]))), dtype=np.int32)
        # idy_train_map = {j: i for i, j in enumerate(idy_train)}
        train_x = np.array(checkx_list, dtype=np.int32)[train_inds]
        trainx_map = {j: i for i, j in enumerate(train_x)}
        edges = []
        edges_x = adj_data[:, 0]
        for i, x1 in enumerate(edges_x):
            if trainx_map.get(x1) != None and trainx_map.get(adj_data[i][1]) != None:
                edges.append(trainx_map[x1])
                edges.append(trainx_map[adj_data[i][1]])
        edges_len = len(edges)
        edges = np.array(edges, dtype = np.int32).reshape((edges_len//2, 2))
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(len(train_x), len(train_x)), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        adj = np.array(adj.todense(), dtype = np.dtype(float))
        # get y feature
        # y_features = []
        # all_y = np.array(list(ydata[:, 0]), dtype=np.int32)
        # all_y_map = {j: i for i, j in enumerate(all_y)}
        # for key in idy_map:
        #     if all_y_map.get(key) != None:
        #         if np.isnan(ydata[all_y_map.get(key), 1]) or np.isnan(ydata[all_y_map.get(key), 2]):
        #             y_features.append([0.0, 0.0])
        #         else:
        #             y_features.append(ydata[all_y_map.get(key), 1:])
        #     else:
        #         y_features.append([0.0, 0.0])
        # y_features = np.array(y_features, dtype = np.dtype(float))

        # get y area
        # area = []
        # y_area = cluster(y_features, n_clusters)
        # for a in y_area:
        #     tmp = [0] * n_clusters
        #     tmp[a] = 1
        #     area.append(tmp)
        # y_features = np.array(area, dtype = np.dtype(float))
        

        # build tensor
        X = torch.FloatTensor(X)
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y = torch.FloatTensor(y)
        y_train = torch.FloatTensor(y_train)
        y_test = torch.FloatTensor(y_test)
        adj = torch.FloatTensor(adj)
        # y_features = torch.FloatTensor(y_features)

        return X, y, X_train, X_test, y_train, y_test, adj

    def Name(self):
        return self.__class__.__name__

class fsq(dataset):
    pass

class dataset1():
    def __init__(self, datadir = "./Datasets"):
        self.datadir = datadir + "/" + self.Name()
        self.datafile = os.path.join(self.datadir, 'Gowalla_totalCheckins.txt')
        self.adjfile = os.path.join(self.datadir, 'Gowalla_edges.txt')
        self.num_class = None
        self.NYfile = None
        self.LAfile = None

    def Data(self):
        X = []
        y = []
        XSize = 800
        test_size = 360
        # train_inds = range(XSize//2)
        # test_inds = range(XSize//2, XSize)

        # get all data from file
        allx = []
        ally = []
        Xdata = []
        ydata = []
        checkin = []

        NYdata, LAdata = self.getNYandLA()

        # new load
        # all_check = []
        # for line in LAdata:
        #     checkx = int(line[0])
        #     checky = int(line[4])

        #     if checky not in ally:
        #         ally.append(checky)
        #         ydata.append(line[2:4])
        #     if checkx not in allx:
        #         allx.append(checkx)
        #     all_check.append([checkx, checky])
        # check_dict = {}
        # for x in allx:
        #     for ci in all_check:
        #         if ci[0] == x:
        #             y1 = ci[1]
        #             if check_dict.get(x) == None:
        #                 check_dict[x] = [y1]
        #             else:
        #                 if y1 not in check_dict[x]:
        #                     check_dict[x].append(y1)
        # checkxlist = []
        # filter_x_data = []
        # for cx, cylist in check_dict.items():
        #     if len(cylist) >= 10:
        #         if len(checkxlist) == XSize:
        #             break
        #         else:
        #             checkxlist.append(cx)
        #             for cy in cylist:
        #                 filter_x_data.append([cx, cy])
        # filter_x_data = np.array(filter_x_data, dtype = np.int32)
        # filter_x_ylist = np.array(sorted(list(set(filter_x_data[:, 1]))), dtype=np.int32)
        # check_y_dict = {}
        # for fxy in filter_x_ylist:
        #     for fxd in filter_x_data:
        #         if fxd[1] == fxy:
        #             x1 = fxd[0]
        #             if check_y_dict.get(fxy) == None:
        #                 check_y_dict[fxy] = [x1]
        #             else:
        #                 if x1 not in check_y_dict[fxy]:
        #                     check_y_dict[fxy].append(x1)
        # new load end

        for line in LAdata:
            checkx = int(line[0])
            checky = int(line[4])
            
            if checky not in ally and len(ally) < 1000:
                ally.append(checky)
                ydata.append(line[2:4])
            if checky in ally:
                if checkx not in allx:
                    allx.append(checkx)
                checkin.append([checkx, checky])
 
        checkin = np.array(checkin, dtype = np.int32)
        ydata = np.array(ydata, dtype = np.dtype(float))
        ally_map = {j: i for i, j in enumerate(ally)}

        # split data
        check_dict = {}
        for x in allx:
            for ci in checkin:
                if ci[0] == x:
                    y1 = ci[1]
                    if check_dict.get(x) == None:
                        check_dict[x] = [y1]
                    else:
                        if y1 not in check_dict[x]:
                            check_dict[x].append(y1)

        checkxlist = []
        data = []
        # NY: -0 other: +410 4: +1200 6: +600 8: +500 12: +350 14: +300 5: +900 LA: -140 other: +10 4: +100 6: +30 5: +65
        for cx, cylist in check_dict.items():
            if len(cylist) < 6:
                if len(checkxlist) == test_size-140:
                    break
                else:
                    checkxlist.append(cx)
                    for cy in cylist:
                        data.append([cx, cy])

        for cx, cylist in check_dict.items():
            if len(cylist) >= 6:
                checkxlist.append(cx)
                for cy in cylist:
                    data.append([cx, cy])

        tmp_cxllen = len(checkxlist)
        for cx, cylist in check_dict.items():
            if len(cylist) < 6:
                if len(checkxlist) == tmp_cxllen+65:
                    break
                else:
                    if cx not in checkxlist:
                        checkxlist.append(cx)
                        for cy in cylist:
                            data.append([cx, cy])

        # for cx, cylist in check_dict.items():
        #     if len(checkxlist) == test_size:
        #         break
        #     else:
        #         checkxlist.append(cx)
        #         for cy in cylist:
        #             data.append([cx, cy])

        # tmp_cxllen = len(checkxlist)
        # for cx, cylist in check_dict.items():
        #     if len(cylist) >= 10:
        #         if len(checkxlist) == tmp_cxllen+50:
        #             break
        #         else:
        #             if cx not in checkxlist:
        #                 checkxlist.append(cx)
        #                 for cy in cylist:
        #                     data.append([cx, cy])
        # tmp_cxllen = len(checkxlist)
        # for cx, cylist in check_dict.items():
        #     if len(cylist) >= 6 and len(cylist) < 10:
        #         if len(checkxlist) == tmp_cxllen+100:
        #             break
        #         else:
        #             if cx not in checkxlist:
        #                 checkxlist.append(cx)
        #                 for cy in cylist:
        #                     data.append([cx, cy])
        # NY:50 LA:30
        # tmp_cxllen = len(checkxlist)
        # for cx, cylist in check_dict.items():
        #     if len(cylist) >= 2 and len(cylist) < 6:
        #         if len(checkxlist) == tmp_cxllen+30:
        #             break
        #         else:
        #             if cx not in checkxlist:
        #                 checkxlist.append(cx)
        #                 for cy in cylist:
        #                     data.append([cx, cy])

        data = np.array(data, dtype = np.int32)
        checkylist = np.array(sorted(list(set(data[:, 1]))), dtype=np.int32)
        check_ydata = []
        for cy in checkylist:
            check_ydata.append(ydata[ally_map[cy]])
        check_ydata = np.array(check_ydata, dtype = np.dtype(float))
        checkymap = {j: i for i, j in enumerate(checkylist)}
        
        # get X feature
        area = []
        n_clusters = 200
        y_area = cluster(check_ydata, n_clusters)
        for cx in checkxlist:
            tmp = [0] * n_clusters
            for d in data:
                if d[0] == cx:
                    tmp[y_area[checkymap[d[1]]]] += 1
            X.append(tmp)
        X = np.array(X, dtype = np.dtype(float))
        X = normalize_features(X)
            
        # for a in y_area:
        #     tmp = [0] * n_clusters
        #     tmp[a] = 1
        #     area.append(tmp)
        # X = np.array(X, dtype = np.dtype(float))

        train_inds = range(test_size)
        test_inds = range(test_size, X.shape[0])
        X_train = X[train_inds]
        X_test = X[test_inds]

        # get label(y)
        idy = np.array(sorted(list(set(data[:, 1]))), dtype=np.int32)
        idy_map = {j: i for i, j in enumerate(idy)}
        self.num_class = len(idy)
        c = 0
        for x1 in checkxlist:
            tmp = [0] * self.num_class
            for d in data:
                if d[0] == x1:
                    tmp[idy_map[int(d[1])]] = 1
            c += 1
            y.append(tmp)
        y = np.array(y, dtype = np.dtype(float))

        y_train = y[train_inds]
        y_test = y[test_inds]
        # get label adj
        adj_data = np.genfromtxt(self.adjfile, delimiter = '\t', autostrip = True, dtype = np.int32)

        # idy_train = np.array(sorted(list(set(data_train[:, 0]))), dtype=np.int32)
        # idy_train_map = {j: i for i, j in enumerate(idy_train)}
        train_x = np.array(checkxlist, dtype=np.int32)[train_inds]
        trainx_map = {j: i for i, j in enumerate(train_x)}
        edges = []
        edges_x = adj_data[:, 0]
        for i, x1 in enumerate(edges_x):
            if trainx_map.get(x1) != None and trainx_map.get(adj_data[i][1]) != None:
                edges.append(trainx_map[x1])
                edges.append(trainx_map[adj_data[i][1]])
        edges_len = len(edges)
        edges = np.array(edges, dtype = np.int32).reshape((edges_len//2, 2))
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(len(train_x), len(train_x)), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        adj = np.array(adj.todense(), dtype = np.dtype(float))

        # build tensor
        X = torch.FloatTensor(X)
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y = torch.FloatTensor(y)
        y_train = torch.FloatTensor(y_train)
        y_test = torch.FloatTensor(y_test)
        adj = torch.FloatTensor(adj)

        for i in range(y_test.shape[0]):
            if torch.sum(y[i]) != 0:
                pass
            else:
                print(i)

        return X, y, X_train, X_test, y_train, y_test, adj
    
    def getNYandLA(self):
        NYcheckin = []
        LAcheckin = []

        with open(self.datafile, 'r') as datafile:
            for line in datafile:
                line_split = line.strip().split('\t')
                if len(line_split) == 5:
                    lat = float(line_split[2])
                    lon = float(line_split[3])
                    if lat >= 40.477399 and lat <= 40.916178:
                        if lon >= -74.25909 and lon <= -73.700009:
                            NYcheckin.append(line_split)
                    if lat >= 33.70366 and lat <= 34.3373:
                        if lon >= -118.66818 and lon <= -118.1553:
                            LAcheckin.append(line_split)

        # adj_data = np.genfromtxt(self.adjfile, delimiter = '\t', autostrip = True, dtype = np.int32)
        # tmp_checkin = []
        # ally = []
        # allx = []
        # for line in NYcheckin:
        #     checkx = int(line[0])
        #     checky = line[4]
            
        #     if checky not in ally:
        #         ally.append(checky)
        #     if checkx not in allx:
        #         allx.append(checkx)
        #     tmp_ally_map = {j: i for i, j in enumerate(ally)}
        #     tmp_checkin.append([checkx, tmp_ally_map[checky]])
        # tmp_checkin = np.array(tmp_checkin, dtype=np.int32)
        # checkx_list = list(set(tmp_checkin[:, 0]))
        # num_user = len(checkx_list)
        # num_poi = len(list(set(tmp_checkin[:, 1])))

        # y = []
        # idy = np.array(sorted(list(set(tmp_checkin[:, 1]))), dtype=np.int32)
        # idy_map = {j: i for i, j in enumerate(idy)}
        # for x1 in checkx_list:
        #     tmp = [0] * num_poi
        #     for d in tmp_checkin:
        #         if d[0] == x1:
        #             tmp[idy_map[int(d[1])]] = 1
        #     y.append(tmp)
        # y = np.array(y, dtype = np.int32)
        # num_checkin = np.sum(y==1)

        # checkx_map = {j:i for i,j in enumerate(checkx_list)}
        # edges = []
        # edges_x = adj_data[:, 0]
        # for i, x1 in enumerate(edges_x):
        #     if checkx_map.get(x1) != None and checkx_map.get(adj_data[i][1]) != None:
        #         edges.append(checkx_map[x1])
        #         edges.append(checkx_map[adj_data[i][1]])
        # edges_len = len(edges) // 2

        # print(num_user)
        # print(num_poi)
        # print(edges_len)
        # print(1-num_checkin/(num_user * num_poi))

        # tmp_checkin = []
        # ally = []
        # allx = []
        # for line in LAcheckin:
        #     checkx = int(line[0])
        #     checky = line[4]
            
        #     if checky not in ally:
        #         ally.append(checky)
        #     if checkx not in allx:
        #         allx.append(checkx)
        #     tmp_ally_map = {j: i for i, j in enumerate(ally)}
        #     tmp_checkin.append([checkx, tmp_ally_map[checky]])
        # tmp_checkin = np.array(tmp_checkin, dtype=np.int32)
        # checkx_list = list(set(tmp_checkin[:, 0]))
        # num_user = len(checkx_list)
        # num_poi = len(list(set(tmp_checkin[:, 1])))

        # y = []
        # idy = np.array(sorted(list(set(tmp_checkin[:, 1]))), dtype=np.int32)
        # idy_map = {j: i for i, j in enumerate(idy)}
        # for x1 in checkx_list:
        #     tmp = [0] * num_poi
        #     for d in tmp_checkin:
        #         if d[0] == x1:
        #             tmp[idy_map[int(d[1])]] = 1
        #     y.append(tmp)
        # y = np.array(y, dtype = np.int32)
        # num_checkin = np.sum(y==1)
        
        # checkx_map = {j:i for i,j in enumerate(checkx_list)}
        # edges = []
        # edges_x = adj_data[:, 0]
        # for i, x1 in enumerate(edges_x):
        #     if checkx_map.get(x1) != None and checkx_map.get(adj_data[i][1]) != None:
        #         edges.append(checkx_map[x1])
        #         edges.append(checkx_map[adj_data[i][1]])
        # edges_len = len(edges) // 2

        # print(num_user)
        # print(num_poi)
        # print(edges_len)
        # print(1-num_checkin/(num_user * num_poi))


        print(len(NYcheckin))
        print(len(LAcheckin))
        return NYcheckin, LAcheckin

    def Name(self):
        return self.__class__.__name__
    
class gowalla(dataset1):
    pass

class dataset2():
    def __init__(self, datadir = "./Datasets"):
        self.datadir = datadir + "/" + self.Name()
        self.datafile = os.path.join(self.datadir, 'Brightkite_totalCheckins.txt')
        self.adjfile = os.path.join(self.datadir, 'Brightkite_edges.txt')
        self.num_class = None

    def Data(self):
        X = []
        y = []
        XSize = 800
        test_size = 360
        # train_inds = range(XSize//2)
        # test_inds = range(XSize//2, XSize)

        # get all data from file
        allx = []
        ally = []
        Xdata = []
        ydata = []
        checkin = []

        NYdata ,LAdata = self.getNYandLA()

        # checkydict = {}
        # checkycount = {}
        # for line in NYdata:
        #     checkx = int(line[0])
        #     checky = line[4]
        #     if checkydict.get(checky) == None:
        #         checkydict[checky] = [checkx]
        #         ydata.append(line[2:4])
        #     else:
        #         if checkx not in checkydict[checky]:
        #             checkydict[checky].append(checkx)
        #     checkycount[checky] = len(checkydict[checky])
        # allylist = list(checkydict.keys())
        # allymap = {j: i for i, j in enumerate(allylist)}
        # sorted_checkycount = sorted(checkycount.items(), key=lambda x: x[1], reverse=True)
        # sorted_checkylist = []
        # for scy in sorted_checkycount[:1000]:
        #     sorted_checkylist.append(scy[0])
        # checkymap = {j: i for i, j in enumerate(sorted_checkylist)}

        # check_ydata = []
        # for cy in sorted_checkylist:
        #     check_ydata.append(ydata[allymap[cy]])
        # check_ydata = np.array(check_ydata, dtype = np.dtype(float))           

        # checkxdict = {}
        # for cy, cxlist in checkydict.items():
        #     if cy in sorted_checkylist:
        #         for cx in cxlist:
        #             if checkxdict.get(cx) == None:
        #                 checkxdict[cx] = [checkymap[cy]]
        #             else:
        #                 if checkymap[cy] not in checkxdict[cx]:
        #                     checkxdict[cx].append(checkymap[cy])
        # tmpcou = 0
        # for cx, cylist in checkxdict.items():
        #     if len(cylist) > 4:
        #         tmpcou += 1
        # print(tmpcou)

        # checkxlist = []
        # data = []
        # LA:100
        # for cx, cylist in checkxdict.items():
        #     if len(cylist) <= 4:
        #         if len(checkxlist) == test_size-100:
        #                 break
        #         checkxlist.append(cx)
        #         for cy in cylist:
        #             data.append([cx, cy])

        # tmp_cxllen = len(checkxlist)
        # for cx, cylist in checkxdict.items():
        #     if len(cylist) > 4:
        #         if cx not in checkxlist:
        #             checkxlist.append(cx)
        #             for cy in cylist:
        #                 data.append([cx, cy])

        # LA:50
        # tmp_cxllen = len(checkxlist)
        # for cx, cylist in checkxdict.items():
        #     if len(cylist) >= 2 and len(cylist) < 4:
        #         if len(checkxlist) == tmp_cxllen+100:
        #             break
        #         if cx not in checkxlist:
        #             checkxlist.append(cx)
        #             for cy in cylist:
        #                 data.append([cx, cy])


        for line in LAdata:
            checkx = int(line[0])
            checky = line[4]
            
            if checky not in ally and len(ally) < 1000:
                ally.append(checky)
                ydata.append(line[2:4])
            if checky in ally:
                if checkx not in allx:
                    allx.append(checkx)
                tmp_ally_map = {j: i for i, j in enumerate(ally)}
                checkin.append([checkx, tmp_ally_map[checky]])

        checkin = np.array(checkin, dtype = np.int32)
        ydata = np.array(ydata, dtype = np.dtype(float))
        ally_map = {j: i for i, j in enumerate(ally)}

        # split data
        check_dict = {}
        for x in allx:
            for ci in checkin:
                if ci[0] == x:
                    y1 = ci[1]
                    if check_dict.get(x) == None:
                        check_dict[x] = [y1]
                    else:
                        if y1 not in check_dict[x]:
                            check_dict[x].append(y1)

        checkxlist = []
        data = []
        # NY: -50 other: +5 4: +60 6: +20 10: +0 LA: -50 other: +100 4: +200 8,10: +60
        # NY:-40 +62 4:+20 6:+10 8:+5 10:+2 12,14:+0, 5:+15 LA:-40 +190 4:+200 6:+100 8,10:+60, 12:+40, 14:+20, 5:+150
        for cx, cylist in check_dict.items():
                if len(cylist) < 6:
                    if len(checkxlist) == test_size-40:
                        break
                    else:
                        checkxlist.append(cx)
                        for cy in cylist:
                            data.append([cx, cy])

        for cx, cylist in check_dict.items():
            if len(cylist) >= 6:
                if len(checkxlist) == 320+190:
                    break
                else:
                    checkxlist.append(cx)
                    for cy in cylist:
                        data.append([cx, cy])

        tmp_cxllen = len(checkxlist) # NY: 92 LA: 210
        for cx, cylist in check_dict.items():
            if len(cylist) < 6:
                if len(checkxlist) == tmp_cxllen+150:
                    break
                else:
                    if cx not in checkxlist:
                        checkxlist.append(cx)
                        for cy in cylist:
                            data.append([cx, cy])

        # checkxlist = allx[:XSize]
        # data = []
        
        # for x in checkxlist:
        #     for ci in checkin:
        #         if ci[0] == int(x):
        #                 data.append(ci)

        data = np.array(data, dtype = np.int32)
        checkylist = np.array(sorted(list(set(data[:, 1]))), dtype=np.int32)
        check_ydata = []
        for cy in checkylist:
            check_ydata.append(ydata[cy])
        check_ydata = np.array(check_ydata, dtype = np.dtype(float))
        checkymap = {j: i for i, j in enumerate(checkylist)}
        
        # get X feature
        area = []
        n_clusters = 160
        y_area = cluster(check_ydata, n_clusters)
        for cx in checkxlist:
            tmp = [0] * n_clusters
            for d in data:
                if d[0] == cx:
                    tmp[y_area[checkymap[d[1]]]] += 1
            X.append(tmp)
        X = np.array(X, dtype = np.dtype(float))
        X = normalize_features(X)
            
        # for a in y_area:
        #     tmp = [0] * n_clusters
        #     tmp[a] = 1
        #     area.append(tmp)
        # X = np.array(X, dtype = np.dtype(float))

        train_inds = range(test_size)
        test_inds = range(test_size, X.shape[0])
        # train_inds = range(X.shape[0]//5*3)
        # test_inds = range(X.shape[0]//5*3, X.shape[0])
        X_train = X[train_inds]
        X_test = X[test_inds]

        # get label(y)
        idy = np.array(sorted(list(set(data[:, 1]))), dtype=np.int32)
        idy_map = {j: i for i, j in enumerate(idy)}
        self.num_class = len(idy)
        c = 0
        for x1 in checkxlist:
            tmp = [0] * self.num_class
            for d in data:
                if d[0] == x1:
                    tmp[idy_map[int(d[1])]] = 1
            c += 1
            y.append(tmp)
        y = np.array(y, dtype = np.dtype(float))

        y_train = y[train_inds]
        y_test = y[test_inds]
        # get label adj
        adj_data = np.genfromtxt(self.adjfile, delimiter = '\t', autostrip = True, dtype = np.int32)

        # idy_train = np.array(sorted(list(set(data_train[:, 0]))), dtype=np.int32)
        # idy_train_map = {j: i for i, j in enumerate(idy_train)}
        train_x = np.array(checkxlist, dtype=np.int32)[train_inds]
        trainx_map = {j: i for i, j in enumerate(train_x)}
        edges = []
        edges_x = adj_data[:, 0]
        for i, x1 in enumerate(edges_x):
            if trainx_map.get(x1) != None and trainx_map.get(adj_data[i][1]) != None:
                edges.append(trainx_map[x1])
                edges.append(trainx_map[adj_data[i][1]])
        edges_len = len(edges)
        edges = np.array(edges, dtype = np.int32).reshape((edges_len//2, 2))
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(len(train_x), len(train_x)), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        adj = np.array(adj.todense(), dtype = np.dtype(float))

        # build tensor
        X = torch.FloatTensor(X)
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y = torch.FloatTensor(y)
        y_train = torch.FloatTensor(y_train)
        y_test = torch.FloatTensor(y_test)
        adj = torch.FloatTensor(adj)

        # for i in range(y_test.shape[0]):
        #     if torch.sum(y_test[i]) >= 2:
        #         pass
        #     else:
        #         print(i)

        return X, y, X_train, X_test, y_train, y_test, adj
        
    def Data1(self):
        X = []
        y = []
        XSize = 800
        test_size = 360
        # train_inds = range(XSize//2)
        # test_inds = range(XSize//2, XSize)

        # get all data from file
        allx = []
        ally = []
        Xdata = []
        ydata = []
        checkin = []

        NYdata ,LAdata = self.getNYandLA()

        for line in NYdata:
            checkx = int(line[0])
            checky = line[4]
            
            if checky not in ally and len(ally) < 2000:
                ally.append(checky)
                ydata.append(line[2:4])
            if checky in ally:
                if checkx not in allx:
                    allx.append(checkx)
                tmp_ally_map = {j: i for i, j in enumerate(ally)}
                checkin.append([checkx, tmp_ally_map[checky]])

        checkin = np.array(checkin, dtype = np.int32)
        ydata = np.array(ydata, dtype = np.dtype(float))
        ally_map = {j: i for i, j in enumerate(ally)}
        allx_map = {j: i for i, j in enumerate(allx)}
        checkxlist = list(allx_map.keys())

        # split data
        self.num_class = len(ally_map)
        c = 0
        for x1, xindex in allx_map.items():
            tmp = [0] * self.num_class
            for ck in checkin:
                if ck[0] == x1:
                    tmp[ck[1]] = 1
            c += 1
            y.append(tmp)
        y = np.array(y, dtype = np.dtype(float))
        DU = np.sum(y, axis=1)
        DO = np.sum(y, axis=0)
        X = compute_relevant_socre(y, DU, DO)

        train_inds = range(X.shape[0]//5*3)
        test_inds = range(X.shape[0]//5*3, X.shape[0])
        X_train = X[train_inds]
        X_test = X[test_inds]

        # get label(y)
        y_train = y[train_inds]
        y_test = y[test_inds]
        # get label adj
        adj_data = np.genfromtxt(self.adjfile, delimiter = '\t', autostrip = True, dtype = np.int32)

        # idy_train = np.array(sorted(list(set(data_train[:, 0]))), dtype=np.int32)
        # idy_train_map = {j: i for i, j in enumerate(idy_train)}
        train_x = np.array(checkxlist, dtype=np.int32)[train_inds]
        trainx_map = {j: i for i, j in enumerate(train_x)}
        edges = []
        edges_x = adj_data[:, 0]
        for i, x1 in enumerate(edges_x):
            if trainx_map.get(x1) != None and trainx_map.get(adj_data[i][1]) != None:
                edges.append(trainx_map[x1])
                edges.append(trainx_map[adj_data[i][1]])
        edges_len = len(edges)
        edges = np.array(edges, dtype = np.int32).reshape((edges_len//2, 2))
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(len(train_x), len(train_x)), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        adj = np.array(adj.todense(), dtype = np.dtype(float))

        # build tensor
        X = torch.FloatTensor(X)
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y = torch.FloatTensor(y)
        y_train = torch.FloatTensor(y_train)
        y_test = torch.FloatTensor(y_test)
        adj = torch.FloatTensor(adj)

        # for i in range(y_test.shape[0]):
        #     if torch.sum(y_test[i]) >= 2:
        #         pass
        #     else:
        #         print(i)

        return X, y, X_train, X_test, y_train, y_test, adj\
    
    def getNYandLA(self):
        NYcheckin = []
        LAcheckin = []

        with open(self.datafile, 'r') as datafile:
            for line in datafile:
                line_split = line.strip().split('\t')
                if len(line_split) == 5:
                    lat = float(line_split[2])
                    lon = float(line_split[3])
                    if lat >= 40.477399 and lat <= 40.916178:
                        if lon >= -74.25909 and lon <= -73.700009:
                            NYcheckin.append(line_split)
                    if lat >= 33.70366 and lat <= 34.3373:
                        if lon >= -118.66818 and lon <= -118.1553:
                            LAcheckin.append(line_split)

        # adj_data = np.genfromtxt(self.adjfile, delimiter = '\t', autostrip = True, dtype = np.int32)
        # tmp_checkin = []
        # ally = []
        # allx = []
        # for line in NYcheckin:
        #     checkx = int(line[0])
        #     checky = line[4]
            
        #     if checky not in ally:
        #         ally.append(checky)
        #     if checkx not in allx:
        #         allx.append(checkx)
        #     tmp_ally_map = {j: i for i, j in enumerate(ally)}
        #     tmp_checkin.append([checkx, tmp_ally_map[checky]])
        # tmp_checkin = np.array(tmp_checkin, dtype=np.int32)
        # checkx_list = list(set(tmp_checkin[:, 0]))
        # num_user = len(checkx_list)
        # num_poi = len(list(set(tmp_checkin[:, 1])))

        # y = []
        # idy = np.array(sorted(list(set(tmp_checkin[:, 1]))), dtype=np.int32)
        # idy_map = {j: i for i, j in enumerate(idy)}
        # for x1 in checkx_list:
        #     tmp = [0] * num_poi
        #     for d in tmp_checkin:
        #         if d[0] == x1:
        #             tmp[idy_map[int(d[1])]] = 1
        #     y.append(tmp)
        # y = np.array(y, dtype = np.int32)
        # num_checkin = np.sum(y==1)

        # checkx_map = {j:i for i,j in enumerate(checkx_list)}
        # edges = []
        # edges_x = adj_data[:, 0]
        # for i, x1 in enumerate(edges_x):
        #     if checkx_map.get(x1) != None and checkx_map.get(adj_data[i][1]) != None:
        #         edges.append(checkx_map[x1])
        #         edges.append(checkx_map[adj_data[i][1]])
        # edges_len = len(edges) // 2

        # print(num_user)
        # print(num_poi)
        # print(edges_len)
        # print(1-num_checkin/(num_user * num_poi))

        # tmp_checkin = []
        # ally = []
        # allx = []
        # for line in LAcheckin:
        #     checkx = int(line[0])
        #     checky = line[4]
            
        #     if checky not in ally:
        #         ally.append(checky)
        #     if checkx not in allx:
        #         allx.append(checkx)
        #     tmp_ally_map = {j: i for i, j in enumerate(ally)}
        #     tmp_checkin.append([checkx, tmp_ally_map[checky]])
        # tmp_checkin = np.array(tmp_checkin, dtype=np.int32)
        # checkx_list = list(set(tmp_checkin[:, 0]))
        # num_user = len(checkx_list)
        # num_poi = len(list(set(tmp_checkin[:, 1])))

        # y = []
        # idy = np.array(sorted(list(set(tmp_checkin[:, 1]))), dtype=np.int32)
        # idy_map = {j: i for i, j in enumerate(idy)}
        # for x1 in checkx_list:
        #     tmp = [0] * num_poi
        #     for d in tmp_checkin:
        #         if d[0] == x1:
        #             tmp[idy_map[int(d[1])]] = 1
        #     y.append(tmp)
        # y = np.array(y, dtype = np.int32)
        # num_checkin = np.sum(y==1)
        
        # checkx_map = {j:i for i,j in enumerate(checkx_list)}
        # edges = []
        # edges_x = adj_data[:, 0]
        # for i, x1 in enumerate(edges_x):
        #     if checkx_map.get(x1) != None and checkx_map.get(adj_data[i][1]) != None:
        #         edges.append(checkx_map[x1])
        #         edges.append(checkx_map[adj_data[i][1]])
        # edges_len = len(edges) // 2

        # print(num_user)
        # print(num_poi)
        # print(edges_len)
        # print(1-num_checkin/(num_user * num_poi))


        print(len(NYcheckin))
        print(len(LAcheckin))
        return NYcheckin, LAcheckin

    def Name(self):
        return self.__class__.__name__

class brightkite(dataset2):
    pass
        
if __name__ == '__main__':
    dataset = eval('gowalla')()
    dataset.Data()