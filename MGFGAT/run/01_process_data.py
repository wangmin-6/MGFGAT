import torch
import torch_geometric
import numpy as np
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import os
import csv
import numpy as np
import scipy.io as scio
from sklearn.neighbors import kneighbors_graph


def getid(str):
    i = str.find("Sub_")
    return str[i + 4:i + 16]
def getname(str):
    i = str.find("Sub_")
    return str[i + 4:i + 14]
def takethrid(elem):
    return elem[2]


def get_1NN(Dist):
    # construct a knn graph
    neighbors_graph = kneighbors_graph(
        Dist, 1, mode='distance', include_self=False)
    W = 0.5 * (neighbors_graph + neighbors_graph.T)
    return W

from MGFGAT.util.ConsensusGraphLearning import cgl


def create_graph_fusion(corr_temp, feature):
    u = []
    v = []
    w = []
    knn_graph = get_1NN(corr_temp)
    knn_graph = knn_graph.todense()
    knn_graph = np.array(knn_graph)
    # 归一化
    knn_graph = (knn_graph - knn_graph.min()) / (knn_graph.max() - knn_graph.min())
    mutli_graph = np.zeros((corr_temp.shape[0],corr_temp.shape[0],2))
    mutli_graph[:,:,0] = corr_temp
    mutli_graph[:,:,1] = knn_graph

    corr = cgl(mutli_graph,100,4)

    return corr


def train_val_test_lable():
    train_val_test_dic = {}
    label_dic = {}
    with open("train_val_test_lable.csv", "r", encoding='utf-8') as f:
        reader = csv.reader(f)
        for line in reader:
            train_val_test_dic[line[0]] = line[6]
            label_dic[line[0]] = line[2]
    return train_val_test_dic, label_dic

def gen_graph(fusion,classa):  # 处理数据的函数,最关键（怎么创建，怎么保存）
    ADS = {}
    CNS = {}
    MCIS = {}
    EMCIS = {}
    LMCIS = {}
    ADS_feature = {}
    CNS_feature = {}
    MCIS_feature = {}
    EMCIS_feature = {}
    LMCIS_feature = {}
    fusion_corr = {}
    '''Functional connectivity matrix storage path '''
    path_Philips = "F:/data/ROICorr/Philips"
    path_SIEMENS = "F:/data/ROICorr/SIEMENS"
    files_SIEMENS = os.listdir(path_SIEMENS)
    files_Philips = os.listdir(path_Philips)

    files = []
    for file in files_Philips:
        dataFile = os.path.join(path_Philips, file)
        files.append(dataFile)
    for file in files_SIEMENS:
        dataFile = os.path.join(path_SIEMENS, file)
        files.append(dataFile)

    train_val_test_dic, label_dic = train_val_test_lable()
    for dataFile in files:
        id = getid(dataFile)
        name = getname(dataFile)
        label_character = label_dic[name]

        if id in fusion_corr.keys():
            a = len(fusion_corr[id])
            if a > 10:
                continue

        data = scio.loadmat(dataFile)
        corr = data["fun_con"]
        if np.isnan(corr).any():
            print(file, "包含nan")
            continue
        if not np.isfinite(corr).all():
            print(file, "包含inf")
        feature = corr.copy()
        fused_corr,nums = create_graph_fusion(corr, feature)
        if label_character == classa:
            if id in ADS.keys():
                ADS[id].append(fused_corr)
                ADS_feature[id].append(feature)
            else:
                tempgraph = []
                tempgraph.append(fused_corr)
                ADS[id] = tempgraph
                tempfeature = []
                tempfeature.append(feature)
                ADS_feature[id] = tempfeature
        if label_character == "CN":
            if id in CNS.keys():
                CNS[id].append(fused_corr)
                CNS_feature[id].append(feature)
            else:
                tempgraph = []
                tempgraph.append(fused_corr)
                CNS[id] = tempgraph
                tempfeature = []
                tempfeature.append(feature)
                CNS_feature[id] = tempfeature
        if label_character == "MCI":
            if id in MCIS.keys():
                MCIS[id].append(fused_corr)
                MCIS_feature[id].append(feature)
            else:
                tempgraph = []
                tempgraph.append(fused_corr)
                MCIS[id] = tempgraph
                tempfeature = []
                tempfeature.append(feature)
                MCIS_feature[id] = tempfeature
        if label_character == "EMCI":
            if id in EMCIS.keys():
                EMCIS[id].append(fused_corr)
                EMCIS_feature[id].append(feature)
            else:
                tempgraph = []
                tempgraph.append(fused_corr)
                EMCIS[id] = tempgraph
                tempfeature = []
                tempfeature.append(feature)
                EMCIS_feature[id] = tempfeature
        if label_character == "LMCI":
            if id in LMCIS.keys():
                LMCIS[id].append(fused_corr)
                LMCIS_feature[id].append(feature)
            else:
                tempgraph = []
                tempgraph.append(fused_corr)
                LMCIS[id] = tempgraph
                tempfeature = []
                tempfeature.append(feature)
                LMCIS_feature[id] = tempfeature
    if fusion:
        save_path = "data/imageid/resample/fusion/"
        torch.save(fusion_corr, save_path+"AD.pt")
    else:
        save_path = "data/imageid/resample/orgin/"

    if fusion:
        torch.save(ADS, save_path+"AD.pt")
        torch.save(ADS_feature, save_path+"AD_feature.pt")
        torch.save(CNS, save_path+"CN.pt")
        torch.save(CNS_feature, save_path+"CN_feature.pt")
        torch.save(MCIS, save_path+"MCI.pt")
        torch.save(MCIS_feature, save_path+"MCI_feature.pt")
        torch.save(EMCIS, save_path+"EMCI.pt")
        torch.save(EMCIS_feature, save_path+"EMCI_feature.pt")
        torch.save(LMCIS, save_path+"MCI.pt")
        torch.save(LMCIS_feature, save_path+"LMCI_feature.pt")


def get_corr(x,roi):  #select ROI
    temp_v = []
    ii = True
    for i in roi:
        if ii:
            temp_v = x[i, :]
            temp_v = np.transpose(temp_v)
            ii = False
        else:
            temp_v = np.vstack((temp_v, x[i, :]))
    temp_h = []
    ii = True
    for i in roi:
        if ii:
            temp_h = temp_v[:, i]
            temp_h = np.transpose(temp_h)
            ii = False
        else:
            temp_h = np.vstack((temp_h, temp_v[:, i]))
    x = temp_h
    return x

def gen_roi_corr_AD_CN(roi_num):
    AD = torch.load("data/imageid/resample/fusion/AD.pt")
    CN = torch.load("data/imageid/resample/fusion/CN.pt")
    AD_feature = torch.load("data/imageid/resample/fusion/AD_feature.pt")
    CN_feature = torch.load("data/imageid/resample/fusion/CN_feature.pt")
    ADS = {}
    CNS = {}
    ADFs={}
    CNFs={}
    roi = torch.load("data/select_roi/unresample_fusion_AD_CN_roi.pt")
    roi = roi[0:roi_num, 0]
    roi = np.sort(roi)
    for key in AD.keys():
        datas = AD[key]
        features = AD_feature[key]
        for i in range(len(datas)):
        # for data,feature in datas,features:
            if i >= 10:
                continue
            print(roi_num,key,i,"done")
            data = datas[i]
            feature = features[i]
            roi_corr = get_corr(data,roi)
            feature_corr = get_corr(feature,roi)
            if key in ADS.keys():
                ADS[key].append(roi_corr)
                ADFs[key].append(feature_corr)
            else:
                tempgraph = []
                tempgraph.append(roi_corr)
                ADS[key] = tempgraph
                tempfea = []
                tempfea.append(feature_corr)
                ADFs[key] = tempfea


    for key in CN.keys():
        datas = CN[key]
        features = CN_feature[key]
        # for data,feature in datas,features:
        for i in range(len(datas)):
        # for data,feature in datas,features:
            if i >= 10:
                continue
            print(roi_num,key,i,"done")
            data = datas[i]
            feature = features[i]
            feature_corr = get_corr(feature, roi)
            # x = np.reshape(data,(1,116*116))
            roi_corr = get_corr(data,roi)
            if key in CNS.keys():
                CNS[key].append(roi_corr)
                CNFs[key].append(feature_corr)
            else:
                tempgraph = []
                tempgraph.append(roi_corr)
                CNS[key] = tempgraph
                tempfea = []
                tempfea.append(feature_corr)
                CNFs[key] = tempfea


    AD_CN = [ADS,CNS]
    AD_CN_fea = [ADFs,CNFs]
    print("AD",len(ADS),"CN",len(CNS),"AD_CN",len(AD_CN))
    torch.save(AD_CN,"data/imageid/resample/fusion_roi/AD_CN "+str(roi_num)+"_corr.pt")
    torch.save(AD_CN_fea,"data/imageid/resample/fusion_roi/AD_CN "+str(roi_num)+"_feature.pt")


def gen_roi_corr_EMCI_LMCI(roi_num):

    EMCI = torch.load("data/imageid/resample/fusion/AD.pt")
    LMCI = torch.load("data/imageid/resample/fusion/CN.pt")
    EMCI_feature = torch.load("data/imageid/resample/fusion/AD_feature.pt")
    LMCI_feature = torch.load("data/imageid/resample/fusion/CN_feature.pt")

    EMCIS = {}
    LMCIS = {}
    EMCIFs={}
    LMCIFs={}
    roi = torch.load("data/select_roi/unresample_fusion_EMCI_LMCI_roi.pt")
    roi = roi[0:roi_num, 0]
    roi = np.sort(roi)
    for key in EMCI.keys():
        datas = EMCI[key]
        features = EMCI_feature[key]
        for i in range(len(datas)):
        # for data,feature in datas,features:
            if i >= 10:
                continue
            print(roi_num,key,i,"done")
            data = datas[i]
            feature = features[i]
            roi_corr = get_corr(data,roi)
            feature_corr = get_corr(feature,roi)
            if key in EMCIS.keys():
                EMCIS[key].append(roi_corr)
                EMCIFs[key].append(feature_corr)
            else:
                tempgraph = []
                tempgraph.append(roi_corr)
                EMCIS[key] = tempgraph
                tempfea = []
                tempfea.append(feature_corr)
                EMCIFs[key] = tempfea


    for key in LMCI.keys():
        datas = LMCI[key]
        features = LMCI_feature[key]
        # for data,feature in datas,features:
        for i in range(len(datas)):
        # for data,feature in datas,features:
            if i >= 10:
                continue
            print(roi_num,key,i,"done")
            data = datas[i]
            feature = features[i]
            feature_corr = get_corr(feature, roi)
            # x = np.reshape(data,(1,116*116))
            roi_corr = get_corr(data,roi)
            if key in LMCIS.keys():
                LMCIS[key].append(roi_corr)
                LMCIFs[key].append(feature_corr)
            else:
                tempgraph = []
                tempgraph.append(roi_corr)
                LMCIS[key] = tempgraph
                tempfea = []
                tempfea.append(feature_corr)
                LMCIFs[key] = tempfea


    EMCI_LMCI = [EMCIS,LMCIS]
    EMCI_LMCI_fea = [EMCIFs,LMCIFs]
    torch.save(EMCI_LMCI,"data/imageid/resample/fusion_roi/EMCI_LMCI "+str(roi_num)+"_corr.pt")
    torch.save(EMCI_LMCI_fea,"data/imageid/resample/fusion_roi/EMCI_LMCI "+str(roi_num)+"_feature.pt")
def gen_roi_corr_MCI_CN(roi_num):
    MCI = torch.load("data/imageid/resample/fusion/MCI.pt")
    EMCI = torch.load("data/imageid/resample/fusion/EMCI.pt")
    LMCI = torch.load("data/imageid/resample/fusion/LMCI.pt")
    CN = torch.load("data/imageid/resample/fusion/CN.pt")
    MCI_feature = torch.load("data/imageid/resample/fusion/MCI_feature.pt")
    EMCI_feature = torch.load("data/imageid/resample/fusion/EMCI_feature.pt")
    LMCI_feature = torch.load("data/imageid/resample/fusion/LMCI_feature.pt")
    CN_feature = torch.load("data/imageid/resample/fusion/CN_feature.pt")
    AMCIS = {}
    CNS = {}
    AMCIFs={}
    CNFs={}
    roi = torch.load("data/select_roi/unresample_fusion_MCI_CN_roi.pt")
    roi = roi[0:roi_num, 0]
    roi = np.sort(roi)
    for key in MCI.keys():
        datas = MCI[key]
        features = MCI_feature[key]
        for i in range(len(datas)):
            if i >= 10:
                continue
            print(roi_num,key,i,"done")
            data = datas[i]
            feature = features[i]
            roi_corr = get_corr(data,roi)
            feature_corr = get_corr(feature,roi)
            if key in AMCIS.keys():
                AMCIS[key].append(roi_corr)
                AMCIFs[key].append(feature_corr)
            else:
                tempgraph = []
                tempgraph.append(roi_corr)
                AMCIS[key] = tempgraph
                tempfea = []
                tempfea.append(feature_corr)
                AMCIFs[key] = tempfea
    for key in EMCI.keys():
        datas = EMCI[key]
        features = EMCI_feature[key]
        for i in range(len(datas)):
            if i >= 10:
                continue
            print(roi_num,key,i,"done")
            data = datas[i]
            feature = features[i]
            roi_corr = get_corr(data,roi)
            feature_corr = get_corr(feature,roi)
            if key in AMCIS.keys():
                AMCIS[key].append(roi_corr)
                AMCIFs[key].append(feature_corr)
            else:
                tempgraph = []
                tempgraph.append(roi_corr)
                AMCIS[key] = tempgraph
                tempfea = []
                tempfea.append(feature_corr)
                AMCIFs[key] = tempfea
    for key in LMCI.keys():
        datas = LMCI[key]
        features = LMCI_feature[key]
        for i in range(len(datas)):
            if i >= 10:
                continue
            print(roi_num,key, i, "done")
            data = datas[i]
            feature = features[i]
            roi_corr = get_corr(data,roi)
            feature_corr = get_corr(feature,roi)
            if key in AMCIS.keys():
                AMCIS[key].append(roi_corr)
                AMCIFs[key].append(feature_corr)
            else:
                tempgraph = []
                tempgraph.append(roi_corr)
                AMCIS[key] = tempgraph
                tempfea = []
                tempfea.append(feature_corr)
                AMCIFs[key] = tempfea

    for key in CN.keys():
        datas = CN[key]
        features = CN_feature[key]
        for i in range(len(datas)):
            if i >= 10:
                continue
            print(roi_num,key,i,"done")
            data = datas[i]
            feature = features[i]
            feature_corr = get_corr(feature, roi)
            roi_corr = get_corr(data,roi)
            if key in CNS.keys():
                CNS[key].append(roi_corr)
                CNFs[key].append(feature_corr)
            else:
                tempgraph = []
                tempgraph.append(roi_corr)
                CNS[key] = tempgraph
                tempfea = []
                tempfea.append(feature_corr)
                CNFs[key] = tempfea
    MCI_CN = [AMCIS,CNS]
    MCI_CN_fea = [AMCIFs,CNFs]
    torch.save(MCI_CN,"data/imageid/resample/fusion_roi/MCI_CN "+str(roi_num)+"_corr.pt")
    torch.save(MCI_CN_fea,"data/imageid/resample/fusion_roi/MCI_CN "+str(roi_num)+"_feature.pt")

def create_graph(corr, feature):
    u = []
    v = []
    w = []
    corr = 0.5*(corr+corr.T)
    for i in range(corr.shape[0]):
        corr[i][i] = 1
    sum = 0
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):

            if corr[i][j] > 0:
                # print(corr[i][j])
                u.append(i)
                v.append(j)
                w.append([corr[i][j]])
                sum = sum + 1
    x = torch.tensor(feature, dtype=torch.float32)
    edge_index = np.array([u,v])
    edge_index = torch.tensor(edge_index,dtype=torch.int64)
    edge_attr = torch.tensor(np.array(w), dtype=torch.float32)
    label = [0]
    label = torch.tensor(label, dtype=torch.int64)
    pos = np.eye(116)
    pos = torch.tensor(pos, dtype=torch.float32)
    data = Data(edge_index=edge_index, x=x, edge_attr=edge_attr, y=label, pos=pos)
    return data, sum

def kfold_dataset(class1,class2,roi_num,orgin_fusion):
    AD,CN  = torch.load("data/imageid/resample/"+orgin_fusion+"/"+class1+"_"+class2+" "+str(roi_num)+"_feature.pt")
    a,b = torch.load("data/imageid/resample/"+orgin_fusion+"/"+class1+"_"+class2+" "+str(roi_num)+"_corr.pt")
    print(roi_num,class1,len(a),class2,len(b))
    dataset = {}
    print("delete",roi_num,class1,len(a),class2,len(b))
    for key in a.keys():
        datas = a[key]
        features = AD[key]
        pa_datas = []
        for i in range(len(datas)):
            data = datas[i]
            feature = features[i]
            if np.isnan(data).any():
                print(i, "data包含nan")
                continue
            if not np.isfinite(feature).all():
                print(i, "feature包含inf")
                continue
            label = torch.tensor(0, dtype=torch.int64)
            graph,nums = create_graph(data, feature)
            graph.y = label
            pa_datas.append(graph)
        dataset[key] = pa_datas
    for key in b.keys():
        datas = b[key]
        features = CN[key]
        pa_datas = []
        # print(roi_num,class1,key,len(datas))
        for i in range(len(datas)):
            data = datas[i]
            feature = features[i]
            if np.isnan(data).any():
                print(i, "data包含nan")
                continue
            if not np.isfinite(feature).all():
                print(i, "feature包含inf")
                continue
            label = torch.tensor(1, dtype=torch.int64)
            graph, nums = create_graph(data, feature)
            graph.y = label
            pa_datas.append(graph)
        dataset[key] = pa_datas

    torch.save(dataset,"data/imageid/resample/fusion_roi/"+class1 + "_" + class2 +" roi "+str(roi_num)+ ".pt")


if __name__ == "__main__":
    bo = True
    gen_graph(fusion=bo)
    num_rois = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for roi in num_rois:
       gen_roi_corr_AD_CN(roi)
       gen_roi_corr_EMCI_LMCI(roi)
       gen_roi_corr_MCI_CN(roi)
    for roi in num_rois:
        orgin_fusion = "fusion_roi"
        kfold_dataset("MCI","CN",roi,orgin_fusion)
        kfold_dataset("AD","CN",roi,orgin_fusion)
        kfold_dataset("EMCI","LMCI",roi,orgin_fusion)


