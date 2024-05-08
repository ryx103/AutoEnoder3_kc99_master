import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset
import pickle as pl

import codecs
import os
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch

from imblearn.over_sampling import SMOTE

# GET DATA
def Get_Data(args):
    dataset = args.dataset
    if dataset == 'kdd':
        data_dir='./data/kdd_cup.npz'
        train = KDDCupData(data_dir, 'train')
        dataloader_train = DataLoader(train, batch_size=args.batch_size, 
                                        shuffle=True, num_workers=0)
    
        test = KDDCupData(data_dir, 'test')
        dataloader_test = DataLoader(test, batch_size=args.batch_size, 
                                        shuffle=False, num_workers=0)
        print("data pass")
        print("训练集数据维度：")
        for batch in dataloader_train:
            data, labels = batch
            print("数据形状:", data.shape)
            print("标签形状:", labels.shape)
            break  # 只打印第一个batch的数据维度
        return dataloader_train, dataloader_test 
    elif dataset == 'cicids':
        data_loader = CICIDS2017DataLoader(args)
        train_loader, test_loader = data_loader.load_and_preprocess_data()
        return train_loader, test_loader
    
        # print(f"cicids")
        print("Loading CICIDS 2017 dataset...")
        #TODO

        # 定义训练和测试数据文件的路径
        train_data_path = './data/MachineLearningCVE/ProcessedDataset/train_05.csv'
        test_data_path = './data/MachineLearningCVE/ProcessedDataset/test_05.csv'

        # 读取 CSV 文件
        train_df = pd.read_csv(train_data_path)
        test_df = pd.read_csv(test_data_path)
        print("Reading of csv file has been completed")

        # ratio = 0.1
        # train_df = train_df.sample(frac=ratio)
        # test_df = test_df.sample(frac=ratio)

        print("Start separating features and labels...")
        # 分离特征和标签
        X_train = train_df.drop('Label', axis=1).values
        y_train = train_df['Label'].values
        X_test = test_df.drop('Label', axis=1).values
        y_test = test_df['Label'].values

        # 将 numpy 数组转换为 tensor
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)

        print("X_train shape:", X_train.shape)
        print("X_test shape:", X_test.shape)

        # 创建 TensorDataset
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)


        # 创建 DataLoader
        dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        dataloader_test = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        print("Data loaders prepared for CICIDS 2017.")
        for batch in dataloader_train:
            data, labels = batch
            print("数据形状:", data.shape)
            print("标签形状:", labels.shape)
            break  # 只打印第一个batch的数据维度
        return dataloader_train, dataloader_test


#KDD_CUP_1999

class KDDCupData:
    def __init__(self, data_dir, mode):
        """Loading the data for train and test."""
        data = np.load(data_dir, allow_pickle=True)

        labels = data["kdd"][:,-1]
        features = data["kdd"][:,:-1]
        #In this case, "atack" has been treated as normal data as is mentioned in the paper
        normal_data = features[labels==0] 
        normal_labels = labels[labels==0]

        n_train = int(normal_data.shape[0]*0.5)
        ixs = np.arange(normal_data.shape[0])
        np.random.shuffle(ixs)
        normal_data_test = normal_data[ixs[n_train:]]
        normal_labels_test = normal_labels[ixs[n_train:]]

        if mode == 'train':
            self.x = normal_data[ixs[:n_train]]
            self.y = normal_labels[ixs[:n_train]]
        elif mode == 'test':
            anomalous_data = features[labels==1]
            anomalous_labels = labels[labels==1]
            self.x = np.concatenate((anomalous_data, normal_data_test), axis=0)
            self.y = np.concatenate((anomalous_labels, normal_labels_test), axis=0)

    def __len__(self):
        """Number of images in the object dataset."""
        return self.x.shape[0]

    def __getitem__(self, index):
        """Return a sample from the dataset."""
        return np.float32(self.x[index]), np.float32(self.y[index])


def get_KDDCup99(args, data_dir='./data/kdd_cup.npz'):
    """Returning train and test dataloaders."""
    train = KDDCupData(data_dir, 'train')
    dataloader_train = DataLoader(train, batch_size=args.batch_size, 
                              shuffle=True, num_workers=0)
    
    test = KDDCupData(data_dir, 'test')
    dataloader_test = DataLoader(test, batch_size=args.batch_size, 
                              shuffle=False, num_workers=0)
    return dataloader_train, dataloader_test

# CICIDS_2017
class CICIDS2017DataLoader:
    def __init__(self, args, sample_fraction=0.1):
        self.args = args
        self.data_path = './data/MachineLearningCVE/MachineLearningCVE.csv'
        self.sample_fraction = sample_fraction
        self.total_size = 40000

    def load_and_preprocess_data(self):
        print("Loading CICIDS 2017 dataset...")
        df = pd.read_csv(self.data_path)
        print("Initial data loading complete.")

        # Replace infinite values with maximum non-infinite values
        for col in ['Flow Packets/s', 'Flow Bytes/s']:
            max_value = df.loc[df[col] != np.inf, col].max()
            df[col].replace(np.inf, max_value, inplace=True)

        # Drop any rows with NaN values
        df.dropna(inplace=True)

        # 随机采样数据以减小数据集大小
        # if len(df) > self.total_size:
        #     df = df.sample(n=self.total_size, random_state=42)

        # Downsampling of the benign instances
        benign_sample = df[df['Label'] == 'BENIGN'].sample(frac=self.sample_fraction, random_state=42)
        attack_sample = df[df['Label'] != 'BENIGN']
        df = pd.concat([attack_sample, benign_sample])

        # 随机采样数据以减小数据集大小
        if len(df) > self.total_size:
            df = df.sample(n=self.total_size, random_state=42)

        # Convert categorical features into dummy variables
        df = pd.get_dummies(df, columns=[col for col in df.columns if df[col].dtype == 'object' and col != 'Label'])

        # Encode labels
        label_encoder = LabelEncoder()
        df['Label'] = label_encoder.fit_transform(df['Label'])

        # Split data into features and labels
        X = df.drop('Label', axis=1)
        y = df['Label']


        # # Handling class imbalance with SMOTE
        # smote = SMOTE(sampling_strategy='auto', random_state=42)
        # X = df.drop('Label', axis=1)
        # y = df['Label']
        # X_res, y_res = smote.fit_resample(X, y)
        # df_resampled = pd.DataFrame(X_res, columns=X.columns)
        # df_resampled['Label'] = y_res

        # Split data into training and testing sets
        # X_train, X_test, y_train, y_test = train_test_split(df_resampled.drop('Label', axis=1), df_resampled['Label'], test_size=0.4, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Convert data to tensors
        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.long)
        X_test = torch.tensor(X_test.values, dtype=torch.float32)
        y_test = torch.tensor(y_test.values, dtype=torch.long)

        # Create DataLoaders
        dataloader_train = DataLoader(TensorDataset(X_train, y_train), batch_size=self.args.batch_size, shuffle=True, num_workers=4)
        dataloader_test = DataLoader(TensorDataset(X_test, y_test), batch_size=self.args.batch_size, shuffle=False, num_workers=4)
        
        print("Data loaders prepared for training and testing.")
        return dataloader_train, dataloader_test

class CICIDS2017DataLoader2:
    def __init__(self, args):
        self.args = args
        self.data_path = './data/MachineLearningCVE/MachineLearningCVE.csv'

    def load_and_preprocess_data(self):
        print("Loading CICIDS 2017 dataset...")
        df = pd.read_csv(self.data_path)
        print("Initial data loading complete.")

        # Replace infinite values with maximum non-infinite values
        for col in ['Flow Packets/s', 'Flow Bytes/s']:
            max_value = df.loc[df[col] != np.inf, col].max()
            # df[col].replace(np.inf, max_value, inplace=True)
            df.replace({col: {np.inf: max_value}}, inplace=True)

        # Drop any rows with NaN values
        df.dropna(inplace=True)

        # Randomly sample 10% of the benign instances
        benign_sample = df[df['Label'] == 'BENIGN'].sample(frac=0.1, random_state=42)
        df = pd.concat([df[df['Label'] != 'BENIGN'], benign_sample])

        # Convert categorical features into dummy variables
        df = pd.get_dummies(df, columns=[col for col in df.columns if df[col].dtype == 'object' and col != 'Label'])

        # Encode labels
        label_encoder = LabelEncoder()
        df['Label'] = label_encoder.fit_transform(df['Label'])

        # Handling class imbalance with SMOTE
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X = df.drop('Label', axis=1)
        y = df['Label']
        X_res, y_res = smote.fit_resample(X, y)
        df_resampled = pd.DataFrame(X_res, columns=X.columns)
        df_resampled['Label'] = y_res

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(df_resampled.drop('Label', axis=1), df_resampled['Label'], test_size=0.2, random_state=42)
        
        # Convert data to tensors
        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.long)
        X_test = torch.tensor(X_test.values, dtype=torch.float32)
        y_test = torch.tensor(y_test.values, dtype=torch.long)

        # Create DataLoaders
        dataloader_train = DataLoader(TensorDataset(X_train, y_train), batch_size=self.args.batch_size, shuffle=True, num_workers=0)
        dataloader_test = DataLoader(TensorDataset(X_test, y_test), batch_size=self.args.batch_size, shuffle=False, num_workers=0)
        
        print("Data loaders prepared for training and testing.")
        return dataloader_train, dataloader_test














class CICIDS2017DataLoader1:
    def __init__(self, args):
        self.args = args
        self.train_data_path = './data/MachineLearningCVE/ProcessedDataset/train_10.csv'
        self.test_data_path = './data/MachineLearningCVE/ProcessedDataset/test_10.csv'

    def load_and_preprocess_data(self):
        print("Loading CICIDS 2017 dataset...")

        # 读取 CSV 文件
        train_df = pd.read_csv(self.train_data_path)
        test_df = pd.read_csv(self.test_data_path)
        print("Reading of csv file has been completed")

        # 处理类别不平衡
        train_df = self.handle_imbalance(train_df)

        print("Start separating features and labels...")
        # 分离特征和标签
        X_train = train_df.drop('Label', axis=1).values
        y_train = train_df['Label'].values
        X_test = test_df.drop('Label', axis=1).values
        y_test = test_df['Label'].values

        # 将 numpy 数组转换为 tensor
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)

        # 创建 DataLoader
        dataloader_train = DataLoader(TensorDataset(X_train, y_train), batch_size=self.args.batch_size, shuffle=True, num_workers=0)
        dataloader_test = DataLoader(TensorDataset(X_test, y_test), batch_size=self.args.batch_size, shuffle=False, num_workers=0)

        print("Data loaders prepared for CICIDS 2017.")
        return dataloader_train, dataloader_test

    def handle_imbalance(self, df):
        print("Handling class imbalance with SMOTE...")
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X = df.drop('Label', axis=1)
        y = df['Label']
        X_res, y_res = smote.fit_resample(X, y)
        return pd.concat([X_res, y_res], axis=1)



def _to_utf8(filename: str, encoding="latin1", blocksize=1048576):
    tmpfilename = filename + ".tmp"
    with codecs.open(filename, "r", encoding) as source:
        with codecs.open(tmpfilename, "w", "utf-8") as target:
            while True:
                contents = source.read(blocksize)
                if not contents:
                    break
                target.write(contents)
    os.remove(filename)
    # replace the original file
    os.rename(tmpfilename, filename)

def _renaming_class_label(df: pd.DataFrame):
    labels = {"Web Attack \x96 Brute Force": "Web Attack-Brute Force",
              "Web Attack \x96 XSS": "Web Attack-XSS",
              "Web Attack \x96 Sql Injection": "Web Attack-Sql Injection"}

    for old_label, new_label in labels.items():
        df.Label.replace(old_label, new_label, inplace=True)

# Renaming labels

def _renaming_class_label(df: pd.DataFrame):
    labels = {"Web Attack � Brute Force": "Web Attack-Brute Force",
              "Web Attack � XSS": "Web Attack-XSS",
              "Web Attack � Sql Injection": "Web Attack-Sql Injection"}

    for old_label, new_label in labels.items():
        # df.Label.replace(old_label, new_label, inplace=True)
        df['Label'] = df['Label'].replace(old_label, new_label)

