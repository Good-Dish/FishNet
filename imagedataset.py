from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import os

class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, label_column="Family", split='Train'):
        self.data_frame = pd.read_csv(csv_file)
        if label_column == 'Troph':
            # remove items with Troph are empty
            bool_series = pd.isnull(self.data_frame["Troph"])
            self.data_frame = self.data_frame[~bool_series]

            mu, var = self.data_frame['Troph'].mean(), self.data_frame['Troph'].std()
            print('Troph mean/variance', mu, var)
            self.data_frame['Troph'] = (self.data_frame['Troph'] - mu) / var  # normalize Trophic values
        elif label_column == 'MultiCls':
            # remove items whose attibutes are empty
            self.all_columns = ['FeedingPath', 'Tropical', 'Temperate', 'Subtropical', 'Boreal', 'Polar', 'freshwater',
                                'saltwater', 'brackish']
            bool_series = np.ones(len(self.data_frame), )
            for col in self.all_columns:
                bool_col = ~pd.isnull(self.data_frame[col])
            bool_series = bool_series * bool_col
            self.data_frame = self.data_frame[bool_series.astype(np.bool)]

        # select the ratio to train
        self.root_dir = root_dir
        self.transform = transform
        self.label_col = label_column
        self.image_col = "image"
        self.folder_col = "Folder"
        print('csv file: {} has {} item.'.format(csv_file, len(self.data_frame)))

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):

        img_name = self.data_frame.iloc[idx][self.image_col]
        img_name = img_name.split('/')[-1]
        folder = self.data_frame.iloc[idx][self.folder_col]
        # img_path = os.path.join(folder, img_name)
        img_path = img_name
        image = Image.open(self.root_dir + img_path)

        csv_file_type = 'anns/train_full_meta_new.csv'  # read all classes information
        meta_df = pd.read_csv(csv_file_type)

        if self.label_col == "Family":
            cls_name = self.data_frame.iloc[idx][self.label_col]
            label = meta_df.loc[meta_df['Family'] == cls_name]['Family_cls'].values[0]
        elif self.label_col == "Order":
            cls_name = self.data_frame.iloc[idx][self.label_col]
            if '/' in cls_name:
                cls_name = cls_name.split('/')[0]
            label = meta_df.loc[meta_df['Order_new'] == cls_name]['Order_cls'].values[0]
        elif self.label_col == 'Troph':
            label = self.data_frame.iloc[idx][self.label_col]
        #             label = all_classes.index(cls_name)
        elif self.label_col == 'MultiCls':
            label = []
            for col in self.all_columns:
                val = self.data_frame.iloc[idx][col]
                if col == 'FeedingPath':
                    if val == 'pelagic':
                        val = 1
                    elif val == 'benthic':
                        val = 0
                label.append(val)
            label = np.asarray(label)
        if self.transform:
            image = self.transform(image)
        return (image, label, self.root_dir + img_path)