import os
import glob
import csv

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.image as mpimg
import pandas as pd


class MappingDataset(Dataset):
    dataset_folder = 'dataset'
    category_folder = {
        'train': 'train',
        'valid': 'valid',
        'test': 'test',
    }
    img_extension = 'jpeg'
    label_extension = 'xls'

    map_file_name = 'map.csv'

    def __init__(self, root, mode='train', generate_map=False):
        self.root = os.path.abspath(root)
        self.mode = mode

        if generate_map:
            self.generate_map_file()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use generate_map=True to create a list of address')

        # TODO: if there is no below folder, need to make a dir.
        images_folder = os.path.join(self.root, self.dataset_folder, self.category_folder[self.mode])
        map_file_path = os.path.join(images_folder, self.map_file_name)

        self.image_target_list = []
        # TODO: test whether successfully parse the csv file
        csv_reader = csv.reader(open(map_file_path))
        for row in csv_reader:
            self.image_target_list.append((row[0], row[1]))
        # print(len(self))

    def __getitem__(self, index):
        image_label_pair = self.image_target_list[index]
        # TODO: change into your process of loading sample
        img = mpimg.imread(image_label_pair[0])
        return img, image_label_pair[1]

    def __len__(self):
        return len(self.image_target_list)

    def _check_exists(self):
        return os.path.exists(
            os.path.join(self.root, self.dataset_folder, self.category_folder[self.mode], self.map_file_name))

    def generate_map_file(self):
        images_folder = os.path.join(self.root, self.dataset_folder, self.category_folder[self.mode])
        map_file_path = os.path.join(images_folder, self.map_file_name)
        label_path = glob.glob(f'{images_folder}/*.{self.label_extension}')[0]

        labels = pd.read_excel(label_path)
        labels['path'] = ''

        images_path = f'{images_folder}/*.{self.img_extension}'
        images = glob.glob(images_path)
        for idx, path in enumerate(images):
            file_name = path.split(os.sep)[-1].split('.')[0]
            labels.loc[(labels['images'] == file_name), "path"] = path

        labels = labels.loc[labels["path"] != '']
        labels.to_csv(path_or_buf=map_file_path, columns=['path', 'labels'], header=False, index=False)


if __name__ == '__main__':
    # dataset = MappingDataset(os.path.join('.'), mode='train', generate_map=False)
    dataset = MappingDataset(os.path.join('.'), mode='train', generate_map=True)
    # dataset = MappingDataset(os.path.join('.'), mode='valid', generate_map=True)
    # dataset = MappingDataset(os.path.join('.'), mode='test', generate_map=True)
    train_loader = DataLoader(dataset, batch_size=10, shuffle=False)

    for idx, (img, label) in enumerate(train_loader):
        print(img.shape)
        print(label)
