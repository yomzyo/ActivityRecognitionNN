import pandas as pd


class data_org():
    def __init__(self, dataset='ntu_rgbd_dataset'):
        self.dataset = dataset
        self.partition = {}
        self._generate_partition()
        self.labels = {}
        self._generate_labels()
        self.label_ids = {}
        self._generate_label_ids()

    def _generate_partition(self):
        partition_df = pd.read_csv(
            "Dataset_Separation/" + self.dataset + "/dataset_separations.csv")
        for data in partition_df:
            self.partition[data] = []
            for j, video in enumerate(partition_df[data]):
                if not(isinstance(video, float)):
                    self.partition[data].append(partition_df[data][j])

    def _generate_labels(self):
        labels_df = pd.read_csv(
            "Dataset_Separation/" + self.dataset + "/labeled_list.csv")
        self.labels = labels_df.set_index('video').T.to_dict('records')[0]

    def _generate_label_ids(self):
        label_ids_df = pd.read_csv("Dataset_Separation/" + self.dataset
                                   + "/labeled_list_decodings.csv")
        self.label_ids =\
            label_ids_df.set_index('action').T.to_dict('records')[0]
