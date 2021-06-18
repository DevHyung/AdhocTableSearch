import argparse

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from dataset import QueryTableDataset, query_table_collate_fn
from table_bert import TableBertModel


class QueryTableDataModule(pl.LightningDataModule):
    def __init__(self, params):
        print(">>> QueryTableDataModule init")
        super().__init__()
        self.data_dir = params.data_dir

        table_model = TableBertModel.from_pretrained(params.tabert_path)
        # self.query_tokenizer = BertTokenizer.from_pretrained(params.bert_path)
        self.table_tokenizer = table_model.tokenizer
        self.query_tokenizer = table_model.tokenizer

        self.train_batch_size = params.train_batch_size
        self.valid_batch_size = params.valid_batch_size
        
        self.min_row = params.min_row

    def prepare_data(self):
        # Download, tokenize, etc
        # Write to disk or that need to be done only from a single GPU in distributed settings
        QueryTableDataset(data_dir=self.data_dir, data_type='train',
                          query_tokenizer=self.query_tokenizer,
                          table_tokenizer=self.table_tokenizer,
                          min_row=self.min_row,
                          prepare=True)


    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            table_full = QueryTableDataset(data_dir=self.data_dir, data_type='train', min_row=self.min_row)
            self.train, self.valid = random_split(table_full, [len(table_full) - int(len(table_full) * 0.1),
                                                               int(len(table_full) * 0.1)])

    def train_dataloader(self):
        print(">>> QueryTableDataModule train_dataloader")
        """
        TODO2: batch 끝나고 negative 평가 다시 
        1. setup()함수부분의 table_full부분을 다시 만들어야함
        
        2. on_train_epoch_end가 끝나면 data/bench/${i} 의 negative_order파일을읽어
           오더링된 순서를 기준으로 재정렬하면될거같음 
           
        3. 여기서 model에서 씌이는 model을 받아야함 
        
        4. <Q, [sling-tables]> 현재는 전체의 negative-table을 전체를 쓰는게 아닌
           Top-n개든, threshold이상의 score를 지닌 테이블들을 쓰고
           
           <Q, [postive slicing-tables], [negative slicing-tables]>를 쓰는
           모델에서는 negative table부분을 바꿔줘야함 
        """
        
        return DataLoader(self.train,
                          batch_size=self.train_batch_size,
                          shuffle=True,
                          collate_fn=query_table_collate_fn,
                          num_workers=4,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.valid,
                          batch_size=self.valid_batch_size,
                          collate_fn=query_table_collate_fn,
                          num_workers=4,
                          )


if __name__ == "__main__":
    args = argparse.Namespace()
    args.data_dir = 'data'
    args.tabert_path = 'model/tabert_base_k3/model.bin'
    args.train_batch_size = 2
    args.valid_batch_size = 2

    data_module = QueryTableDataModule(args)
    data_module.prepare_data()
    data_module.setup('fit')
    for batch in data_module.train_dataloader():
        print(*batch)

