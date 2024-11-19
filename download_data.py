from modelscope.msdatasets import MsDataset

# 指定下载路径
ds = MsDataset.load('ddzhu123/seq-monkey', cache_dir='MateConv/dataset')
