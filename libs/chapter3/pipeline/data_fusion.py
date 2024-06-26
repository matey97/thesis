import pandas as pd


def fuse_data(sp_data, sw_data):
    sp_data = sp_data.set_index(['timestamp'])
    sp_data.index = pd.to_datetime(sp_data.index)
    sw_data = sw_data.set_index(['timestamp'])   
    sw_data.index = pd.to_datetime(sw_data.index)

    sp_start = sp_data.index.min()
    sw_start = sw_data.index.min()
    global_start = max(sp_start, sw_start)
    
    sp_end = sp_data.index.max()
    sw_end = sw_data.index.max()
    global_end = max(sp_end, sw_end)
    
    sp_data = sp_data.loc[global_start:global_end]
    sw_data = sw_data.loc[global_start:global_end]
    
    sp_samples = sp_data.resample('1S')['label'].count()
    sw_samples = sw_data.resample('1S')['label'].count()
    sp_samples = sp_samples.rename('sp')
    sw_samples = sw_samples.rename('sw')
    n_samples = pd.merge(sp_samples, sw_samples, how='outer', left_index=True, right_index=True)
    n_samples = n_samples.fillna(0)
    
    sp_cursor = 0
    sw_cursor = 0

    fused = []

    for index, row in n_samples.iterrows():
        n_sp = int(row['sp'])
        n_sw = int(row['sw'])

        take = min(n_sp, n_sw)

        sp_part = sp_data.iloc[sp_cursor:sp_cursor+take].reset_index()
        sw_part = sw_data.iloc[sw_cursor:sw_cursor+take].reset_index()
        fused.append(sp_part.join(sw_part, lsuffix='_sp', rsuffix='_sw'))

        sp_cursor += n_sp
        sw_cursor += n_sw

    data_fused = pd.concat(fused, axis=0, ignore_index=True)
    data_fused = data_fused[data_fused.label_sp == data_fused.label_sw]
    data_fused['label'] = data_fused['label_sp']
    data_fused = data_fused.drop(['label_sp', 'label_sw'], axis=1)
    return data_fused