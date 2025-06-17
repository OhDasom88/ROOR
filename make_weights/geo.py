import torch


def expand(ori_path, new_path):
    st_dict = torch.load(ori_path)
    st_dict['net.geolayoutlm_model.text_encoder.embeddings.position_ids'] = \
        torch.cat(
            (st_dict['net.geolayoutlm_model.text_encoder.embeddings.position_ids'],
             st_dict['net.geolayoutlm_model.text_encoder.embeddings.position_ids']), dim=1)
    st_dict['net.geolayoutlm_model.text_encoder.embeddings.position_embeddings.weight'] = \
        torch.cat(
            (st_dict['net.geolayoutlm_model.text_encoder.embeddings.position_embeddings.weight'],
             st_dict['net.geolayoutlm_model.text_encoder.embeddings.position_embeddings.weight']), dim=0)
    st_dict['net.geolayoutlm_model.text_encoder.embeddings.line_rank_embeddings.weight'] = \
        torch.cat(
            (st_dict['net.geolayoutlm_model.text_encoder.embeddings.line_rank_embeddings.weight'],
             st_dict['net.geolayoutlm_model.text_encoder.embeddings.line_rank_embeddings.weight']), dim=0)
    torch.save(st_dict, new_path)


def expand_(ori_path, new_path):
    st_dict = torch.load(ori_path)
    st_dict['geolayoutlm_model.text_encoder.embeddings.position_ids'] = \
        torch.cat(
            (st_dict['geolayoutlm_model.text_encoder.embeddings.position_ids'],
             st_dict['geolayoutlm_model.text_encoder.embeddings.position_ids']), dim=1)
    st_dict['geolayoutlm_model.text_encoder.embeddings.position_embeddings.weight'] = \
        torch.cat(
            (st_dict['geolayoutlm_model.text_encoder.embeddings.position_embeddings.weight'],
             st_dict['geolayoutlm_model.text_encoder.embeddings.position_embeddings.weight']), dim=0)
    st_dict['geolayoutlm_model.text_encoder.embeddings.line_rank_embeddings.weight'] = \
        torch.cat(
            (st_dict['geolayoutlm_model.text_encoder.embeddings.line_rank_embeddings.weight'],
             st_dict['geolayoutlm_model.text_encoder.embeddings.line_rank_embeddings.weight']), dim=0)
    torch.save(st_dict, new_path)


# expand('/path/to/epoch.105-f1_labeling.0.9232.pt', '/path/to/labeling_1024.pt')
# expand('/path/to/epoch.182-f1_linking.0.8923.pt', '/path/to/linking_1024.pt')
# expand_('/path/to/geolayoutlm_large_pretrain.pt', '/path/to/pretrain_1024.pt')


from tqdm import tqdm
project_path = '/home/dasom/ROOR'
base_path = f'{project_path}/make_weights/geolayoutlm'
pbar = tqdm(total=3)
pbar.set_description("Expanding labeling")
expand(f'{base_path}/epoch.105-f1_labeling.0.9232.pt', f'{base_path}/labeling_1024.pt')
pbar.update(1)
pbar.set_description("Expanding linking")
expand(f'{base_path}/epoch.182-f1_linking.0.8923.pt', f'{base_path}/linking_1024.pt')
pbar.update(1)
pbar.set_description("Expanding pretrain")
expand_(f'{base_path}/geolayoutlm_large_pretrain.pt', f'{base_path}/pretrain_1024.pt')
pbar.update(1)
pbar.close()
