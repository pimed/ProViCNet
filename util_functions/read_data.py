import os
import SimpleITK as sitk
import json


def prep_for_training():
    with open("/mnt/local_data/stanford/data_split/splits_final.json") as json_file:
        stanford_fold_data = json.load(json_file)
    stanford_train_ids = stanford_fold_data[0]['train']
    stanford_val_ids = stanford_fold_data[0]['val']

    with open("/mnt/local_data/picai/data_split/splits_all.json") as json_file:
        picai_fold_data = json.load(json_file)
    picai_train_case_ids = picai_fold_data[0]['inner_train']
    picai_val_case_ids = picai_fold_data[0]['inner_val']

    with open('/mnt/local_data/ucla/data_split/split_all_names_new.json') as json_file:
        ucla_split_data = json.load(json_file)
    ucla_train_case_ids = ucla_split_data['train']
    ucla_val_case_ids = ucla_split_data['val']

    train = stanford_train_ids + picai_train_case_ids + ucla_train_case_ids
    val = stanford_val_ids + picai_val_case_ids + ucla_val_case_ids
    return train, val

def prep_for_testing():
    with open('/mnt/pimed/data_processed/Stanford_Prostate_Processed/Stanford_Bx_preprocessed/data_split/Stanford_multimodal_bx_test_data.json') as json_file:
        stanford_test_json = json.load(json_file)

    stanford_test_list = stanford_test_json['bx_test']
    stanford_test_ids = [test['Anon_ID'] for test in stanford_test_list]

    with open('/mnt/pimed/results2/Challenges/2022_PICAI/5Fold_CV/PICAI_test_names.json') as json_file:
        picai_test_ids = json.load(json_file)
    
    with open('/mnt/local_data/ucla/data_split/split_all_names_new.json') as json_file:
        ucla_split_data = json.load(json_file)
    ucla_test_case_ids = ucla_split_data['test']
    
    test = stanford_test_ids + picai_test_ids + ucla_test_case_ids
    return test
