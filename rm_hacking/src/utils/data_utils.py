from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd
import os
import jsonlines

from torch import distributed as dist


def save_parquet(data, save_dir, split, num_shards=1):
    # data: list of dict
    df = pd.DataFrame(data)
    ds = Dataset.from_pandas(df)
    #ds = Dataset.from_list(data)
    for shard_idx in range(num_shards):
        shard = ds.shard(index=shard_idx, num_shards=num_shards)
        shard.to_parquet(f"{save_dir}/{split}-{shard_idx:05d}.parquet")
    
    # write readme
    readme_str = f"---\nconfigs:\n- config_name: default\n  data_files:\n  - split: {split}\n    path: {split}*\n---"
    with open(os.path.join(save_dir, "README.md"), "w") as f:
        f.write(readme_str)
    

def save_json(data, save_dir, split):
    os.makedirs(save_dir, exist_ok=True)
    with jsonlines.open(os.path.join(save_dir, split + ".json"), mode="w") as writer:
        writer.write_all(data)
    
    # write readme
    readme_str = f"---\nconfigs:\n- config_name: default\n  data_files:\n  - split: {split}\n    path: {split}*\n---"
    with open(os.path.join(save_dir, "README.md"), "w") as f:
        f.write(readme_str)


def gather_data_distributed(data, save_dir, rank, total_num=None, del_cache=False):
    # write to cache
    with jsonlines.open(save_dir + str(rank), mode="w") as writer:
        writer.write_all(data)
    
    # wait till all processes finished
    dist.barrier()

    # gather data in rank 0
    if rank == 0:
        world_size = dist.get_world_size()
        files = [save_dir + str(rank) for rank in range(world_size)]

        gathered_data = []
        data_num = 0
        for file in files:
            with jsonlines.open(file, mode="r") as reader:
                data = [obj for obj in reader]
                data_num += len(data)
                gathered_data.append(data)
            if del_cache:
                os.remove(file)

        if total_num == None:
            total_num = data_num
        # remap to original ids
        remapped_data = [None] * total_num
        for rank, data in enumerate(gathered_data):
            for e_id, e in enumerate(data):
                ori_id = rank + e_id * world_size
                if ori_id < total_num:
                    remapped_data[ori_id] = e 
                
        for line in remapped_data:
            assert(line != None)
        
        return remapped_data

        
        