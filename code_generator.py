
import json
import random
import numpy as np
from datetime import datetime
import os
import re

# import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# lemmatizer = WordNetLemmatizer()
# eng_stopwords = stopwords.words('english')

time_now = datetime.now().strftime(r"%m_%d_%Y-%H_%M_%S")

data_dir = "atlas_data"
train_data_dir = "data"

os.makedirs(f'{data_dir}/{train_data_dir}', exist_ok=True)

test_file_name = f"test_{time_now}.jsonl"
dev_file_name = f"dev_{time_now}.jsonl"

# Hyper-parameters
params = {
    'train_steps' : 60,
    'eval_steps' : 1
}


# def preprocess_text(text):
#     text = re.sub('[^A-Za-z0-9]',' ', text)
#     text = text.lower().split()

#     text = [lemmatizer.lemmatize(word) for word in text if not word in eng_stopwords]
#     text = ' '.join(text)
#     return text

def seperate_train_test_data(samples):
    train_samples = []
    test_samples = []
    
    for sample in samples:
        # sample['claim'] = preprocess_text(sample['claim'])
        sample['claim'] = sample['claim'].lower()
        if sample['label'] != "NOT ENOUGH INFO":
            if sample['contributed_by'] != 'group 10':
                train_samples.append(sample)
            else:
                test_samples.append(sample)
    
    return train_samples, test_samples

def generate_dev_test(samples, test_file_name, dev_file_name):
    random.seed(42)
    
    samples_len = len(samples)
    nums = random.sample(range(samples_len), int(0.5*samples_len))
    
    train_nums = list(set(nums))
    dev_nums = list(set(range(samples_len)) - set(train_nums))
    
    temp_arr = np.array(samples)
    test_data = temp_arr[train_nums]
    dev_data = temp_arr[dev_nums]
    
    
    with open(f"{data_dir}/{train_data_dir}/{test_file_name}", 'w') as f:
        for line in test_data:
            json.dump(line, f)
            f.write("\n")
            
    with open(f"{data_dir}/{train_data_dir}/{dev_file_name}", 'w') as f:
        for line in dev_data:
            json.dump(line, f)
            f.write("\n")
    print("Test and Dev file generated")


def generate_train_data(samples, num_of_samples: int=None):
    random.seed(42)
    train_file_name = f"train_{time_now}.jsonl"
    
    if num_of_samples!=None:
        samples_len = len(samples)
        nums = random.sample(range(samples_len), num_of_samples)
        
        temp_arr = np.array(samples)
        train_data = temp_arr[nums]
        
        with open(f"{data_dir}/{train_data_dir}/{train_file_name}", 'w') as f:
            for line in train_data:
                json.dump(line, f)
                f.write("\n")
        return train_file_name
    return None



# Generate Script file
def generate_bash_script(email, sjsu_id, train_data_dir, train_file_name, eval_file_name):
    exp_name = f"{sjsu_id}_{time_now}"

    s = f"""#!/bin/bash
#SBATCH --mail-user={email}
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=gpuTest_{sjsu_id}
#SBATCH --output=gpuTest_%j.out
#SBATCH --error=gpuTest_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu   

# on coe-hpc1 cluster load
#module load python3/3.8.8
#
# on coe-hpc2 cluster load:
module load python-3.10.8-gcc-11.2.0-c5b5yhp slurm

export http_proxy=http://172.16.1.2:3128; export https_proxy=http://172.16.1.2:3128

cd /home/{sjsu_id}/atlas

DATA_DIR=/home/{sjsu_id}/atlas/atlas_data
port=$(shuf -i 15000-16000 -n 1)
TRAIN_FILE="${{DATA_DIR}}/{train_data_dir}/{train_file_name}"
EVAL_FILES="${{DATA_DIR}}/{train_data_dir}/{eval_file_name}"
SAVE_DIR=${{DATA_DIR}}/experiments
EXPERIMENT_NAME={exp_name}
TRAIN_STEPS={params['train_steps']}
EVAL_STEPS={params['eval_steps']}

# submit your code to Slurm 
python3 /home/{sjsu_id}/atlas/train.py --shuffle  --train_retriever  --gold_score_mode pdist   --use_gradient_checkpoint_reader --use_gradient_checkpoint_retriever  --precision bf16   --shard_optim --shard_grads   --temperature_gold 0.01   --refresh_index -1   --query_side_retriever_training  --target_maxlength 16   --reader_model_type google/t5-base-lm-adapt --dropout 0.1 --weight_decay 0.01 --lr 4e-5 --lr_retriever 4e-5 --scheduler linear   --text_maxlength 256   --model_path "/home/{sjsu_id}/atlas/atlas_data/models/atlas/base/"  --train_data ${{TRAIN_FILE}}   --eval_data ${{EVAL_FILES}}   --per_gpu_batch_size 1  --n_context 10   --retriever_n_context 10   --name ${{EXPERIMENT_NAME}}   --checkpoint_dir ${{SAVE_DIR}}   --eval_freq ${{EVAL_STEPS}}   --log_freq 4   --total_steps ${{TRAIN_STEPS}}   --warmup_steps 5  --save_freq ${{TRAIN_STEPS}}   --main_port $port   --write_results   --task fever   --index_mode flat   --passages "/home/{sjsu_id}/atlas/atlas_data/corpora/unified-passage-set-v1.jsonl"  --save_index_path ${{SAVE_DIR}}/${{EXPERIMENT_NAME}}/saved_index 
"""
    
    bash_file_name = f"train_job_{time_now}.sh"
    
    with open(bash_file_name, 'w') as f:
        f.write(s)
    
    print("Bash file generated Successfully!")
    return exp_name, bash_file_name


def generate_test_bash_script(email, sjsu_id, train_data_dir, test_file_name, exp_name):
    eval_results_name = f"Eval_{sjsu_id}_{time_now}"
    s = f"""#!/bin/bash
#SBATCH --mail-user={email}
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=gpuTest_{sjsu_id}
#SBATCH --output=gpuTest_%j.out
#SBATCH --error=gpuTest_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu   

# on coe-hpc1 cluster load
#module load python3/3.8.8
#
# on coe-hpc2 cluster load:
module load python-3.10.8-gcc-11.2.0-c5b5yhp slurm

export http_proxy=http://172.16.1.2:3128; export https_proxy=http://172.16.1.2:3128

cd /home/{sjsu_id}/atlas

DATA_DIR=/home/{sjsu_id}/atlas/atlas_data
port=$(shuf -i 15000-16000 -n 1)
EVAL_FILES="${{DATA_DIR}}/{train_data_dir}/{test_file_name}"
SAVE_DIR=${{DATA_DIR}}/experiments/
EXPERIMENT_NAME={exp_name}
TRAIN_STEPS={params['train_steps']}
EVAL_STEPS={params['eval_steps']}

# submit your code to Slurm 
python3 /home/{sjsu_id}/atlas/evaluate.py --name {eval_results_name} --generation_max_length 16 --gold_score_mode "pdist" --precision fp32 --reader_model_type google/t5-base-lm-adapt --model_path ${{SAVE_DIR}}/${{EXPERIMENT_NAME}}/checkpoint/step-{params['train_steps']} --eval_data ${{EVAL_FILES}} --per_gpu_batch_size 1 --n_context 40 --retriever_n_context 40 --checkpoint_dir ${{SAVE_DIR}} --main_port $port --index_mode "flat" --task "fever" --load_index_path ${{SAVE_DIR}}/${{EXPERIMENT_NAME}}/saved_index --write_results
"""

    test_bash_file_name = f"test_job_{time_now}.sh"
    
    with open(test_bash_file_name, 'w') as f:
        f.write(s)
    
    print("Test Bash file generated Successfully!")
    return test_bash_file_name



if __name__=="__main__":
    with open("fc_unified_questions_apr13.jsonl") as f:
        data = f.read()
    data_dict_list = [json.loads(d) for d in data.split("\n") if len(d.strip())>0]
    
    # Seperates train and test samples 
    train_samples, test_samples = seperate_train_test_data(data_dict_list)

    # Generate dev and test files
    generate_dev_test(test_samples, test_file_name, dev_file_name)

    # Generate train files
    num_of_train_samples = int(input("Enter number of train_samples to generate: "))
    train_file_name = generate_train_data(train_samples, num_of_train_samples)


    email = "munjalkishorbhai.desai@sjsu.edu"
    sjsu_id = "014828684"
    # email = input("Enter sjsu email id: ")
    # sjsu_id = input("Enter sjsu id: ")
    exp_name, bash_file_name = generate_bash_script(email, sjsu_id, train_data_dir, train_file_name, dev_file_name)

    test_bash_file_name = generate_test_bash_script(email, sjsu_id, train_data_dir, test_file_name, exp_name)

    # run shell script
    print("Running Trainig shell script...")
    os.system(f"sbatch {bash_file_name}")

    print("Testing shell script name ...", test_bash_file_name)
    

    print("Great Success!!!")


   
