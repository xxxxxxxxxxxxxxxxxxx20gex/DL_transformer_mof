# load the packages
import os
import yaml
import pandas as pd

# load the runner
from finetune_transformer import FineTune
# load config file
config = yaml.load(open("config_ft_transformer.yaml", "r"), Loader=yaml.FullLoader)
config['dataloader']['randomSeed'] = 0
print(config)

if 'hMOF' in config['dataset']['data_name']:
    task_name = config['dataset']['data_name']
    pressure = config['dataset']['data_name'].split('_')[-1]
if 'QMOF' in config['dataset']['data_name']:
    task_name = 'QMOF'

# ftf: finetuning from
# ptw: pre-trained with
if config['fine_tune_from'] == 'scratch':
    ftf = 'scratch'
    ptw = 'scratch'
else:
    ftf = config['fine_tune_from'].split('/')[-1]
    ptw = config['trained_with']

seed = config['dataloader']['randomSeed']

log_dir = os.path.join(
    'training_results/finetuning/Transformer',
    'Trans_{}_{}_{}'.format(ptw,task_name,seed)
)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# run the training on dataset
fine_tune = FineTune(config, log_dir)
fine_tune.train()
loss, metric = fine_tune.test()

# save the training results
fn = 'Trans_{}_{}_{}.csv'.format(ptw,task_name,seed)
print(fn)
df = pd.DataFrame([[loss, metric.item()]])
df.to_csv(
    os.path.join(log_dir, fn),
    mode='a', index=False, header=False
)