from .util import mkdir


# directory to store the results

# results_dir = './results_no_agu/'
results_dir = './results/'
mkdir(results_dir)

# root to the testsets

# dataroot = '/srv/beegfs02/scratch/generative_modeling/data/Deepfake/CNNDetection_data/dataset/test'
dataroot = '/scratch_net/kringel/chuqli/dataset/test/total_test'

# list of synthesis algorithms

# vals = ['progan', 'stylegan', 'biggan', 'cyclegan', 'stargan', 'gaugan',
#         'crn', 'imle', 'seeingdark', 'san', 'deepfake', 'stylegan2', 'whichfaceisreal']
vals = ['progan',  'biggan', 'cyclegan', 'gaugan',
         'imle']
# vals = ['gaugan']

# indicates if corresponding testset has multiple classes

# multiclass = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
multiclass = [1, 0, 1, 0, 0]
# multiclass = [0]

# model

# model_path = 'checkpoints/no_aug/model_epoch_best.pth'
# model_path = 'checkpoints/no_aug_gaugan/model_epoch_best.pth'
model_path = ''
