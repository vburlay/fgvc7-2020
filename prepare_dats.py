import os
import random
import shutil
import csv
def split_data(SOURCE_DIR, TRAINING_DIR, VALIDATION_DIR, SPLIT_SIZE):
  """
  Splits the data into train and test sets
  Args:
    SOURCE_DIR (string): directory path containing the images
    TRAINING_DIR (string): directory path to be used for training
    VALIDATION_DIR (string): directory path to be used for validation
    SPLIT_SIZE (float): proportion of the dataset to be used for training
  Returns:
    None
  """

  dataset = []
  for unitData in os.listdir(SOURCE_DIR):
    data = SOURCE_DIR + unitData
    if (os.path.getsize(data) > 0):
        dataset.append(unitData)
    else:
        print(unitData + " is zero length, so ignoring")

  val_data_length = int(len(dataset) * SPLIT_SIZE)
  train_data_length = int(len(dataset) - val_data_length)
  shuffled_set = random.sample(dataset, len(dataset))
  train_set = shuffled_set[0:train_data_length]
  val_set = shuffled_set[-val_data_length:]
  for unitData in train_set:
    temp_train_data = SOURCE_DIR + unitData
    final_train_data = TRAINING_DIR + unitData
    shutil.copyfile(temp_train_data, final_train_data)
  for unitData in val_set:
    temp_test_data = SOURCE_DIR + unitData
    final_val_data = VALIDATION_DIR + unitData
    shutil.copyfile(temp_test_data, final_val_data)
def parse_data_from_input(filename,source_dir,target_dir):
    with open(filename,'r') as file:
        csv_reader = csv.reader(file,delimiter=',')
        # Skip header
        next(csv_reader, None)

        for row in csv_reader:
            if row[1] == '1':
                ziel_dir = os.path.join(target_dir,'healthy/')
                temp_test_data = source_dir + row[0] + '.jpg'
                final_val_data = ziel_dir +  row[0] + '.jpg'
                shutil.copyfile(temp_test_data, final_val_data)
            elif  row[2] == '1':
                ziel_dir = os.path.join(target_dir, 'multiple_diseases/')
                temp_test_data = source_dir + row[0] + '.jpg'
                final_val_data = ziel_dir + row[0] + '.jpg'
                shutil.copyfile(temp_test_data, final_val_data)
            elif row[3] == '1':
                ziel_dir = os.path.join(target_dir, 'rust/')
                temp_test_data = source_dir + row[0] + '.jpg'
                final_val_data = ziel_dir + row[0] + '.jpg'
                shutil.copyfile(temp_test_data, final_val_data)
            elif row[4] == '1':
                ziel_dir = os.path.join(target_dir, 'scab/')
                temp_test_data = source_dir + row[0] + '.jpg'
                final_val_data = ziel_dir + row[0] + '.jpg'
                shutil.copyfile(temp_test_data, final_val_data)

if (__name__) == '__main__':
    cwd = os.getcwd()
    SOURCE_DIR = os.path.join(cwd,'data/train_full/')
    TRAINING_DIR = os.path.join(cwd, 'data/train/')
    VALIDATION_DIR = os.path.join(cwd, 'data/validation/')
    SPLIT_SIZE = 0.2
    filename = os.path.join(cwd, 'data/train.csv')

    source_dir = os.path.join(cwd, 'data/train_full/')
    target_dir = source_dir
    parse_data_from_input(filename,source_dir,target_dir)




#    split_data(SOURCE_DIR, TRAINING_DIR, VALIDATION_DIR, SPLIT_SIZE)



