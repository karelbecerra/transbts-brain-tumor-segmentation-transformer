def read(data_dir):
  base_dir = '.'
  data_dir = '/' + data_dir + '/'
  train_dir = base_dir + data_dir + 'Train'
  valid_dir = base_dir + data_dir + 'Valid'
  checkpoint_dir = base_dir + '/checkpoint/'

  train_file = base_dir + data_dir + 'train.txt'
  valid_file = base_dir + data_dir + 'valid.txt'
  return train_dir, train_file, valid_dir, valid_file, checkpoint_dir
