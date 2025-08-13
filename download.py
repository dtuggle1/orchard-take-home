import os

def check_if_downloaded(folder_name, filenames):
  """
  Checks if a specified folder exists in the root directory, then check to if the folder contains the filenames that are expected.
  Returns True if both are True, False otherwise
  """
  if folder_name in os.listdir(): # check if the folder exists
    if set(os.listdir(folder_name)) == filenames: # check if the filenames match up with what is expected
      return True
  return False

def download_image_folder(folder_name, folder_link, expected_filenames):
  """
  Checks if the folder is already downloaded properly, and if not, downloads the files.
  """
  if not check_if_downloaded(folder_name, expected_filenames):
    gdown.download_folder(folder_link, quiet=True)
    if not check_if_downloaded(folder_name, expected_filenames):
      raise ValueError(f'{folder_name} Not Downloaded Correctly')
  else:
    print(f'{folder_name} already downloaded')
    return