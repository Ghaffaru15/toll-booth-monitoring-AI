import os, shutil

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

TRAIN_DIR = os.path.join(BASE_DIR, 'cars_train')

VALIDATION_DIR = os.path.join(BASE_DIR, 'cars_validation')

TEST_DIR = os.path.join(BASE_DIR, 'cars_test')

TRAIN_E_DIR = os.path.join(TRAIN_DIR, 'E')
TRAIN_L_DIR = os.path.join(TRAIN_DIR, 'L')
TRAIN_M_DIR = os.path.join(TRAIN_DIR, 'M')
TRAIN_S_DIR = os.path.join(TRAIN_DIR, 'S')

VALIDATION_E_DIR = os.path.join(VALIDATION_DIR, 'E')

VALIDATION_L_DIR = os.path.join(VALIDATION_DIR, 'L')

VALIDATION_M_DIR = os.path.join(VALIDATION_DIR, 'M')

VALIDATION_S_DIR = os.path.join(VALIDATION_DIR, 'S')

TEST_E_DIR = os.path.join(TEST_DIR, 'E')
TEST_L_DIR = os.path.join(TEST_DIR, 'L')
TEST_M_DIR = os.path.join(TEST_DIR, 'M')
TEST_S_DIR = os.path.join(TEST_DIR, 'S')

# print(len(os.listdir(os.path.join(BASE_DIR, 'cars_train/E'))))

# E_FILE_NAMES = os.listdir(TRAIN_E_DIR)
# E_FILE_NAMES = E_FILE_NAMES[0:80]
#
# for fname in E_FILE_NAMES:
#     src = os.path.join(TRAIN_E_DIR, fname)
#     des = os.path.join(VALIDATION_E_DIR, fname)
#     shutil.copyfile(src, des)
#
# L_FILE_NAMES = os.listdir(TRAIN_L_DIR)
# L_FILE_NAMES = L_FILE_NAMES[0:200]
#
# for fname in L_FILE_NAMES:
#     src = os.path.join(TRAIN_L_DIR, fname)
#     des = os.path.join(VALIDATION_L_DIR, fname
#                        )
#     shutil.copyfile(src, des)
#
# M_FILE_NAMES = os.listdir(TRAIN_M_DIR)
# M_FILE_NAMES = M_FILE_NAMES[0:1000]
#
# for fname in M_FILE_NAMES:
#     src = os.path.join(TRAIN_M_DIR, fname)
#     des = os.path.join(VALIDATION_M_DIR, fname)
#
#     shutil.copyfile(src, des)
#
# S_FILE_NAMES = os.listdir(TRAIN_S_DIR)
# S_FILE_NAMES = S_FILE_NAMES[0:2000]
#
# for fname in S_FILE_NAMES:
#     src = os.path.join(TRAIN_S_DIR, fname)
#     des = os.path.join(VALIDATION_S_DIR, fname)
#
#     shutil.copyfile(src, des)
