import glob
import math
import cv2
import numpy as np

def bottle():
    X_list = glob.glob('bottle/train/good/*.png')
    Y_list1 = glob.glob('bottle/test/broken_large/*.png')
    Y_list2 = glob.glob('bottle/test/broken_small/*.png')
    Y_list3 = glob.glob('bottle/test/contamination/*.png')
    Y_list4 = glob.glob('bottle/test/good/*.png')
    X_train = None
    X_test_good = None
    X_test_error = None
    for X_file in X_list:
      im = cv2.imread(X_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_train is not None:
        X_train = np.concatenate((X_train, im))
      if X_train is None:
        X_train = im

    for Y_file in Y_list1:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list2:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list3:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list4:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_good is not None:
        X_test_good = np.concatenate((X_test_good, im))
      if X_test_good is None:
        X_test_good = im
        
    X_train = X_train/255
    X_test_good = X_test_good/255
    X_test_error = X_test_error/255
    X_test = np.concatenate([X_test_good, X_test_error])
    
    y_tr = np.ones(len(X_train))
    y_tg = np.ones(len(X_test_good))
    y_te = []
    for i in range(len(X_test_error)):
        y_te.append(-1)
    y_te = np.array(y_te)
    y_ts = np.concatenate([y_tg, y_te])

    y_tr = np.reshape(y_tr,(X_train.shape[0], 1))
    y_tg = np.reshape(y_tg,(X_test_good.shape[0], 1))
    y_te = np.reshape(y_te,(X_test_error.shape[0], 1))
    y_ts = np.reshape(y_ts,(X_test.shape[0], 1))
    
    return X_train, X_test, X_test_good, X_test_error, y_tr, y_tg, y_te, y_ts


def cable():
    X_list = glob.glob('cable/train/good/*.png')
    Y_list1 = glob.glob('cable/test/bent_wire/*.png')
    Y_list2 = glob.glob('cable/test/cable_swap/*.png')
    Y_list3 = glob.glob('cable/test/combined/*.png')
    Y_list4 = glob.glob('cable/test/cut_inner_insulation/*.png')
    Y_list5 = glob.glob('cable/test/cut_outer_insulation/*.png')
    Y_list6 = glob.glob('cable/test/missing_cable/*.png')
    Y_list7 = glob.glob('cable/test/missingwire/*.png')
    Y_list8 = glob.glob('cable/test/poke_insulation/*.png')
    Y_list9 = glob.glob('cable/test/good/*.png')
    X_train = None
    X_test_good = None
    X_test_error = None
    for X_file in X_list:
      im = cv2.imread(X_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_train is not None:
        X_train = np.concatenate((X_train, im))
      if X_train is None:
        X_train = im

    for Y_file in Y_list1:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list2:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list3:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list4:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im
        
    for Y_file in Y_list5:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im
        
    for Y_file in Y_list6:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im
        
    for Y_file in Y_list7:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im
        
    for Y_file in Y_list8:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list9:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_good is not None:
        X_test_good = np.concatenate((X_test_good, im))
      if X_test_good is None:
        X_test_good = im
        
    X_train = X_train/255
    X_test_good = X_test_good/255
    X_test_error = X_test_error/255
    X_test = np.concatenate([X_test_good, X_test_error])
    
    y_tr = np.ones(len(X_train))
    y_tg = np.ones(len(X_test_good))
    y_te = []
    for i in range(len(X_test_error)):
        y_te.append(-1)
    y_te = np.array(y_te)
    y_ts = np.concatenate([y_tg, y_te])

    y_tr = np.reshape(y_tr,(X_train.shape[0], 1))
    y_tg = np.reshape(y_tg,(X_test_good.shape[0], 1))
    y_te = np.reshape(y_te,(X_test_error.shape[0], 1))
    y_ts = np.reshape(y_ts,(X_test.shape[0], 1))
    
    return X_train, X_test, X_test_good, X_test_error, y_tr, y_tg, y_te, y_ts


def capsule():
    X_list = glob.glob('capsule/train/good/*.png')
    Y_list1 = glob.glob('capsule/test/crack/*.png')
    Y_list2 = glob.glob('capsule/test/faulty_imprint/*.png')
    Y_list3 = glob.glob('capsule/test/poke/*.png')
    Y_list4 = glob.glob('capsule/test/scratch/*.png')
    Y_list5 = glob.glob('capsule/test/squeeze/*.png')
    Y_list6 = glob.glob('capsule/test/good/*.png')
    X_train = None
    X_test_good = None
    X_test_error = None
    for X_file in X_list:
      im = cv2.imread(X_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_train is not None:
        X_train = np.concatenate((X_train, im))
      if X_train is None:
        X_train = im

    for Y_file in Y_list1:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list2:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list3:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im
        
    for Y_file in Y_list4:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im
     
    for Y_file in Y_list5:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list6:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_good is not None:
        X_test_good = np.concatenate((X_test_good, im))
      if X_test_good is None:
        X_test_good = im
        
    X_train = X_train/255
    X_test_good = X_test_good/255
    X_test_error = X_test_error/255
    X_test = np.concatenate([X_test_good, X_test_error])
    
    y_tr = np.ones(len(X_train))
    y_tg = np.ones(len(X_test_good))
    y_te = []
    for i in range(len(X_test_error)):
        y_te.append(-1)
    y_te = np.array(y_te)
    y_ts = np.concatenate([y_tg, y_te])

    y_tr = np.reshape(y_tr,(X_train.shape[0], 1))
    y_tg = np.reshape(y_tg,(X_test_good.shape[0], 1))
    y_te = np.reshape(y_te,(X_test_error.shape[0], 1))
    y_ts = np.reshape(y_ts,(X_test.shape[0], 1))
    
    return X_train, X_test, X_test_good, X_test_error, y_tr, y_tg, y_te, y_ts


def carpet():
    X_list = glob.glob('carpet/train/good/*.png')
    Y_list1 = glob.glob('carpet/test/color/*.png')
    Y_list2 = glob.glob('carpet/test/cut/*.png')
    Y_list3 = glob.glob('carpet/test/hole/*.png')
    Y_list4 = glob.glob('carpet/test/metal_contamination/*.png')
    Y_list5 = glob.glob('carpet/test/thread/*.png')
    Y_list6 = glob.glob('carpet/test/good/*.png')
    X_train = None
    X_test_good = None
    X_test_error = None
    for X_file in X_list:
      im = cv2.imread(X_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_train is not None:
        X_train = np.concatenate((X_train, im))
      if X_train is None:
        X_train = im

    for Y_file in Y_list1:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list1:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im
    
    for Y_file in Y_list2:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im
    
    for Y_file in Y_list3:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list4:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list5:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list6:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_good is not None:
        X_test_good = np.concatenate((X_test_good, im))
      if X_test_good is None:
        X_test_good = im
        
    X_train = X_train/255
    X_test_good = X_test_good/255
    X_test_error = X_test_error/255
    X_test = np.concatenate([X_test_good, X_test_error])
    
    y_tr = np.ones(len(X_train))
    y_tg = np.ones(len(X_test_good))
    y_te = []
    for i in range(len(X_test_error)):
        y_te.append(-1)
    y_te = np.array(y_te)
    y_ts = np.concatenate([y_tg, y_te])

    y_tr = np.reshape(y_tr,(X_train.shape[0], 1))
    y_tg = np.reshape(y_tg,(X_test_good.shape[0], 1))
    y_te = np.reshape(y_te,(X_test_error.shape[0], 1))
    y_ts = np.reshape(y_ts,(X_test.shape[0], 1))
    
    return X_train, X_test, X_test_good, X_test_error, y_tr, y_tg, y_te, y_ts


def grid():
    X_list = glob.glob('grid/train/good/*.png')
    Y_list1 = glob.glob('grid/test/bent/*.png')
    Y_list2 = glob.glob('grid/test/broken/*.png')
    Y_list3 = glob.glob('grid/test/glue/*.png')
    Y_list4 = glob.glob('grid/test/metal_contamination/*.png')
    Y_list5 = glob.glob('grid/test/thread/*.png')
    Y_list6 = glob.glob('grid/test/good/*.png')
    X_train = None
    X_test_good = None
    X_test_error = None
    for X_file in X_list:
      im = cv2.imread(X_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_train is not None:
        X_train = np.concatenate((X_train, im))
      if X_train is None:
        X_train = im

    for Y_file in Y_list1:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list2:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list3:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list4:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list5:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list6:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_good is not None:
        X_test_good = np.concatenate((X_test_good, im))
      if X_test_good is None:
        X_test_good = im
        
    X_train = X_train/255
    X_test_good = X_test_good/255
    X_test_error = X_test_error/255
    X_test = np.concatenate([X_test_good, X_test_error])
    
    y_tr = np.ones(len(X_train))
    y_tg = np.ones(len(X_test_good))
    y_te = []
    for i in range(len(X_test_error)):
        y_te.append(-1)
    y_te = np.array(y_te)
    y_ts = np.concatenate([y_tg, y_te])

    y_tr = np.reshape(y_tr,(X_train.shape[0], 1))
    y_tg = np.reshape(y_tg,(X_test_good.shape[0], 1))
    y_te = np.reshape(y_te,(X_test_error.shape[0], 1))
    y_ts = np.reshape(y_ts,(X_test.shape[0], 1))
    
    return X_train, X_test, X_test_good, X_test_error, y_tr, y_tg, y_te, y_ts


def hazelnut():
    X_list = glob.glob('hazelnut/train/good/*.png')
    Y_list1 = glob.glob('hazelnut/test/crack/*.png')
    Y_list2 = glob.glob('hazelnut/test/cut/*.png')
    Y_list3 = glob.glob('hazelnut/test/hole/*.png')
    Y_list4 = glob.glob('hazelnut/test/print/*.png')
    Y_list5 = glob.glob('hazelnut/test/good/*.png')
    X_train = None
    X_test_good = None
    X_test_error = None
    for X_file in X_list:
      im = cv2.imread(X_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_train is not None:
        X_train = np.concatenate((X_train, im))
      if X_train is None:
        X_train = im

    for Y_file in Y_list1:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list2:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list3:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list4:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list5:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_good is not None:
        X_test_good = np.concatenate((X_test_good, im))
      if X_test_good is None:
        X_test_good = im
        
    X_train = X_train/255
    X_test_good = X_test_good/255
    X_test_error = X_test_error/255
    X_test = np.concatenate([X_test_good, X_test_error])
    
    y_tr = np.ones(len(X_train))
    y_tg = np.ones(len(X_test_good))
    y_te = []
    for i in range(len(X_test_error)):
        y_te.append(-1)
    y_te = np.array(y_te)
    y_ts = np.concatenate([y_tg, y_te])

    y_tr = np.reshape(y_tr,(X_train.shape[0], 1))
    y_tg = np.reshape(y_tg,(X_test_good.shape[0], 1))
    y_te = np.reshape(y_te,(X_test_error.shape[0], 1))
    y_ts = np.reshape(y_ts,(X_test.shape[0], 1))
    
    return X_train, X_test, X_test_good, X_test_error, y_tr, y_tg, y_te, y_ts


def leather():
    X_list = glob.glob('leather/train/good/*.png')
    Y_list1 = glob.glob('leather/test/color/*.png')
    Y_list2 = glob.glob('leather/test/cut/*.png')
    Y_list3 = glob.glob('leather/test/fold/*.png')
    Y_list4 = glob.glob('leather/test/glue/*.png')
    Y_list5 = glob.glob('leather/test/poke/*.png')
    Y_list6 = glob.glob('leather/test/good/*.png')
    X_train = None
    X_test_good = None
    X_test_error = None
    for X_file in X_list:
      im = cv2.imread(X_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_train is not None:
        X_train = np.concatenate((X_train, im))
      if X_train is None:
        X_train = im

    for Y_file in Y_list1:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list2:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list3:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list4:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list5:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list6:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_good is not None:
        X_test_good = np.concatenate((X_test_good, im))
      if X_test_good is None:
        X_test_good = im
        
    X_train = X_train/255
    X_test_good = X_test_good/255
    X_test_error = X_test_error/255
    X_test = np.concatenate([X_test_good, X_test_error])
    
    y_tr = np.ones(len(X_train))
    y_tg = np.ones(len(X_test_good))
    y_te = []
    for i in range(len(X_test_error)):
        y_te.append(-1)
    y_te = np.array(y_te)
    y_ts = np.concatenate([y_tg, y_te])

    y_tr = np.reshape(y_tr,(X_train.shape[0], 1))
    y_tg = np.reshape(y_tg,(X_test_good.shape[0], 1))
    y_te = np.reshape(y_te,(X_test_error.shape[0], 1))
    y_ts = np.reshape(y_ts,(X_test.shape[0], 1))
    
    return X_train, X_test, X_test_good, X_test_error, y_tr, y_tg, y_te, y_ts


def metal_nut():
    X_list = glob.glob('metal_nut/train/good/*.png')
    Y_list1 = glob.glob('metal_nut/test/bent/*.png')
    Y_list2 = glob.glob('metal_nut/test/color/*.png')
    Y_list3 = glob.glob('metal_nut/test/flip/*.png')
    Y_list4 = glob.glob('metal_nut/test/scratch/*.png')
    Y_list5 = glob.glob('metal_nut/test/good/*.png')
    X_train = None
    X_test_good = None
    X_test_error = None
    for X_file in X_list:
      im = cv2.imread(X_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_train is not None:
        X_train = np.concatenate((X_train, im))
      if X_train is None:
        X_train = im

    for Y_file in Y_list1:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list2:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list3:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list4:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list5:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_good is not None:
        X_test_good = np.concatenate((X_test_good, im))
      if X_test_good is None:
        X_test_good = im
        
    X_train = X_train/255
    X_test_good = X_test_good/255
    X_test_error = X_test_error/255
    X_test = np.concatenate([X_test_good, X_test_error])
    
    y_tr = np.ones(len(X_train))
    y_tg = np.ones(len(X_test_good))
    y_te = []
    for i in range(len(X_test_error)):
        y_te.append(-1)
    y_te = np.array(y_te)
    y_ts = np.concatenate([y_tg, y_te])

    y_tr = np.reshape(y_tr,(X_train.shape[0], 1))
    y_tg = np.reshape(y_tg,(X_test_good.shape[0], 1))
    y_te = np.reshape(y_te,(X_test_error.shape[0], 1))
    y_ts = np.reshape(y_ts,(X_test.shape[0], 1))
    
    return X_train, X_test, X_test_good, X_test_error, y_tr, y_tg, y_te, y_ts


def pill():
    X_list = glob.glob('pill/train/good/*.png')
    Y_list1 = glob.glob('pill/test/color/*.png')
    Y_list2 = glob.glob('pill/test/combined/*.png')
    Y_list3 = glob.glob('pill/test/contamination/*.png')
    Y_list4 = glob.glob('pill/test/crack/*.png')
    Y_list5 = glob.glob('pill/test/faulty_imprint/*.png')
    Y_list6 = glob.glob('pill/test/pill_type/*.png')
    Y_list7 = glob.glob('pill/test/scratch/*.png')
    Y_list8 = glob.glob('pill/test/good/*.png')
    X_train = None
    X_test_good = None
    X_test_error = None
    for X_file in X_list:
      im = cv2.imread(X_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_train is not None:
        X_train = np.concatenate((X_train, im))
      if X_train is None:
        X_train = im

    for Y_file in Y_list1:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list2:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list3:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list4:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list5:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list6:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list7:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list8:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_good is not None:
        X_test_good = np.concatenate((X_test_good, im))
      if X_test_good is None:
        X_test_good = im
        
    X_train = X_train/255
    X_test_good = X_test_good/255
    X_test_error = X_test_error/255
    X_test = np.concatenate([X_test_good, X_test_error])
    
    y_tr = np.ones(len(X_train))
    y_tg = np.ones(len(X_test_good))
    y_te = []
    for i in range(len(X_test_error)):
        y_te.append(-1)
    y_te = np.array(y_te)
    y_ts = np.concatenate([y_tg, y_te])

    y_tr = np.reshape(y_tr,(X_train.shape[0], 1))
    y_tg = np.reshape(y_tg,(X_test_good.shape[0], 1))
    y_te = np.reshape(y_te,(X_test_error.shape[0], 1))
    y_ts = np.reshape(y_ts,(X_test.shape[0], 1))
    
    return X_train, X_test, X_test_good, X_test_error, y_tr, y_tg, y_te, y_ts


def screw():
    X_list = glob.glob('screw/train/good/*.png')
    Y_list1 = glob.glob('screw/test/manipulated_front/*.png')
    Y_list2 = glob.glob('screw/test/scratch_head/*.png')
    Y_list3 = glob.glob('screw/test/scratch_neck/*.png')
    Y_list4 = glob.glob('screw/test/thread_side/*.png')
    Y_list5 = glob.glob('screw/test/thread_top/*.png')
    Y_list6 = glob.glob('screw/test/good/*.png')
    X_train = None
    X_test_good = None
    X_test_error = None
    for X_file in X_list:
      im = cv2.imread(X_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_train is not None:
        X_train = np.concatenate((X_train, im))
      if X_train is None:
        X_train = im

    for Y_file in Y_list1:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list2:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list3:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list4:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list5:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list6:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_good is not None:
        X_test_good = np.concatenate((X_test_good, im))
      if X_test_good is None:
        X_test_good = im
        
    X_train = X_train/255
    X_test_good = X_test_good/255
    X_test_error = X_test_error/255
    X_test = np.concatenate([X_test_good, X_test_error])
    
    y_tr = np.ones(len(X_train))
    y_tg = np.ones(len(X_test_good))
    y_te = []
    for i in range(len(X_test_error)):
        y_te.append(-1)
    y_te = np.array(y_te)
    y_ts = np.concatenate([y_tg, y_te])

    y_tr = np.reshape(y_tr,(X_train.shape[0], 1))
    y_tg = np.reshape(y_tg,(X_test_good.shape[0], 1))
    y_te = np.reshape(y_te,(X_test_error.shape[0], 1))
    y_ts = np.reshape(y_ts,(X_test.shape[0], 1))
    
    return X_train, X_test, X_test_good, X_test_error, y_tr, y_tg, y_te, y_ts


def tile():
    X_list = glob.glob('tile/train/good/*.png')
    Y_list1 = glob.glob('tile/test/crack/*.png')
    Y_list2 = glob.glob('tile/test/glue_strip/*.png')
    Y_list3 = glob.glob('tile/test/gray_stroke/*.png')
    Y_list4 = glob.glob('tile/test/oil/*.png')
    Y_list5 = glob.glob('tile/test/rough/*.png')
    Y_list6 = glob.glob('tile/test/good/*.png')
    X_train = None
    X_test_good = None
    X_test_error = None
    for X_file in X_list:
      im = cv2.imread(X_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_train is not None:
        X_train = np.concatenate((X_train, im))
      if X_train is None:
        X_train = im

    for Y_file in Y_list1:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list2:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list3:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list4:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list5:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list6:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_good is not None:
        X_test_good = np.concatenate((X_test_good, im))
      if X_test_good is None:
        X_test_good = im
        
    X_train = X_train/255
    X_test_good = X_test_good/255
    X_test_error = X_test_error/255
    X_test = np.concatenate([X_test_good, X_test_error])
    
    y_tr = np.ones(len(X_train))
    y_tg = np.ones(len(X_test_good))
    y_te = []
    for i in range(len(X_test_error)):
        y_te.append(-1)
    y_te = np.array(y_te)
    y_ts = np.concatenate([y_tg, y_te])

    y_tr = np.reshape(y_tr,(X_train.shape[0], 1))
    y_tg = np.reshape(y_tg,(X_test_good.shape[0], 1))
    y_te = np.reshape(y_te,(X_test_error.shape[0], 1))
    y_ts = np.reshape(y_ts,(X_test.shape[0], 1))
    
    return X_train, X_test, X_test_good, X_test_error, y_tr, y_tg, y_te, y_ts


def toothbrush():
    X_list = glob.glob('toothbrush/train/good/*.png')
    Y_list1 = glob.glob('toothbrush/test/defective/*.png')
    Y_list2 = glob.glob('toothbrush/test/good/*.png')
    X_train = None
    X_test_good = None
    X_test_error = None
    for X_file in X_list:
      im = cv2.imread(X_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_train is not None:
        X_train = np.concatenate((X_train, im))
      if X_train is None:
        X_train = im

    for Y_file in Y_list1:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list2:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_good is not None:
        X_test_good = np.concatenate((X_test_good, im))
      if X_test_good is None:
        X_test_good = im
        
    X_train = X_train/255
    X_test_good = X_test_good/255
    X_test_error = X_test_error/255
    X_test = np.concatenate([X_test_good, X_test_error])
    
    y_tr = np.ones(len(X_train))
    y_tg = np.ones(len(X_test_good))
    y_te = []
    for i in range(len(X_test_error)):
        y_te.append(-1)
    y_te = np.array(y_te)
    y_ts = np.concatenate([y_tg, y_te])

    y_tr = np.reshape(y_tr,(X_train.shape[0], 1))
    y_tg = np.reshape(y_tg,(X_test_good.shape[0], 1))
    y_te = np.reshape(y_te,(X_test_error.shape[0], 1))
    y_ts = np.reshape(y_ts,(X_test.shape[0], 1))
    
    return X_train, X_test, X_test_good, X_test_error, y_tr, y_tg, y_te, y_ts


def transistor():
    X_list = glob.glob('transistor/train/good/*.png')
    Y_list1 = glob.glob('transistor/test/bent_lead/*.png')
    Y_list2 = glob.glob('transistor/test/cut_lead/*.png')
    Y_list3 = glob.glob('transistor/test/damaged_case/*.png')
    Y_list4 = glob.glob('transistor/test/misplaced/*.png')
    Y_list5 = glob.glob('transistor/test/good/*.png')
    X_train = None
    X_test_good = None
    X_test_error = None
    for X_file in X_list:
      im = cv2.imread(X_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_train is not None:
        X_train = np.concatenate((X_train, im))
      if X_train is None:
        X_train = im

    for Y_file in Y_list1:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list2:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list3:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list4:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im
        
    for Y_file in Y_list5:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_good is not None:
        X_test_good = np.concatenate((X_test_good, im))
      if X_test_good is None:
        X_test_good = im
        
    X_train = X_train/255
    X_test_good = X_test_good/255
    X_test_error = X_test_error/255
    X_test = np.concatenate([X_test_good, X_test_error])
    
    y_tr = np.ones(len(X_train))
    y_tg = np.ones(len(X_test_good))
    y_te = []
    for i in range(len(X_test_error)):
        y_te.append(-1)
    y_te = np.array(y_te)
    y_ts = np.concatenate([y_tg, y_te])

    y_tr = np.reshape(y_tr,(X_train.shape[0], 1))
    y_tg = np.reshape(y_tg,(X_test_good.shape[0], 1))
    y_te = np.reshape(y_te,(X_test_error.shape[0], 1))
    y_ts = np.reshape(y_ts,(X_test.shape[0], 1))
    
    return X_train, X_test, X_test_good, X_test_error, y_tr, y_tg, y_te, y_ts


def wood():
    X_list = glob.glob('wood/train/good/*.png')
    Y_list1 = glob.glob('wood/test/color/*.png')
    Y_list2 = glob.glob('wood/test/combined/*.png')
    Y_list3 = glob.glob('wood/test/hole/*.png')
    Y_list4 = glob.glob('wood/test/liquid/*.png')
    Y_list5 = glob.glob('wood/test/scratch/*.png')
    Y_list6 = glob.glob('wood/test/good/*.png')
    X_train = None
    X_test_good = None
    X_test_error = None
    for X_file in X_list:
      im = cv2.imread(X_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_train is not None:
        X_train = np.concatenate((X_train, im))
      if X_train is None:
        X_train = im

    for Y_file in Y_list1:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list2:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list3:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im
    
    for Y_file in Y_list4:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im
    
    for Y_file in Y_list5:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list6:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_good is not None:
        X_test_good = np.concatenate((X_test_good, im))
      if X_test_good is None:
        X_test_good = im
        
    X_train = X_train/255
    X_test_good = X_test_good/255
    X_test_error = X_test_error/255
    X_test = np.concatenate([X_test_good, X_test_error])
    
    y_tr = np.ones(len(X_train))
    y_tg = np.ones(len(X_test_good))
    y_te = []
    for i in range(len(X_test_error)):
        y_te.append(-1)
    y_te = np.array(y_te)
    y_ts = np.concatenate([y_tg, y_te])

    y_tr = np.reshape(y_tr,(X_train.shape[0], 1))
    y_tg = np.reshape(y_tg,(X_test_good.shape[0], 1))
    y_te = np.reshape(y_te,(X_test_error.shape[0], 1))
    y_ts = np.reshape(y_ts,(X_test.shape[0], 1))
    
    return X_train, X_test, X_test_good, X_test_error, y_tr, y_tg, y_te, y_ts


def zipper():
    X_list = glob.glob('zipper/train/good/*.png')
    Y_list1 = glob.glob('zipper/test/broken_teeth/*.png')
    Y_list2 = glob.glob('zipper/test/combined/*.png')
    Y_list3 = glob.glob('zipper/test/fabric_border/*.png')
    Y_list4 = glob.glob('zipper/test/fabric_interior/*.png')
    Y_list5 = glob.glob('zipper/test/rough/*.png')
    Y_list6 = glob.glob('zipper/test/split_teeth/*.png')
    Y_list7 = glob.glob('zipper/test/squeezed_teeth/*.png')
    Y_list8 = glob.glob('zipper/test/good/*.png')
    X_train = None
    X_test_good = None
    X_test_error = None
    for X_file in X_list:
      im = cv2.imread(X_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_train is not None:
        X_train = np.concatenate((X_train, im))
      if X_train is None:
        X_train = im

    for Y_file in Y_list1:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list2:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list3:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list4:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list5:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list6:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list7:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_error is not None:
        X_test_error = np.concatenate((X_test_error, im))
      if X_test_error is None:
        X_test_error = im

    for Y_file in Y_list8:
      im = cv2.imread(Y_file)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (256,256))
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
      if X_test_good is not None:
        X_test_good = np.concatenate((X_test_good, im))
      if X_test_good is None:
        X_test_good = im
        
    X_train = X_train/255
    X_test_good = X_test_good/255
    X_test_error = X_test_error/255
    X_test = np.concatenate([X_test_good, X_test_error])
    
    y_tr = np.ones(len(X_train))
    y_tg = np.ones(len(X_test_good))
    y_te = []
    for i in range(len(X_test_error)):
        y_te.append(-1)
    y_te = np.array(y_te)
    y_ts = np.concatenate([y_tg, y_te])

    y_tr = np.reshape(y_tr,(X_train.shape[0], 1))
    y_tg = np.reshape(y_tg,(X_test_good.shape[0], 1))
    y_te = np.reshape(y_te,(X_test_error.shape[0], 1))
    y_ts = np.reshape(y_ts,(X_test.shape[0], 1))
    
    return X_train, X_test, X_test_good, X_test_error, y_tr, y_tg, y_te, y_ts
