import pandas as pd
from bert_juman import BertWithJumanModel
import numpy as np
import pandas as pd

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import lightgbm as lgb

from sklearn import datasets
from sklearn.model_selection import train_test_split

import numpy as np

"""LightGBM を使った多値分類のサンプルコード"""


def main():
    # Iris データセットを読み込む
    #iris = datasets.load_iris()
    #X, y = iris.data, iris.target
    #print(X)
    #print(y)
    #df = pd.read_csv('../gitweb.txt',delimiter='\t', header=None)
    df = pd.read_csv('./enre_test_changed.txt',delimiter=',')
    X = np.load('out_description.npy')
    y = df['genre_num'].values 
    # 訓練データとテストデータに分割する
    X_train, X_test, y_train, y_test = train_test_split(X, y)

  
    # データセットを生成する
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    print(lgb_train)
    print(lgb_eval)
    # LightGBM のハイパーパラメータ
    lgbm_params = {
        # 多値分類問題
        'objective': 'multiclass',
        # クラス数は 25
        'num_class': 29,
    }

    # 上記のパラメータでモデルを学習する
    model = lgb.train(lgbm_params, lgb_train, valid_sets=lgb_eval)

    import pickle 
    # テストデータを予測する
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred_max = np.argmax(y_pred, axis=1)  # 最尤と判断したクラスの値にする

    # 精度 (Accuracy) を計算する
    accuracy = sum(y_test == y_pred_max) / len(y_test)
    print(accuracy)


    with open('model2.pickle',mode = 'wb') as fp:
        pickle.dump(model,fp)

if __name__ == '__main__':
    main()

