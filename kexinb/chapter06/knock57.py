# task 57. 特徴量の重みの確認
# 52で学習したロジスティック回帰モデルの中で，重みの高い特徴量トップ10と，重みの低い特徴量トップ10を確認せよ．

from sklearn.metrics import *
from knock53 import *
import numpy as np

# extract col names of the feature matrix (names of features used in lr)
feat = X_train.columns.values 


if __name__ == "__main__" :
    # lr.classes_: class labels
    # lr.coef_: coefficients of features for each class.
    ind = [i for i in range(1, 11)]
    for c, coef in zip(lr.classes_, lr.coef_):
        top10 = pd.DataFrame(feat[np.argsort(-coef)[:10]],
                            columns=["Most Important (+) 10"], index=ind)
        worst10 = pd.DataFrame(feat[np.argsort(coef)[:10]], 
                               columns=["Most Important (-) 10"], index=ind)
        least10 = pd.DataFrame(feat[np.argsort(np.abs(coef))[:10]], 
                                       columns=["Least Important 10"], index=ind)
        print(f"Features for Class {c}")
        print(top10)
        print()
        print(worst10)
        print()
        print(least10)
        print("---------------------")

'''
Features for Class b
   Most Important (+) 10
1                    fed
2                   bank
3                    ecb
4                    oil
5                 stocks
6                  china
7                   euro
8                ukraine
9              obamacare
10                   ipo

   Most Important (-) 10
1                  video
2                    her
3                  ebola
4                    and
5               facebook
6                  virus
7              microsoft
8                tmobile
9                     tv
10                   the

   Least Important 10
1                need
2             the top
3              facing
4             may not
5              patent
6              virgin
7              senate
8               fears
9              who is
10               tony
---------------------
Features for Class e
   Most Important (+) 10
1             kardashian
2                  chris
3                    kim
4                  movie
5                   film
6                   star
7                    her
8                wedding
9                   paul
10              jennifer

   Most Important (-) 10
1                 update
2                     us
3                 google
4                   says
5                  china
6                     gm
7                  study
8                   data
9                billion
10                   ceo

   Least Important 10
1              unveil
2                must
3              future
4            stunning
5               mixed
6                ever
7              brings
8             york ap
9             finally
10              comes
---------------------
Features for Class m
   Most Important (+) 10
1                  ebola
2                   drug
3                 cancer
4                  study
5                    fda
6                   mers
7                  cases
8                    cdc
9                medical
10                 could

   Most Important (-) 10
1                     gm
2               facebook
3                  sales
4                    ceo
5                 google
6                   bank
7                 amazon
8                   deal
9                twitter
10               climate

   Least Important 10
1           no longer
2             and the
3               model
4                aged
5               agree
6             weather
7             dies at
8                zone
9               doing
10           comments
---------------------
Features for Class t
   Most Important (+) 10
1                 google
2               facebook
3                  apple
4                climate
5              microsoft
6                tmobile
7                     gm
8                  tesla
9                googles
10               samsung

   Most Important (-) 10
1                   drug
2                    fed
3               american
4                    her
5                 cancer
6                ukraine
7                 stocks
8                 shares
9                  movie
10                   ecb

   Least Important 10
1               soars
2               shock
3                fund
4                girl
5          on twitter
6                wont
7                 our
8                 any
9                turn
10              by us
'''