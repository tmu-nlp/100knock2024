c = LR.coef_
"""
[開始:終了:ステップ]
"""
c0 = np.sort(abs(c[0]))[::-1]
print(c0[:10])