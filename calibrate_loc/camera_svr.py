from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import pickle as pkl
# df = pd.read_csv('hiro_calibration_camera_xls.csv')
# df = pd.read_csv('cam_locs.csv')
df = pd.read_csv('sheet.csv')
# xregr = make_pipeline(StandardScaler(), SVR())

# xregr.fit( df[['xa','ya']], df.x)
# xregr.predict(df[['xa','ya']])
# import pdb; pdb.set_trace()

from sklearn.ensemble import RandomForestRegressor
xregr = RandomForestRegressor(n_estimators=10)

# xregr = make_pipeline(StandardScaler(), SVR(kernel='linear'))

#!!!! Adding 10 because there seems to be a systemic offset in sheets.csv

xregr.fit( df[['xa','ya']], df.x+10)
print(f"X R^2: {xregr.score( df[['xa','ya']], df.x+10)}")
xpreds = xregr.predict(df[['xa','ya']])
print(f"Mean Residual: {np.mean(np.abs(xpreds-(df.x+10)))}")
print(f"Error Center: {np.mean(xpreds-(df.x+10))}")
# print(f"Predicted X for (0,200): {xregr.predict([[0,200]])}")
# print(f"Predicted X for (200,200): {xregr.predict([[200,200]])}")

yregr = RandomForestRegressor(n_estimators=10)
# yregr = make_pipeline(StandardScaler(), SVR(kernel='linear'))
yregr.fit( df[['xa','ya']], df.y)
print(f"Y R^2: {yregr.score( df[['xa','ya']], df.y)}")
ypreds = yregr.predict(df[['xa','ya']])
print(yregr.predict([[-224.59,193.8]]))
print(f"Mean Residual: {np.mean(np.abs(ypreds-df.y))}")
print(f"Error Center: {np.mean(ypreds-df.y)}")

# import pdb; pdb.set_trace()

with open('random_forest_view_regressor.pkl', 'wb+') as outfile:
    pkl.dump((xregr, yregr), outfile)

# with open('random_forest_view_regressor.pkl', 'wb+') as outfile:
#     pkl.dump((xregr, yregr), outfile)