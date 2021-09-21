from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import pickle as pkl
# df = pd.read_csv('hiro_calibration_camera_xls.csv')
df = pd.read_csv('move_locs.csv')
# regr = make_pipeline(StandardScaler(), SVR())
# import pdb; pdb.set_trace()
# regr.fit( df[['xhat','yhat']], df.x)
# regr.predict(df[['xhat','yhat']])


from sklearn.ensemble import RandomForestRegressor
xregr = RandomForestRegressor(n_estimators=5)
# xregr = make_pipeline(StandardScaler(), SVR(kernel='linear'))
xregr.fit( df[['x','y']], df.xa)
print(f"X R^2: {xregr.score( df[['x','y']], df.xa)}")
xpreds = xregr.predict(df[['x','y']])
print(f"Mean Residual: {np.mean(np.abs(xpreds-df.xa))}")
print(f"Error Center: {np.mean(xpreds-df.xa)}")
print(f"Predicted X for (0,200): {xregr.predict([[0,200]])}")
print(f"Predicted X for (200,200): {xregr.predict([[200,200]])}")

# yregr = RandomForestRegressor(n_estimators=5)
yregr = make_pipeline(StandardScaler(), SVR(kernel='linear'))
yregr.fit( df[['x','y']], df.ya)
print(f"Y R^2: {yregr.score( df[['x','y']], df.ya)}")
ypreds = yregr.predict(df[['x','y']])
print(f"Mean Residual: {np.mean(np.abs(ypreds-df.ya))}")
print(f"Error Center: {np.mean(ypreds-df.ya)}")
print(f"Predicted y for (-150,310): {yregr.predict([[-150,310]])}")
print(f"Predicted y for (0,310): {yregr.predict([[0,310]])}")
print(f"Predicted y for (150,310): {yregr.predict([[150,310]])}")

# import pdb; pdb.set_trace()

with open('random_forest_locomotion_regressor.pkl', 'wb+') as outfile:
    pkl.dump((xregr, yregr), outfile)

# with open('random_forest_view_regressor.pkl', 'wb+') as outfile:
#     pkl.dump((xregr, yregr), outfile)