#%%
# load linear model and the corresponding test set
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

with open ('linear_model.pkl', 'rb') as file:
  linear = pickle.load(file)

with open ('./data/intermid/X_test_fs.pkl', 'rb') as file:
  X_test_fs = pickle.load(file)

with open ('./data/intermid/y_test.pkl', 'rb') as file:
  y_test = pickle.load(file)

# evaluate the model
y_pred = linear.predict(X_test_fs)
# evaluate predictions
sns.lineplot(y_test, y_pred)
plt.xlabel('True rent price')
plt.ylabel('Predicted price')
plt.show()

# %%
