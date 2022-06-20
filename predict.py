#%%
# load linear model and the corresponding test set
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

with open ('./models/linear_model.pkl', 'rb') as file:
  linear = pickle.load(file)

with open ('./data/intermid/X_test.pkl', 'rb') as file:
  X_test = pickle.load(file)

with open ('./data/intermid/y_test.pkl', 'rb') as file:
  y_test = pickle.load(file)

# evaluate the model
y_pred = linear.predict(X_test)
# evaluate predictions
sns.lineplot(y_test, y_pred)
plt.xlabel('True rent price')
plt.ylabel('Predicted price')
plt.show()
#%%
linear.score(X_test,y_test)

# %%
# load model2 and X_test y_test
import pickle
model = pickle.load(open('./models/model.pkl', 'rb'))

X2_test = pickle.load(open('./data/final/X_test.pkl','rb'))
y2_test = pickle.load(open('./data/final/y_test.pkl','rb'))

model.predict(X2_test)[:5]

# %%
y2_test[:5]

# %%
model.score(X2_test,y2_test)
# %%
