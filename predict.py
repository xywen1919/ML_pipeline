#%%
# load linear model and the corresponding test set
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

with open ('linear_model.pkl', 'rb') as file:
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
# load model and X0_test y0_test
import pickle
model = pickle.load(open('./models/model.pkl', 'rb'))

X0_test = pickle.load(open('./data/X0_test.pkl','rb'))
y0_test = pickle.load(open('./data/y0_test.pkl','rb'))

model.predict(X0_test)[:5]

# %%
y0_test[:5]

# %%
model.score(X0_test,y_test)
# %%
