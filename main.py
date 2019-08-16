from src.linear_regression import LinearRegression

model = LinearRegression()
model.load_data_from_csv("data.csv", y_col="last", remove_header=True)
model.train(nb_iter=300, learning_rate=0.05)
price = model.predict([24000])
print(price)
