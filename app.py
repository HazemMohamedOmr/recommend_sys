from flask import Flask, request, jsonify
from flask_restful import Api, Resource

import pandas as pd
import numpy as np
import json
import tensorflow as tf
from joblib import load
from sklearn.model_selection import train_test_split  # I just need scikit-learn to be installed

app = Flask(__name__)
api = Api(app)

database_path = './database/database.csv'

model = tf.keras.models.load_model("recommend_sys")

scalerItem = load('./Scaling/item_scaler.bin')
scalerUser = load('./Scaling/user_scaler.bin')
scalerTarget = load('./Scaling/target_scaler.bin')


# TODO def function : Connect Database
# TODO def function : Transform Product

class Predict(Resource):
    def post(self):
        try:
            user = request.json
            user = list(user.values())

            user_vec = np.array(user)

            item_vecs = pd.read_csv(database_path)  # TODO Change to DB Instance
            item_vecs = np.array(item_vecs)

            user_vecs = np.tile(user_vec, (len(item_vecs), 1))
            # scale our user and item vectors
            suser_vecs = scalerUser.transform(user_vecs[:, 3:])
            sitem_vecs = scalerItem.transform(item_vecs[:, 5:])
            y_p = model.predict([suser_vecs, sitem_vecs])
            y_pu = scalerTarget.inverse_transform(y_p)
            # yyy = y_pu * y_pu

            sorted_index = np.argsort(-y_pu, axis=0).reshape(-1).tolist()  # negate to get largest rating first
            sorted_ypu = y_pu[sorted_index]
            sorted_items = item_vecs[sorted_index]
            sorted_ypuDF = pd.DataFrame(sorted_ypu)
            sorted_itemsDF = pd.DataFrame(sorted_items)

            sorted_itemsDF.rename(
                columns={0: "asin", 1: "userID", 2: "rating", 3: "Unnamed: 0.1", 4: "category", 5: "ratingCount",
                         6: "ratingAvg", 7: "pants", 8: "jeans", 9: "shirt", 10: "t-shirt", 11: "jacket", 12: "coat",
                         13: "hoodies", 14: "sweatshirts", 15: "blazer", 16: "sneaker", 17: "boot", 18: "oxford",
                         19: "blouseClean", 20: "skirtClean", 21: "tie"}, inplace=True)

            result = np.array(sorted_itemsDF)
            # result = sorted_itemsDF.T.to_json()
            # result2 = sorted_itemsDF.loc[:10, :].T.to_json()
            # print(result2)
            return result.tolist()
        except Exception as e:
            return {'error': str(e)}, 400


class AddProduct(Resource):
    def post(self):  # TODO Change all the post method logic
        try:
            row = request.json
            row = pd.DataFrame(data=[row])

            item_vecs = pd.read_csv(database_path)
            item_vecs = pd.concat([item_vecs, row], ignore_index=True)

            item_vecs.to_csv('./database/database.csv', index=False, mode='w')

            test = pd.read_csv(database_path)
            print(test.iloc[-5:, :])
            return 'success', 200

        except Exception as e:
            return {'error': str(e)}, 400


class Test(Resource):
    def get(self):
        return 'tested Successfully! yes!!', 200


    # APIs EndPoints
api.add_resource(Predict, '/predict')
api.add_resource(AddProduct, '/add_product')
api.add_resource(Test, '/test')

if __name__ == '__main__':
    # app.config['ENV'] = 'development'
    # app.config['DEBUG'] = True
    # app.config['TESTING'] = True
    # app.run(debug=True)
    app.run()

# @app.route('/')
# def hello_world():  # put application's code here
#     return 'Hello World!'
#
