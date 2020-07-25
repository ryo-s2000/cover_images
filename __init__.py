import json
import base64
import logging
import sys

from flask import Flask, render_template, request

# 配列平坦化
import itertools

# 数値計算
import numpy as np

# 画像を行列に変換
from PIL import Image

# 訓練データ
import sklearn

# s3からデータを取得するのに使用
import boto3
from boto3.session import Session

# バイナリデータを画像として読み込み
from io import BytesIO

# パス操作
import os

# エラー処理
import traceback

# ファイル圧縮
import tarfile

# 学習済みモデル保存
import pickle

# configを読み込む
import config

# aws関連の定数
AWS_S3_BUCKET_NAME = config.AWS_S3_BUCKET_NAME
S3_ACCESS_KEY_ID = config.S3_ACCESS_KEY_ID
S3_SECRET_ACCESS_KEY = config.S3_SECRET_ACCESS_KEY
AWS_S3_PREFIX_MODEL_DATA = config.AWS_S3_PREFIX_MODEL_DATA

def create_app(test_config=None):
    app = Flask(__name__)

    @app.route("/")
    def main():
        session = Session(aws_access_key_id=S3_ACCESS_KEY_ID,
                    aws_secret_access_key=S3_SECRET_ACCESS_KEY,
                    region_name='ap-northeast-1')
        s3 = session.resource('s3')
        bucket = s3.Bucket(AWS_S3_BUCKET_NAME)

        models = []
        for object in bucket.objects.filter(Prefix=AWS_S3_PREFIX_MODEL_DATA).all():
            models.append(object.key)

        return render_template('index.html', models=models)

    @app.route('/send', methods=['POST'])
    def send():
        # 入力データを受け取る
        file = request.files['img_file']
        aws_s3_learning_model = request.form["model"]

        if file.filename == '':
            return render_template('err.html', err_message='推論したい書影データを選択してください。')
        extension = ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']
        if not file.filename.split('.')[-1] in extension:
            return render_template('err.html', err_message="ファイル拡張子は{}でお願いします。".format(extension))
        if aws_s3_learning_model == '':
            return render_template('err.html', err_message='学習済みモデルを選択してください')

        img = Image.open(file).resize((int(aws_s3_learning_model.split('/')[1].split('-')[-2]), int(aws_s3_learning_model.split('/')[1].split('-')[-1])))
        width, height = img.size
        image_matrix = [np.array([[img.getpixel((w,h))[0:3] for w in range(width)] for h in range(height)])]
        # 画像ファイルの色情報を平坦化
        tmp_data = [[list(itertools.chain.from_iterable(image_matrix[i][j])) for j in range(image_matrix[i].shape[0])] for i in range(len(image_matrix))]
        data = np.array([list(itertools.chain.from_iterable(tmp_data[i])) for i in range(len(image_matrix))])
        filename = file.filename.split('_')[0]

        cover_images = sklearn.utils.Bunch()
        cover_images.data = data
        cover_images.target = [filename]

        # 学習済モデルダウンロード
        s3 = boto3.client('s3',
            aws_access_key_id=S3_ACCESS_KEY_ID,
            aws_secret_access_key=S3_SECRET_ACCESS_KEY,
            region_name='ap-northeast-1'
        )

        # 学習済モデルを解凍
        learning_model_dirpath = './learning_model/'+aws_s3_learning_model.split('/')[1]+'/'
        if not os.path.exists(learning_model_dirpath):
            s3.download_file(AWS_S3_BUCKET_NAME ,aws_s3_learning_model, 'model.tar.gz')
            os.mkdir(learning_model_dirpath)
            with tarfile.open('./model.tar.gz', mode='r:gz') as tar:
                for file in tar:
                    with tar.extractfile(file.name) as model:
                        with open(learning_model_dirpath+file.name, mode='wb') as f:
                            f.write(model.read())
            os.remove('./model.tar.gz')

        # 学習済モデルを読み込む
        lda = pickle.load(open(learning_model_dirpath+'lda.pickle', 'rb'))
        scaler = pickle.load(open(learning_model_dirpath+'scaler.pickle', 'rb'))
        X_scaler = pickle.load(open(learning_model_dirpath+'X_scaler.pickle', 'rb'))
        leaning_product_ids = pickle.load(open(learning_model_dirpath+'leaning_product_ids.pickle', 'rb'))

        # 画像データ
        X_train = cover_images.data
        input_product_ids = cover_images.target

        # モデルを適応
        X_train_lda = lda.transform(X_train)

        # 標準化
        X_train_scaler = scaler.transform(X_train_lda)

        # ユークリッド距離を計算して、近いと判断された画像を表示する
        close_image = calculation_euclidean_distance(X_scaler, X_train_scaler)

        # 表示するデータ数
        display_image = 4

        # テストデータから近い画像を表示
        result = []
        for i in range(X_train_scaler.shape[0]):
            result.append("https://www.dlsite.com/maniax/work/=/product_id/{}.html".format(input_product_ids[i]))
            for j in close_image[i][0:display_image]:
                result.append("https://www.dlsite.com/maniax/work/=/product_id/{}.html".format(leaning_product_ids[j[0]-3]))

        # 学習済みモデルを取得
        session = Session(aws_access_key_id=S3_ACCESS_KEY_ID,
                    aws_secret_access_key=S3_SECRET_ACCESS_KEY,
                    region_name='ap-northeast-1')
        s3 = session.resource('s3')
        bucket = s3.Bucket(AWS_S3_BUCKET_NAME)

        models = []
        for object in bucket.objects.filter(Prefix=AWS_S3_PREFIX_MODEL_DATA).all():
            models.append(object.key)

        return render_template('result.html', result=result, models=models, use_leaning_data=aws_s3_learning_model)

    @app.route("/about")
    def about():
        return render_template('about.html')

    @app.errorhandler(404)
    def not_found(error):
        return render_template('err.html', err_message='ページが見つかりませんでした。')

    def calculation_euclidean_distance(X_scaler, X_train_scaler):
        # ユークリッド距離を測定
        euclidean_distance = np.array([[{i:np.sum(pow(train-test, 2))} for i, train in enumerate(X_scaler)] for test in X_train_scaler])

        # 測定した距離を辞書型に変換
        euclidean_distance_dict_array = []
        for test_index in range(euclidean_distance.shape[0]):
            base = {}
            for train_index in range(euclidean_distance.shape[1]):
                base.update(euclidean_distance[test_index][train_index])
            euclidean_distance_dict_array.append(base)

        # 辞書型に変換された距離をソート
        sorted_euclidean_distance_dict_array = []
        for i in euclidean_distance_dict_array:
            sorted_euclidean_distance_dict_array.append(sorted(i.items(), key=lambda x:x[1]))

        return sorted_euclidean_distance_dict_array

    return app

    # if __name__ == '__main__':
    #     app.run(debug=True, host='0.0.0.0', port=8080, threaded=True)
