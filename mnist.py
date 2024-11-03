import os
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image
import numpy as np

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
image_size = 28

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.secret_key = 'a_very_secret_random_string_here'  # flashを使うためにSECRET_KEYが必要

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model = load_model('./model.keras')  # 学習済みモデルをロード

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # ファイルがリクエストに含まれているかを確認
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        
        file = request.files['file']

        # ファイル名が空かどうかを確認
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        
        # ファイルが有効かどうかを確認
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            # 画像を読み込んで予測
            img = image.load_img(filepath, color_mode='grayscale', target_size=(image_size, image_size))
            img = image.img_to_array(img)
            data = np.array([img])
            result = model.predict(data)[0]
            predicted = result.argmax()
            pred_answer = "これは " + classes[predicted] + " です"
            return render_template("index.html", answer=pred_answer)
        
        flash('許可されていないファイル形式です')
        return redirect(request.url)

    return render_template("index.html", answer="")

#--- 内部テスト用
"""
if __name__ == "__main__":
    app.run()
"""

#--- 公開用
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)
