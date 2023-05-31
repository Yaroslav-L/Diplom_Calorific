import flask
from flask import Flask,request,render_template, redirect, send_file
from KCAL import main
import os
from PIL import Image
import base64 
import io
app = Flask(__name__)

@app.route("/home",methods=['POST','GET'])
def upload_image():
    if request.method == "POST":
        image = request.files['file']
        if image.filename == '':
            print("Обманка раскрыта")
            return redirect(request.url)
        img = Image.open(image)
        x = main(img)
        return render_template("index.html",name=x['name'],rez=x['KcaL'])
    return render_template("index.html")

@app.route("/api",methods=['POST','GET'])
def api():
    x = main(Image.open(request.files['image']))
    print(x)
    return x


def run_server_api():
    app.run(host='0.0.0.0', port=8000)
  
  
if __name__ == "__main__":     
    run_server_api()