
# app/__init__.py
from flask import Flask
from app.route import index, cctv , transform, webcam_feed , video_feed1, video_feed2, video_feed3, video_detect_feed3 , picam_feed , picam_feed2 , result, predict ,predict_p, upload_to_panorama
def create_app():
    # 指定 templates、static 的搜寻路径
    # 因为 __init__.py 路径在 app/ 下，所以加 ../ 才能回到 PROJECT/ 下寻找
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static"
    )

    # 绑定 URL → Python 函数
    app.add_url_rule('/', 'index', index)
    # 绑定路由: GET /cctv -> cctv() 函数
    app.add_url_rule("/cctv", "cctv", cctv)
    # 绑定路由: GET /cctv -> cctv() 函数
    app.add_url_rule("/transform", "transform", transform)
    app.add_url_rule('/video_feed1', 'video_feed1', video_feed1)
    app.add_url_rule('/video_feed2', 'video_feed2', video_feed2)
    app.add_url_rule('/video_feed3', 'video_feed3', video_feed3)
    app.add_url_rule('/webcam_feed', 'webcam_feed', webcam_feed)
    app.add_url_rule('/picam_feed', 'picam_feed', picam_feed)
    app.add_url_rule('/picam_feed2', 'picam_feed2', picam_feed2)
    app.add_url_rule('/video_detect_feed3', 'video_detect_feed3', video_detect_feed3)
    app.add_url_rule('/result', 'result', result, methods=['POST'])
    app.add_url_rule('/predict', 'predict', predict, methods=['POST'])
    app.add_url_rule('/predict_p', 'predict_p', predict_p, methods=['POST'])
    app.add_url_rule('/uploadvideo_to_panorama', 'uploadvideo_to_panorama', upload_to_panorama, methods=['POST'])
    return app
