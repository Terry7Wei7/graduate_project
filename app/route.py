# app/route.py

from flask import render_template, Response, jsonify, request, send_file
import cv2
import os
import subprocess
import uuid
import time
import glob
import torch
import random
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# ------------- 常量及路径配置 -------------
RTSP_URL1 = "rtsp://root:pass@140.116.185.213:9663/axis-media/media.amp"
RTSP_URL2 = "rtsp://root:pass@140.116.185.213:9664/axis-media/media.amp"
RTSP_URL3 = "rtsp://terrywei:07150715@192.168.0.57:554/stream1"
HTTP_URL1 = "http://192.168.0.246:8889/cam1/"
RTSP_URL4 = "rtsp://192.168.0.246:8554/cam1"
UPLOAD_FOLDER = './uploads'
PANORAVA_UPLOAD_FOLDER = './panorama_uploads'
RESULT_FOLDER = '/home/neat/Desktop/project/runs/detect'
DETECT_RESULT_FILE = "detect_result.txt"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PANORAVA_UPLOAD_FOLDER, exist_ok=True)
# ------------- 加载 Ultralytics YOLO 模型 -------------
model = YOLO("yolo11x.pt")  # 加载 YOLOv8/YOLOv11权重 (这里文件名叫 "yolo11x.pt")
class_names = model.names   # 模型的类别映射表 (字典或列表)

# 生成颜色列表，用于画框
colors = [
    [random.randint(0, 255) for _ in range(3)]
    for _ in range(len(class_names))
]

def draw_boxes(img_bgr, results, class_names, colors):
    """
    在图像上绘制YOLO的检测框与标签
    results: Ultralytics的 predict() 结果 (List[Batch])
             每个 Batch 对象包含 .boxes
    """
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])   # 边框坐标
            conf = float(box.conf[0])               # 置信度
            cls = int(box.cls[0])                   # 类别ID
            label = f'{class_names[cls]} {conf:.2f}'
            color = colors[cls % len(colors)]
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img_bgr, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
    return img_bgr

def gen_detect_frames(source):
    """
    读取影像来源 -> YOLO模型.predict(frame) -> 画框 -> 以 MJPEG 流形式返回
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"无法连接到 {source}")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 使用官方 model.predict() 进行YOLO推理
        results = model.predict(frame)

        # 调用 draw_boxes() 绘制检测框
        annotated_frame = draw_boxes(frame, results, class_names, colors)

        # 编码成JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        # 通过yield返回multipart格式
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

def gen_frames(source):
    """
    无检测的普通推流：将视频帧直接输出为MJPEG
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"无法连接到 {source}")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

# ---------- 以下为其它路由 ----------
def index():
    # 渲染 templates/index.html
    return render_template('index.html')

def cctv():
    # 渲染 templates/index.html
    return render_template('cctv.html')

def transform():
    # 渲染 templates/index.html
    return render_template('transform.html')

def video_feed1():
    return Response(
        gen_frames(RTSP_URL1),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

def video_feed2():
    return Response(
        gen_frames(RTSP_URL2),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

def video_feed3():
    return Response(
        gen_frames(RTSP_URL3),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

def webcam_feed():
    """本地摄像头的 MJPEG 串流"""
    return Response(
        gen_frames(0),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

def picam_feed():
    """本地摄像头的 MJPEG 串流"""
    return Response(
        gen_frames(HTTP_URL1),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

def picam_feed2():
    """本地摄像头的 MJPEG 串流"""
    return Response(
        gen_frames(RTSP_URL4),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

def video_detect_feed3():
    """
    带YOLO推理的RTSP
    """
    return Response(
        gen_detect_frames(RTSP_URL3),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

def result():
    """返回 detect_result.txt 里的识别结果"""
    if not os.path.exists(DETECT_RESULT_FILE) or os.path.getsize(DETECT_RESULT_FILE) == 0:
        return jsonify({'error': 'Detection result not available'}), 500

    with open(DETECT_RESULT_FILE, "r") as f:
        return jsonify(f.read()), 200
def predict_p():
    # 获取所有上传的文件（前端 input 标签应设置 multiple 属性）
    files = request.files.getlist('file')
    if not files or len(files) == 0:
        return jsonify({'error': 'No file part'}), 400

    # 判断文件类型，假设所有文件类型一致
    first_file_ext = files[0].filename.rsplit('.', 1)[-1].lower()

    panorama_file = None

    # 视频模式：仅允许单个视频文件
    if first_file_ext in ['mp4', 'mov']:
        if len(files) > 1:
            return jsonify({'error': 'Please upload only one video file.'}), 400
        file = files[0]
        file_name = f"{uuid.uuid4().hex}.{first_file_ext}"
        file_path = os.path.join(UPLOAD_FOLDER, file_name)
        file.save(file_path)
        panorama_file = video_to_panorama(file_path)
    # 图片模式：允许上传多张图片，至少两张
    elif first_file_ext in ['jpg', 'jpeg', 'png']:
        if len(files) < 2:
            return jsonify({'error': 'At least two images are required for panorama stitching.'}), 400
        image_paths = []
        for file in files:
            file_ext = file.filename.rsplit('.', 1)[-1].lower()
            if file_ext not in ['jpg', 'jpeg', 'png']:
                continue
            file_name = f"{uuid.uuid4().hex}.{file_ext}"
            file_path = os.path.join(UPLOAD_FOLDER, file_name)
            file.save(file_path)
            image_paths.append(file_path)
        panorama_file = images_to_panorama(image_paths)
    else:
        return jsonify({'error': 'Unsupported file type.'}), 400

    if not panorama_file:
        return jsonify({'error': 'Panorama generation failed'}), 500

    # 生成全景图成功后，调用物件检测
    weights_path = 'yolov8x.pt'
    conf_value = '0.25'
    img_size = '1280'

    try:
        subprocess.run(
            [
                'yolo', 'detect', 'predict',
                f'model={weights_path}',
                f'source={panorama_file}',
                f'conf={conf_value}',
                f'imgsz={img_size}',
                'save_txt=True',       # 不保存文本标签
                'project=runs/detect',
                'name=panorama_detect', # 检测结果存放的子文件夹名称
                'show_conf=False',
                'exist_ok=True'
            ],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        return jsonify({'error': f'Detection on panorama failed: {e.stderr}'}), 500

    # 解析最新的 `labels/*.txt` 并覆盖写入 `detect_result.txt`
    labels_folder = os.path.join(RESULT_FOLDER, "panorama_detect", "labels")
    parse_latest_label_and_save(DETECT_RESULT_FILE, labels_folder)

    # 获取最新检测结果文件（这里假设检测生成的结果图片在 RESULT_FOLDER 下，文件夹名包含 "panorama_detect"）
    predict_folders = glob.glob(f"{RESULT_FOLDER}/panorama_detect*")
    if not predict_folders:
        return jsonify({'error': 'Detection result folder not found'}), 500

    latest_result_folder = max(predict_folders, key=os.path.getmtime)
    result_files = glob.glob(f"{latest_result_folder}/*")
    if not result_files:
        return jsonify({'error': 'No output files found in panorama_detect/'}), 500

    result_file_path = max(result_files, key=os.path.getmtime)

    # 等待文件写入完毕（最多等待5秒）
    max_wait_time = 5
    wait_time = 0
    while not os.path.exists(result_file_path) or os.path.getsize(result_file_path) == 0:
        time.sleep(1)
        wait_time += 1
        if wait_time >= max_wait_time:
            return jsonify({'error': 'Result file not found or still being written'}), 500

    # 返回检测后的全景图（图片）
    return send_file(result_file_path, mimetype='image/jpeg', as_attachment=False)

def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_ext = file.filename.rsplit('.', 1)[-1].lower()
    if file_ext not in ['jpg', 'jpeg', 'png', 'mp4', 'mov']:
        return jsonify({'error': 'Unsupported file type'}), 400

    file_name = f"{uuid.uuid4().hex}.{file_ext}"
    file_path = os.path.join(UPLOAD_FOLDER, file_name)
    file.save(file_path)

    weights_path = 'yolov8x.pt'
    conf_value = '0.25'
    img_size = '1280'

    try:
        subprocess.run(
            [
                'yolo', 'detect', 'predict',
                f'model={weights_path}',
                f'source={file_path}',
                f'conf={conf_value}',
                f'imgsz={img_size}',
                'save_txt=True',  # 强制保存 labels/*.txt
                'project=runs/detect',
                'name=predict',
                'show_conf=False',
                'exist_ok=True'
            ],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        return jsonify({'error': f'Detection failed: {e.stderr}'}), 500

    # 解析最新的 `labels/*.txt` 并覆盖写入 `detect_result.txt`
    labels_folder = os.path.join(RESULT_FOLDER, "predict", "labels")
    parse_latest_label_and_save(DETECT_RESULT_FILE, labels_folder)

    # 获取最新推理结果文件
    predict_folders = glob.glob(f"{RESULT_FOLDER}/predict*")
    if not predict_folders:
        return jsonify({'error': 'Result folder not found'}), 500

    latest_result_folder = max(predict_folders, key=os.path.getmtime)
    result_files = glob.glob(f"{latest_result_folder}/*")

    if not result_files:
        return jsonify({'error': 'No output files found in predict/'}), 500

    result_file_path = max(result_files, key=os.path.getmtime)

    # 等待文件写入完毕
    max_wait_time = 5
    wait_time = 0
    while not os.path.exists(result_file_path) or os.path.getsize(result_file_path) == 0:
        time.sleep(1)
        wait_time += 1
        if wait_time >= max_wait_time:
            return jsonify({'error': 'Result file not found or still being written'}), 500

    # 返回图片或视频
    if file_ext == 'mp4':
        return send_file(result_file_path, mimetype='video/mp4', as_attachment=False)
    elif file_ext == 'mov':
        return send_file(result_file_path, mimetype='video/mov', as_attachment=False)
    else:
        return send_file(result_file_path, mimetype='image/jpeg', as_attachment=False)

def parse_latest_label_and_save(detect_result_file, labels_folder):
    """解析 `labels/` 目录中最新的 .txt 并覆盖写入 detect_result.txt"""
    labels_files = glob.glob(f"{labels_folder}/*.txt")

    # 如果 labels/ 为空，则写入 "No objects detected"
    if not labels_files:
        with open(detect_result_file, "w") as f:
            f.write("No objects detected")
        return

    # 获取最新的 .txt 文件
    latest_label_file = max(labels_files, key=os.path.getmtime)

    detection_results = {}

    # 读取最新的 .txt 统计类别 ID
    with open(latest_label_file, "r") as f:
        for line in f:
            class_id = line.split()[0]  # 读取类别 ID
            detection_results[class_id] = detection_results.get(class_id, 0) + 1

    # 格式化检测结果
    formatted_results = []
    for class_id, count in detection_results.items():
        class_name = class_names.get(int(class_id), f"Class {class_id}")
        formatted_results.append(f"{class_name}: {count}")

    final_result = ", ".join(formatted_results)

    # 覆盖写入 detect_result.txt
    with open(detect_result_file, "w") as f:
        f.write(final_result)


# 上传接口，支持视频或多张图片拼接
def upload_to_panorama():
    # 获取所有上传的文件（前端 input 标签应设置 multiple 属性）
    files = request.files.getlist('file')
    if not files or len(files) == 0:
        return jsonify({'error': 'No file part'}), 400

    # 检查第一个文件扩展名，假设所有文件类型一致
    first_file_ext = files[0].filename.rsplit('.', 1)[-1].lower()

    # 视频模式：仅允许单个视频文件
    if first_file_ext in ['mp4', 'mov']:
        if len(files) > 1:
            return jsonify({'error': 'Please upload only one video file.'}), 400
        file = files[0]
        file_name = f"{uuid.uuid4().hex}.{first_file_ext}"
        file_path = os.path.join(PANORAVA_UPLOAD_FOLDER, file_name)
        file.save(file_path)
        panorama_file = video_to_panorama(file_path)
        if not panorama_file:
            return jsonify({'error': 'Panorama generation failed'}), 500
        return send_file(panorama_file, mimetype='image/jpeg', as_attachment=False)

    # 图片模式：允许上传多张图片，至少两张
    elif first_file_ext in ['jpg', 'jpeg', 'png']:
        if len(files) < 2:
            return jsonify({'error': 'At least two images are required for panorama stitching.'}), 400
        image_paths = []
        for file in files:
            file_ext = file.filename.rsplit('.', 1)[-1].lower()
            if file_ext not in ['jpg', 'jpeg', 'png']:
                continue
            file_name = f"{uuid.uuid4().hex}.{file_ext}"
            file_path = os.path.join(PANORAVA_UPLOAD_FOLDER, file_name)
            file.save(file_path)
            image_paths.append(file_path)
        panorama_file = images_to_panorama(image_paths)
        if not panorama_file:
            return jsonify({'error': 'Panorama generation failed'}), 500
        return send_file(panorama_file, mimetype='image/jpeg', as_attachment=False)

    else:
        return jsonify({'error': 'Unsupported file type.'}), 400


def video_to_panorama(video_path, 
                      output_directory='/home/neat/Desktop/project/panorama_results', 
                      frame_interval=15, 
                      resize_factor=0.5, 
                      output_size=(1920, 1080), 
                      high_res_size=(3840, 2160), 
                      timeout=30):
    # 尝试打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件:", video_path)
        return None

    start_time = time.time()
    frames = []
    frame_count = 0

    # 提取视频帧
    while True:
        # 检查超时
        if time.time() - start_time > timeout:
            print("提取帧超时")
            cap.release()
            return None

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # 每隔 frame_interval 帧提取一帧
        if frame_count % frame_interval == 0:
            # 缩小图像尺寸，节省后续拼接时间和内存
            new_width = int(frame.shape[1] * resize_factor)
            new_height = int(frame.shape[0] * resize_factor)
            frame_resized = cv2.resize(frame, (new_width, new_height))
            frames.append(frame_resized)

    cap.release()

    if len(frames) < 2:
        print("视频中没有足够的帧进行拼接")
        return None

    # 拼接全景图
    stitcher = cv2.Stitcher_create()
    start_time = time.time()
    status, panorama = stitcher.stitch(frames)
    if time.time() - start_time > timeout:
        print("全景拼接超时")
        return None

    if status != cv2.Stitcher_OK:
        print(f"全景拼接失败，错误代码：{status}")
        return None

    # 修补黑边：将拼接结果转换为灰度图并生成掩码，再用 inpaint 填补黑边区域
    start_time = time.time()
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)
    panorama_filled = cv2.inpaint(panorama, mask, 3, cv2.INPAINT_TELEA)
    if time.time() - start_time > timeout:
        print("黑边修补超时")
        return None

    if panorama_filled is None:
        print("黑边修补失败")
        return None

    # 调整全景图尺寸
    result_resized = cv2.resize(panorama_filled, output_size)
    # 如果需要高分辨率输出，进一步放大
    result_high_res = cv2.resize(result_resized, high_res_size, interpolation=cv2.INTER_CUBIC)

    if result_high_res is None:
        print("图像尺寸调整失败")
        return None

    # 保存生成的全景图
    os.makedirs(output_directory, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_directory, f'panorama_high_res_{timestamp}.jpg')

    if not cv2.imwrite(output_path, result_high_res):
        print("无法保存全景图")
        return None

    print(f"高解析度全景图已保存到：{output_path}")
    return output_path

# 图片转全景图函数（针对多张图片）
def images_to_panorama(image_paths, 
                       output_directory='/home/neat/Desktop/project/panorama_results', 
                       timeout=30, 
                       output_size=(1920, 1080), 
                       high_res_size=(3840, 2160)):
    frames = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            frames.append(img)
    if len(frames) < 2:
        print("没有足够的图片进行拼接")
        return None

    stitcher = cv2.Stitcher_create()
    start_time = time.time()
    status, panorama = stitcher.stitch(frames)
    if time.time() - start_time > timeout:
        print("全景拼接超时")
        return None

    if status != cv2.Stitcher_OK:
        print(f"全景拼接失败，错误代码：{status}")
        return None

    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)
    panorama_filled = cv2.inpaint(panorama, mask, 3, cv2.INPAINT_TELEA)
    if panorama_filled is None:
        print("黑边修补失败")
        return None

    result_resized = cv2.resize(panorama_filled, output_size)
    result_high_res = cv2.resize(result_resized, high_res_size, interpolation=cv2.INTER_CUBIC)
    if result_high_res is None:
        print("图像尺寸调整失败")
        return None

    os.makedirs(output_directory, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_directory, f'panorama_high_res_{timestamp}.jpg')

    if not cv2.imwrite(output_path, result_high_res):
        print("无法保存全景图")
        return None

    print(f"高解析度全景图已保存到：{output_path}")
    return output_path


"""
def uploadvideo_to_panorama():
    # 检查是否有文件上传
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # 只接受 mp4 格式的视频
    file_ext = file.filename.rsplit('.', 1)[-1].lower()
    if file_ext not in ['mp4', 'mov']:
        return jsonify({'error': 'Unsupported file type. Only mp4 is allowed for panorama generation.'}), 400

    # 保存上传的文件
    file_name = f"{uuid.uuid4().hex}.{file_ext}"
    file_path = os.path.join(PANORAVA_UPLOAD_FOLDER, file_name)
    file.save(file_path)

    # 调用全景生成函数
    panorama_file = video_to_panorama(file_path)
    if not panorama_file:
        return jsonify({'error': 'Panorama generation failed'}), 500

    # 返回生成的全景图 (图片)
    return send_file(panorama_file, mimetype='image/jpeg', as_attachment=False)
"""

"""
# app/route.py
from flask import render_template, Response ,jsonify, request, send_file
import cv2
import os
import subprocess
import uuid
import time
import glob
import torch
import random
import numpy as np
from ultralytics import YOLO 
RTSP_URL1 = "rtsp://root:pass@140.116.185.213:9663/axis-media/media.amp"
RTSP_URL2 = "rtsp://root:pass@140.116.185.213:9664/axis-media/media.amp"
RTSP_URL3 = "rtsp://terrywei:07150715@192.168.0.57:554/stream1"
UPLOAD_FOLDER = './uploads'
RESULT_FOLDER = '/home/neat/Desktop/project/runs/detect'
DETECT_RESULT_FILE = "detect_result.txt"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# **加载 YOLOv8v11 模型，获取类别名称**
model = YOLO("yolo11x.pt")  # **加载模型**
class_names = model.names  # **自动获取类别映射**（字典格式）
# ======== YOLOv8v11 Detector 類別 ========
class YOLOv11Detector:
    def __init__(self, weights='yolov11x.pt', device='cuda', conf_thres=0.25, iou_thres=0.45, img_size=640):
        self.device = device
        self.model = YOLO(weights).to(self.device)  # 加載 YOLOv11 模型
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = img_size
    
    def detect(self, img_bgr):
        #單張 BGR 圖片 -> YOLOv8 推論 -> (xyxy, conf, class) 結果 
        results = self.model(img_bgr, conf=self.conf_thres, iou=self.iou_thres)
        return results

    def draw_boxes(self, img_bgr, results):
        #在原圖上繪製 bounding box 與標籤 
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f'{class_names[cls]} {conf:.2f}'
                color = colors[cls % len(colors)]
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img_bgr, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return img_bgr
# 生成顏色用於畫框
colors = [[random.randint(0,255) for _ in range(3)] for _ in range(len(class_names))]

def gen_detect_frames(source):
    #讀取影像來源 -> YOLOv8 推論 -> 畫框 -> 轉 MJPEG 回傳
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"無法連線至 {source}")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # YOLOv8 推論
        results = model.detect(frame)
        # 繪製推論結果
        annotated_frame = model.draw_boxes(frame, results)
        
        # 轉成 JPEG bytes
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        # 透過 yield 回傳 multipart 格式
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
def gen_frames(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"無法連線至 {source}")
        return
    while True:
        success, frame = cap.read()
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

def index():
    # 渲染 templates/index.html
    return render_template('index.html')
def video_feed1():
    return Response(gen_frames(RTSP_URL1),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
def video_feed2():
    return Response(gen_frames(RTSP_URL2),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
def video_feed3():
    return Response(gen_frames(RTSP_URL3),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
def webcam_feed():
    #本地摄像头的 MJPEG 串流
    return Response(gen_frames(0),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
def video_detect_feed3():
    return Response(gen_detect_frames(RTSP_URL3),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
def result():
    #返回 detect_result.txt 里的识别结果
    if not os.path.exists(DETECT_RESULT_FILE) or os.path.getsize(DETECT_RESULT_FILE) == 0:
        return jsonify({'error': 'Detection result not available'}), 500

    with open(DETECT_RESULT_FILE, "r") as f:
        return jsonify(f.read()), 200
    
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_ext = file.filename.rsplit('.', 1)[-1].lower()
    if file_ext not in ['jpg', 'jpeg', 'png', 'mp4', 'mov']:
        return jsonify({'error': 'Unsupported file type'}), 400

    file_name = f"{uuid.uuid4().hex}.{file_ext}"
    file_path = os.path.join(UPLOAD_FOLDER, file_name)
    file.save(file_path)

    weights_path = 'yolov8x.pt'
    conf_value = '0.25'
    img_size = '1280'

    try:
        subprocess.run(
            [
                'yolo', 'detect', 'predict',
                f'model={weights_path}',
                f'source={file_path}',
                f'conf={conf_value}',
                f'imgsz={img_size}',
                'save_txt=True',  # **强制保存 labels/*.txt**
                'project=runs/detect',
                'name=predict',
                'show_conf=False',
                'exist_ok=True'
            ],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        return jsonify({'error': f'Detection failed: {e.stderr}'}), 500

    # **解析最新的 `labels/*.txt` 并覆盖写入 `detect_result.txt`**
    labels_folder = os.path.join(RESULT_FOLDER, "predict", "labels")
    parse_latest_label_and_save(DETECT_RESULT_FILE, labels_folder)

    # **获取最新推理结果文件**
    predict_folders = glob.glob(f"{RESULT_FOLDER}/predict*")
    if not predict_folders:
        return jsonify({'error': 'Result folder not found'}), 500

    latest_result_folder = max(predict_folders, key=os.path.getmtime)
    result_files = glob.glob(f"{latest_result_folder}/*")

    if not result_files:
        return jsonify({'error': 'No output files found in predict/'}), 500

    result_file_path = max(result_files, key=os.path.getmtime)

    # **确保 YOLOv8 结果已写入**
    max_wait_time = 5
    wait_time = 0
    while not os.path.exists(result_file_path) or os.path.getsize(result_file_path) == 0:
        time.sleep(1)
        wait_time += 1
        if wait_time >= max_wait_time:
            return jsonify({'error': 'Result file not found or still being written'}), 500

    # **返回图片或视频**
    if file_ext == 'mp4':
        return send_file(result_file_path, mimetype='video/mp4', as_attachment=False)
    elif file_ext == 'mov':
        return send_file(result_file_path, mimetype='video/mov', as_attachment=False)
    else:
        return send_file(result_file_path, mimetype='image/jpeg', as_attachment=False)
    
def parse_latest_label_and_save(detect_result_file, labels_folder):
    #解析 `labels/` 目录中最新的 .txt 并覆盖写入 detect_result.txt
    labels_files = glob.glob(f"{labels_folder}/*.txt")

    # **如果 labels/ 为空，则写入 "No objects detected"**
    if not labels_files:
        with open(detect_result_file, "w") as f:
            f.write("No objects detected")
        return

    # **获取最新的 .txt 文件**
    latest_label_file = max(labels_files, key=os.path.getmtime)

    detection_results = {}

    # **读取最新的 .txt 统计类别 ID**
    with open(latest_label_file, "r") as f:
        for line in f:
            class_id = line.split()[0]  # 读取类别 ID
            detection_results[class_id] = detection_results.get(class_id, 0) + 1

    # **格式化检测结果**
    formatted_results = []
    for class_id, count in detection_results.items():
        class_name = class_names.get(int(class_id), f"Class {class_id}")  # **动态获取类别名称**
        formatted_results.append(f"{class_name}: {count}")

    final_result = ", ".join(formatted_results)

    # **覆盖写入 detect_result.txt**
    with open(detect_result_file, "w") as f:
        f.write(final_result)
    
    """