# curl -X POST -F file=@/home/hyli/project/OCR/CRNN_Chinese_Characters_Rec/1.jpg http://119.3.49.107:6666/predict -v
# curl -X POST -F file=@/home/hyli/project/OCR/CRNN_Chinese_Characters_Rec/10.jpg http://192.168.2.203:5050/predict -v
/home/hyli/project/TensorRT-7.0.0.11/targets/x86_64-linux-gnu/bin/trtexec --onnx=CORNER-NEW-MERGE.onnx --batch=1 --saveEngine=lp_detect.engine