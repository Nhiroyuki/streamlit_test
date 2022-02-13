
import io
import os
import cv2
import streamlit as st
from PIL import Image

from google.cloud import vision

#Google Visionの準備
 
#key.jsonのディレクトリー設定
base_dir = r'/Users/NakazawaHiroyuki/MyPython/画像認識/物体認識_Discription'

#Google cloud API 認証情報
credential_path = base_dir + r'/key.json'

#サービスアカウントキーへのパスを通す
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

client = vision.ImageAnnotatorClient()

#Streamlit ファイル読み込み

uploaded_file=st.file_uploader("Choose an image...",type='jpg')
if uploaded_file is not None:

    with open(uploaded_file.name, 'wb') as image_file:
            image_file.write(uploaded_file.read())
    remote_image_url = open(uploaded_file.name,'rb')
    remote_image = Image.open(uploaded_file)

    #st.image(remote_image,caption='Uploaded Image.',use_column_width=True)

    # Loads the image into memory
    with io.open(uploaded_file.name, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    #imgへ画像を読み込み        
    img = cv2.imread(image_file.name)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #読み込み画像のサイズを取得
    height, width, channels = img.shape[:3]
    #print(height,width)

    #画像サイズを均一化
    resize=1000
    re_h=re_w=resize/height
    img = cv2.resize(img,dsize=None, fx=re_h , fy=re_w)

    # Performs ObjectAnnotation on the image file

    objects = client.object_localization(
            image=image).localized_object_annotations
    
    st.subheader('Number of objects found: {}'.format(len(objects)))
    for object_ in objects:
        #st.write('\n{} (confidence: {})'.format(object_.name, object_.score))
        #print('Normalized bounding polygon vertices: ')
        
        for vertex in object_.bounding_poly.normalized_vertices:
            #st.write(' - ({}, {})'.format(vertex.x, vertex.y))
            
            #矩形座標取得
            x1=object_.bounding_poly.normalized_vertices[0].x
            y1=object_.bounding_poly.normalized_vertices[0].y
            x2=object_.bounding_poly.normalized_vertices[2].x
            y2=object_.bounding_poly.normalized_vertices[2].y
            
            #print(x1,y1)
            #print(x2,y2)
            
            #矩形座標のスケーリング
            x1=int(x1*width*re_h)
            x2=int(x2*width*re_h)
            y1=int(y1*height*re_h)
            y2=int(y2*height*re_h)
            
            #矩形の登録
            img=cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),3)
            
            #物体名と確信度の表示
            cv2.putText(img,
                text=object_.name,
                org=(x1+5, int((y1+y2)/2)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.2,
                color=(255, 255, 0),
                thickness=2,
                lineType=cv2.LINE_4)
            
    st.image(img)
    for object_ in objects:
        st.subheader('{} (confidence: {}%)'.format(object_.name, int(object_.score*100)))
        
