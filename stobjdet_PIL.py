
import io
import os
import streamlit as st
from PIL import Image,ImageDraw
from PIL import ImageFont

from google.cloud import vision

#from PIL import ImageFont

#textsize = 14 # 描画するテキストの大きさ。今回は14px。

# テキストの描画の準備。"arial.ttf"はフォント名。
#font = ImageFont.truetype(size=textsize)

#Google Visionの準備
 
#key.jsonのディレクトリー設定

#Google cloud API 認証情報
credential_path = 'key.json'

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
    img = Image.open(image_file.name)
    
    
    #img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #読み込み画像のサイズを取得
    height=img.height
    width=img.width
    #print(height,width)

    #画像サイズを均一化
    resize=1000
    re_h=re_w=resize/height
    img = img.resize((int(img.width*re_h) , int(img.height*re_w)), Image.LANCZOS)

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
            
            d = ImageDraw.Draw(img)
            
            #矩形の登録
            d.rectangle([(x1,y1),(x2,y2)],outline='red', width=5)
            #物体名と確信度の表示
            font_path='Kyokasho.ttc'
            font_size=35
            font = ImageFont.truetype(font_path,font_size)
            d.text((x1+5, int(y1+5)),object_.name,font=font,fill='red')
            
            
            
    st.image(img)
    for object_ in objects:
        st.subheader('{} (confidence: {}%)'.format(object_.name, int(object_.score*100)))
        
