import streamlit as st
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageOps
import time
import os
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image

from model import Generator
# torch.backends.cudnn.enabled = False
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

st.set_page_config(
    page_title="웹툰속으로",
    page_icon="webtoon.png",
)
kakao_ad_code1 = """
 <ins class="kakao_ad_area" style="display:none;"
data-ad-unit = "DAN-sRI3oXPwhGF0v3f7"
data-ad-width = "250"
data-ad-height = "250"></ins>
<script type="text/javascript" src="//t1.daumcdn.net/kas/static/ba.min.js" async></script>
"""
kakao_ad_code2 = """
 <ins class="kakao_ad_area" style="display:none;"
data-ad-unit = "DAN-uovlWWrYb0j9mYVv"
data-ad-width = "250"
data-ad-height = "250"></ins>
<script type="text/javascript" src="//t1.daumcdn.net/kas/static/ba.min.js" async></script>
"""
coupang_ad_code="""
<iframe src="https://ads-partners.coupang.com/widgets.html?id=718831&template=carousel&trackingCode=AF3660738&subId=&width=680&height=140&tsource=" width="680" height="140" frameborder="0" scrolling="no" referrerpolicy="unsafe-url"></iframe>
<style>margin: 0 auto;</style>
"""
device = 'cpu'
    
net = Generator()
net.load_state_dict(torch.load('./weights/face_paint_512_v2.pt', map_location="cpu"))
net.to(device).eval()
def load_image(image_path, x32=True):
    img = Image.open(image_path).convert("RGB")

    if x32:
        def to_32s(x):
            return 256 if x < 256 else x - x % 32
        w, h = img.size
        img = img.resize((to_32s(w), to_32s(h)))

    return img
def main():
    
    st.title("_웹툰속으로_:cupid:")
    
    # 파일 업로드 섹션 디자인
    st.subheader('웹툰속으로에서는 단순한 클릭 한 번으로 당신의 사진을 독특하고 재미있는 웹툰 스타일로 변환해드립니다!:sunglasses:')
    st.write(':blue[얼굴 사진을 업로드 해주세요! 사진은 저장되지 않습니다!]')
    # 파일 업로드 컴포넌트
    uploaded_file = st.file_uploader("PNG 또는 JPG 이미지를 업로드하세요.", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        # 이미지를 넘파이 배열로 변환
        image = load_image(uploaded_file)
        image = ImageOps.exif_transpose(image)
        with torch.no_grad():
            image = to_tensor(image).unsqueeze(0) * 2 - 1
            out = net(image.to(device), False).cpu()
            out = out.squeeze(0).clip(-1, 1) * 0.5 + 0.5
            out = to_pil_image(out)
        with st.spinner('AI가 당신의 사진을 웹툰 스타일로 변환하고 있습니다...'):
            time.sleep(3)
            st.success('사진을 웹툰 스타일로 변환을 완료했습니다!')
        st.image(out,use_column_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.components.v1.html(f"<center>{kakao_ad_code1}</center>", height=250, scrolling=False)
    with col2:
        st.components.v1.html(f"<center>{kakao_ad_code2}</center>", height=250, scrolling=False)
    st.components.v1.html(coupang_ad_code, scrolling=False)
    st.markdown('<a target="_blank" href="https://icons8.com/icon/FB4OXbgFr65O/webtoon">Webtoon</a> icon by <a target="_blank" href="https://icons8.com">Icons8</a>', unsafe_allow_html=True)
if __name__ == "__main__":
    main()
