# 텍스트마이닝 데이터 정제

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from os import path
import re
import numpy as np
from PIL import Image

usa_mask = np.array(Image.open("c:/project/usa_im.png"))

script = input( 'input file name : ')

d = path.dirname("c:/project/")
text = open(path.join(d, "%s"%script), mode="r", encoding="utf-8").read()
file = open('c:/project/word.txt', 'r', encoding = 'utf-8')
word = file.read().split(' ')
for i in word:
    text = re.sub(i,'',text)

wordcloud = WordCloud(font_path='C://Windows//Fonts//gulim',
                      stopwords=STOPWORDS, 
                      max_words=1000,
                      background_color='white',
                      max_font_size = 100,
                      min_font_size = 1,
                      mask = usa_mask,
                      colormap='jet').generate(text).to_file('c:/project/cnn_cloud.png')
plt.figure(figsize=(15,15)) 
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

text.close()
file.close()
