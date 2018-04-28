"""
Creating WordCloud images
-------------------------

A wrapper to wordcloud library to create images from files or directories
"""
from wordcloud import WordCloud
import os

from PIL import ImageDraw


def wordcloud_to_file(aggregated, output_name, directory_path="./output/"):
    wordcloud = WordCloud(width=1200, height=600).generate_from_frequencies(aggregated)
    img = wordcloud.to_image()
    draw = ImageDraw.Draw(img)
    # font = ImageFont.truetype("Arial.ttf", 16)
    draw.text((0, 0), output_name, (255, 255, 255))
    img.save(directory_path + output_name + ".png")
    return img


def wordclouds_to_repository(names_and_occurences, directory_path="./output/"):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return [wordcloud_to_file(aggregated[1], aggregated[0], directory_path)
            for i, aggregated in enumerate(names_and_occurences)]
