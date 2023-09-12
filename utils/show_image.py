import matplotlib.pyplot as plt
# 展示一张图片和它对应的标注点作为例子
def show_image(image,title = None):
    """显示带有地标的图片"""
    plt.imshow(image)
    if title is not None:
        plt.title(f"title: {title}")
    plt.axis('off')  # 关闭坐标轴
    plt.savefig('sample_plot.png') 
    # plt.show()
    # print("show image")
