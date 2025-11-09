import tkinter as tk
import random
import threading
import time

def dow():
    window = tk.Tk()
    width, height = window.winfo_screenwidth(), window.winfo_screenheight()
    a = random.randrange(0, width)
    b = random.randrange(0, height)
    window.title("温馨提示")
    window.geometry("220x50+" + str(a) + "+" + str(b))
    # 随机提示文字
    tips = [
        '多喝水哦~', '保持微笑呀', '每天都要元气满满',
        '记得吃水果', '保持好心情', '好好爱自己', '我想你了',
        '梦想成真', '期待下一次见面', '金榜题名',
        '顺顺利利', '早点休息', '愿所有烦恼都消失',
        '别熬夜', '今天过得开心嘛', '天冷了，多穿衣服'
    ]
    tip = random.choice(tips)
    # 随机背景颜色
    bg_colors = [
        'lightpink', 'skyblue', 'lightgreen', 'lavender',
        'lightyellow', 'plum', 'coral', 'bisque', 'aquamarine',
        'mistyrose', 'honeydew', 'lavenderblush', 'oldlace'
    ]
    bg = random.choice(bg_colors)
    tk.Label(
        window,
        text = tip,
        bg = bg,
        font = ('微软雅黑', 18),
        width = 25,
        height = 2
    ).pack()
    window.mainloop()

threads = []
for i in range(30):
    t = threading.Thread(target=dow)
    threads.append(t)
    time.sleep(0.05)
    threads[i].start()


















