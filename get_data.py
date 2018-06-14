# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 16:33:13 2018

@author: lenovo
"""

from requests_html import HTMLSession
import numpy as np
import copy

#从目标网站获取目标信息
#url是网址， sel是selector选择相应的元素
def get_result(url, sel):
    session = HTMLSession()
    r = session.get(url)
    result = r.html.find(sel)
    
    return result

#将出彩信息去除'\n'并以list形式保存
def trans_to_list(result):
    string_result = []
    pick_result = []
    ball_result = []
    for i in range(len(result)):
        k = str(result[i].text)
        string_result.append(k)
        
    for j in range(len(string_result)):
        s = []
        a = ''
        for l in range(len(string_result[j])):
            if string_result[j][l] != '\n':
                a += string_result[j][l]
            elif string_result[j][l] == '\n':
                s.append(a)
                a = ''
        pick_result.append(s)
        
    for i in range(len(pick_result)):
        trans = pick_result[i][1:8]
        ball_result.append(trans)
        
    return ball_result

#将结果转化为1，0结构
def result_to_data(url, sel):
    data_result = []
    result = get_result(url, sel)
    ball_result = trans_to_list(result)
    for i in range(len(ball_result)):
        each_time_result = []
        for l in range(49):
            each_time_result.append(0)
        for j in range(49):
            if j <= 32:
                for k in range(len(ball_result[0])-1):
                    if j+1 == int(ball_result[i][k]):
                        each_time_result[j]=1
            elif  j > 32:
                for k in range(1,17):
                    if j-32 == int(ball_result[i][6]):
                        each_time_result[j] = 1
                        
        data_result.append(each_time_result)
    return data_result

def data_trans_to_array(result):
    data_x = copy.copy(result)
    data_x.remove(data_x[-1])
    data_x = np.array(data_x)
    data_y = copy.copy(result)
    data_y.remove(data_y[0])    
    data_y = np.array(data_y)

    return data_x, data_y
