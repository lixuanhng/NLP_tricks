"""
代码测试脚本
"""
import re
'''
# 测试1：tensorflow
import tensorflow as tf

print(tf.__version__)
a = tf.constant([1.0, 2.0], name='a', dtype=tf.float32)
b = tf.constant([2.0, 3.0], name='b', dtype=tf.float32)

result = a+b
'''

# 测试2：字符串小写
s = 'Python is beautiful!'
print(s[0:6].lower())


# 测试3：贪婪匹配与非贪婪匹配
# the examples of the greedy matching and non-greedy matching

# greedy matching
pattern1 = re.compile(r'.{2,5}')
match1 = pattern1.match('Hello world!')
if match1:
    print("greedy matching result is " + match1.group())

# non-greedy matching
pattern2 = re.compile(r'.{2,5}?')
match2 = pattern2.match('Hello world!')
if match2:
    print("non-greedy matching result is " + match2.group())


# 测试4：字典相似度的匹配
dict1 = {'key1': ['value1']}
dict2 = {'key1': ['value1']}
dict3 = {'key2': True}
dict4 = {'key2': False}
if dict1 == dict2:
    print('True')
print(dict1['key1'])

if dict3 != dict4:
    print('True')