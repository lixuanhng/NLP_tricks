# 汉字数字-阿拉伯数字字典
CN_NUM = {'〇': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
          '六': 6, '七': 7, '八': 8, '九': 9, '零': 0, '壹': 1,
          '贰': 2, '叁': 3, '肆': 4, '伍': 5, '陆': 6, '柒': 7,
          '捌': 8, '玖': 9, '貮': 2, '两': 2, }
CN_UNIT = {'十': 10, '拾': 10, '百': 100, '佰': 100, '千': 1000,
           '仟': 1000, '万': 10000, '萬': 10000}


# 将汉字数字转化为阿拉伯数字
def chinese_to_arabic(str_cn: str) -> int:
    # import pdb
    # pdb.set_trace()

    unit = 0  # 单位
    ldig = []  # 存放最终数据
    # 这一部分循环就是将所有汉字转化为一个个数字
    for cndig in reversed(str_cn):
        if cndig in CN_UNIT:
            unit = CN_UNIT.get(cndig)
            if unit == 10000:
                ldig.append(unit)
                unit = 1
        else:
            dig = CN_NUM.get(cndig)
            if unit:
                dig = dig * unit
                unit = 0
            ldig.append(dig)
    if unit == 10:
        ldig.append(10)
    val, tmp = 0, 0
    # 将一个个数字拼成完整数字
    for x in reversed(ldig):
        if x == 10000:
            val += tmp * x
            tmp = 0
        else:
            tmp += x
    val += tmp
    return val