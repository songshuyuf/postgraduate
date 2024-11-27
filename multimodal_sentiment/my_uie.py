from paddlenlp import Taskflow

def text_emo_analysize(text):
    senta = Taskflow("sentiment_analysis", model="skep_ernie_1.0_large_ch")
    result = senta(text)[0]  # 获取第一个结果
    # 创建两个输出框
    output_1 = {"文本": result['text'], "情感": "积极" if result['label'] == "positive" else "消极"}
    output_2 = {"概率": result['score']}
    
    # 打印两个输出框
    print(output_1)
    print(output_2)
    
    return output_1, output_2

