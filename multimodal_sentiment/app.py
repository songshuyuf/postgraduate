"""
File: app.py
Author: Elena Ryumina and Dmitry Ryumin
Description: Description: Main application file for Facial_Expression_Recognition.
             The file defines the Gradio interface, sets up the main blocks,
             and includes event handlers for various components.
License: MIT License
"""

import gradio as gr
import my_uie
# Importing necessary components for the Gradio app
from app.description import DESCRIPTION_STATIC, DESCRIPTION_DYNAMIC
from app.authors import AUTHORS
from app.app_utils import preprocess_image_and_predict, preprocess_video_and_predict
import voice_sentiment_analysize


def text_emo_analysize(text):
    text_outcome1,text_outcome2 = my_uie.text_emo_analysize(text)
    return text_outcome1,text_outcome2


def voice_emo_analysize(voice):
    voice_outcome1,voice_outcome2 = voice_sentiment_analysize.voice_sentiment(voice)
    return voice_outcome1,voice_outcome2


def clear_static_info():
    return (
        gr.Image(value=None, type="pil"),
        gr.Image(value=None, scale=1, elem_classes="dl5"),
        gr.Image(value=None, scale=1, elem_classes="dl2"),
        gr.Label(value=None, num_top_classes=3, scale=1, elem_classes="dl3"),
    )


def clear_dynamic_info():
    return (
        gr.Video(value=None),
        gr.Video(value=None),
        gr.Video(value=None),
        gr.Video(value=None),
        gr.Plot(value=None),
    )

with gr.Blocks(css="app.css") as demo:
    with gr.Tab("情感分析系统"):
        gr.Markdown(value=DESCRIPTION_DYNAMIC)
        with gr.Row():
            with gr.Column(scale=2):
                input_video = gr.Video(elem_classes="video1")
                with gr.Row():
                    clear_btn_dynamic = gr.Button(
                        value="清除", interactive=True, scale=1
                    )
                    submit_dynamic = gr.Button(
                        value="提交", interactive=True, scale=1, elem_classes="submit"
                    )
            with gr.Column(scale=2, elem_classes="dl4"):
                with gr.Row():
                    output_video = gr.Video(label="Original video", scale=1, elem_classes="video2", visible=False)
                    output_face = gr.Video(label="Pre-processed video", scale=1, elem_classes="video3", visible=False)
                    output_heatmaps = gr.Video(label="Heatmaps", scale=1, elem_classes="video4", visible=False)
                output_statistics = gr.Plot(label="情感数据", elem_classes="stat")
        gr.Examples(
            ["videos/video1.mp4",
            "videos/video2.mp4",
            ],
            [input_video],
        )
    with gr.Row("文本情感分析"):
        with gr.Column():
            gr.Markdown("文本情感分析")
            text_input = gr.Textbox(lines=2, placeholder='在这里输入文本')
            text_submit_button = gr.Button("提交文本情感分析")
        with gr.Column():
        # 增加两个输出框
            text_output_1 = gr.Textbox(label="文本情感")
            text_output_2 = gr.Textbox(label="情感概率")

    # 让按钮处理两个输出
    text_submit_button.click(text_emo_analysize, inputs=text_input, outputs=[text_output_1, text_output_2])

#新建第三栏语音情感分析
    with gr.Row("音频情感分析"):
        with gr.Column():
            gr.Markdown("录音或上传音频文件进行情感分析")
            audio_input = gr.Audio( type="filepath", label="录音或上传音频文件")
            audio_submit_button = gr.Button("提交音频情感分析")
        with gr.Column():
            #音频两个输出框
            audio_output_1 = gr.Textbox(label="音频情感")
            audio_output_2 = gr.Textbox(label="情感概率")
        # 处理音频输入并输出结果
    audio_submit_button.click(voice_emo_analysize, inputs=audio_input, outputs=[audio_output_1, audio_output_2])



    submit_dynamic.click(
        fn=preprocess_video_and_predict,
        inputs=input_video,
        outputs=[
            output_video,
            output_face,
            output_heatmaps, 
            output_statistics
        ],
        queue=True,
    )
    clear_btn_dynamic.click(
        fn=clear_dynamic_info,
        inputs=[],
        outputs=[
            input_video,
            output_video,
            output_face,
            output_heatmaps, 
            output_statistics
        ],
        queue=True,
    )

if __name__ == "__main__":
    demo.queue(api_open=False).launch(share=False)



















   # with gr.Tab("Static App"):
    #    gr.Markdown(value=DESCRIPTION_STATIC)
     #   with gr.Row():
      #      with gr.Column(scale=2, elem_classes="dl1"):
       #         input_image = gr.Image(label="Original image", type="pil")
        #        with gr.Row():
         #           clear_btn = gr.Button(
          #              value="Clear", interactive=True, scale=1, elem_classes="clear"
           #         )
            ##           value="Submit", interactive=True, scale=1, elem_classes="submit"
              #      )
            #with gr.Column(scale=1, elem_classes="dl4"):
             #   with gr.Row():
              #      output_image = gr.Image(label="Face", scale=1, elem_classes="dl5")
               #     output_heatmap = gr.Image(label="Heatmap", scale=1, elem_classes="dl2")
               # output_label = gr.Label(num_top_classes=3, scale=1, elem_classes="dl3")
       # gr.Examples(
        #    [
         #       "images/fig7.jpg",
          #      "images/fig1.jpg",
           #     "images/fig2.jpg",
            #    "images/fig3.jpg",
             #   "images/fig4.jpg",
              # "images/fig5.jpg",
               # "images/fig6.jpg",
           # ],
           # [input_image],
       # )
    #with gr.Tab("Authors"):
     #   gr.Markdown(value=AUTHORS)

    #submit.click(
     #   fn=preprocess_image_and_predict,
      #  inputs=[input_image],
       # outputs=[output_image, output_heatmap, output_label],
       # queue=True,
   # )
    #clear_btn.click(
     #   fn=clear_static_info,
      #  inputs=[],
       # outputs=[input_image, output_image, output_heatmap, output_label],
     #   queue=True,
    #)
