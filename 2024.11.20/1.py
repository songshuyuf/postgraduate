from paddlenlp import Taskflow
dialogue = Taskflow("dialogue")
#输入exit 可退出交互模式
dialogue.interactive_mode(max_turn=3)
