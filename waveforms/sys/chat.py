import pickle
import time
from datetime import datetime
from pathlib import Path

import openai
from IPython import get_ipython
from IPython.display import Markdown, display

DEFAULT_SYSTEM_PROMPT = 'You are a helpful assistant. Respond using markdown.'
DEFAULT_MODEL = "gpt-3.5-turbo"


class Completion():

    def __init__(self,
                 system_prompt=DEFAULT_SYSTEM_PROMPT,
                 model=DEFAULT_MODEL):
        self.messages = [{"role": "system", "content": system_prompt}]
        self.title = 'untitled'
        self.last_time = datetime.now()
        self.completion = None
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.model = model

    def make_title(self):

        text = [
            f'{d["role"]} :\n"""\n{d["content"]}\n"""'
            for d in self.messages[1:]
        ]

        messages = [{
            "role": "system",
            "content": 'You are a helpful assistant.'
        }, {
            'role':
            "user",
            'content': ("总结以下对话的内容并为其取个标题以概括对话的内容，标题长度不超过100个字符。"
                        "不得包含`?:*,<>\\/` 等不能用于文件路径的字符。"
                        "返回的结果除了标题本身，不要包含额外的内容，省略结尾的句号。\n" + '\n\n'.join(text))
        }]
        completion = openai.ChatCompletion.create(model=self.model,
                                                  messages=messages)
        content = completion.choices[0].message['content']
        return f"{time.strftime('%Y%m%d%H%M')} {content}"

    def say(self, msg):
        self.last_time = datetime.now()
        self.messages.append({"role": "user", "content": msg})
        self.completion = openai.ChatCompletion.create(model=self.model,
                                                       messages=self.messages)
        self.total_tokens += self.completion.usage.total_tokens
        self.completion_tokens += self.completion.usage.completion_tokens
        self.prompt_tokens += self.completion.usage.prompt_tokens
        message = self.completion.choices[0].message
        self.messages.append({
            "role": message['role'],
            "content": message['content']
        })
        return message['content']

    def save(self):
        if self.title == 'untitled':
            self.title = self.make_title()

        filepath = Path.home() / 'chatGPT' / f"{self.title}.completion"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)


ipy = get_ipython()

current_completion = Completion()


def chat(line, cell):
    global current_completion
    if line:
        current_completion.save()
        current_completion = Completion()
    if line in ['end', 'save']:
        return
    content = current_completion.say(cell)
    display(Markdown(content))
    ipy.set_next_input('%%chat\n')


def autosave_completion():
    if (datetime.now() - current_completion.last_time).seconds > 300 and len(
            current_completion.messages) >= 3:
        current_completion.save()
    elif len(current_completion.messages) > 7:
        current_completion.save()


ipy.register_magic_function(chat, 'cell', magic_name='chat')
ipy.events.register('post_run_cell', autosave_completion)
