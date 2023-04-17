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

    def __init__(self):
        self.messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
        self.title = 'untitled'
        self.last_time = datetime.now()
        self.completion = None
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

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
                        "返回的结果除了标题本身，不要包含额外的内容，省略结尾的句号。\n" + '\n\n'.join(text))
        }]
        completion = openai.ChatCompletion.create(model=DEFAULT_MODEL,
                                                  messages=messages)
        content = completion.choices[0].message['content']
        return f"{content} {time.asctime()}"

    def say(self, msg):
        self.last_time = datetime.now()
        self.messages.append({"role": "user", "content": msg})
        self.completion = openai.ChatCompletion.create(model=DEFAULT_MODEL,
                                                       messages=self.messages)
        self.total_tokens += self.completion.usage.total_tokens
        self.completion_tokens += self.completion.usage.completion_tokens
        self.prompt_tokens += self.completion.usage.prompt_tokens
        content = self.completion.choices[0].message['content']
        self.messages.append({"role": "assistant", "content": content})
        return content

    def save(self):
        if self.title == 'untitled':
            self.title = self.make_title()

        filepath = Path.home() / 'chatGPT' / f"{self.title}.completion"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.messages, f)


ipy = get_ipython()

current_completion = Completion()


def chat(line, cell):
    global current_completion
    if line:
        current_completion.save()
        current_completion = Completion()
    if line in ['end']:
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
