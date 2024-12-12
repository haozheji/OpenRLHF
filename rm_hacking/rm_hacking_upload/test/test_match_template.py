import re

text = """</s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s> <|user|>
Hello, how are you?</s>
<|assistant|>
I'm doing great. How can I help you today?</s>
</s>"""

template = """<|user|>
{}</s>
<|assistant|>
"""

prompts = template.split("{}")
_, user_assistant = text.split(prompts[0])
user, assistant = user_assistant.split(prompts[1]).replace()
print(user)
print(assistant)