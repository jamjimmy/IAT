from zhipuai import ZhipuAI
client = ZhipuAI(api_key="c9bcc8c8eae1267231a9abc902d7c43d.PdJbzq90JNC8ngNT")  # 请填写您自己的APIKey
response = client.chat.completions.create(
    model="glm-4-plus",  # 请填写您要调用的模型名称
    messages=[
        {"role": "user", "content": "帮我设计一辆现代的，充满科技感的汽车内饰设计。这里给你一个例子:'Photograph of a futuristic,minimalist car interior at sunset. The car features a sleek,white and yellow leather dashboard,digital display,and a quilted patterned door trim. The spacious,modern design includes a central console with a gear shifter and cup holders. The background shows a serene lake and mountains under a golden sky.'。请你发挥想象力，设计的尽量好看。用英文回复我。长度类似示例，并且不要有其他多余的东西。用列表的形式返回给我5个。记住千万不要用有多余的内容"},
    ],
)
print(response)
print(response["choices"][0].message.content.strip())