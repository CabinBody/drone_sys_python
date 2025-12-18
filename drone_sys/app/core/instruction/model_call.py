import os
from openai import OpenAI


def call_qwen_model(prompt):
    """
    调用Qwen模型并返回完整响应及token使用情况

    Args:
        prompt (str): 用户输入的提示词

    Returns:
        tuple: 包含模型生成的完整响应内容和使用的token数量
    """
    # 初始化客户端
    client = OpenAI(
        # 建议通过环境变量配置API Key，避免硬编码。
        api_key=os.getenv("DASHSCOPE_API_KEY", "sk-b462a06d308e4a259a94078edc07e970"),
        # API Key与地域强绑定，请确保base_url与API Key的地域一致。
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 发起非流式请求
    completion = client.chat.completions.create(
        model="qwen3-8b-ft-202512162104-c7c8",
        messages=[
            {"role": "user", "content": prompt}
        ],
        extra_body={"enable_thinking": False}
    )

    # 提取并返回完整响应内容和使用的token数量
    response_content = completion.choices[0].message.content
    total_tokens = completion.usage.total_tokens

    return response_content, total_tokens

if __name__ == "__main__":
    prompt = """
    你是低空空域安全指挥决策模型。
请根据态势输入，给出最终指挥指令及理由。

输出要求：
- 第一行：UAV_101，请 悬停 / 原地降落 / 返航 / 正常飞行（四选一）
- 第二行：理由:≤50字，说明权衡后为何选择该指令

【态势输入】
uav_id: UAV_101
battery_pct: 36

nearest_sep_m: 32
any_route_intersect: 1

inside_zone: 0
over_limit_alt: 1

low_battery: 0
critical_battery: 0


    """
    #循环10次
    for i in range(10):
        response, tokens = call_qwen_model(prompt)
        print(f"Response: {response}")
        print(f"Tokens used: {tokens}")

